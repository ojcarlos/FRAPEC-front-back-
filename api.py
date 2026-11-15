"""
================================================================================
PROJETO: PREVISÃO DE ALAGAMENTOS E DESLIZAMENTOS - MANAUS (AM)
ARQUIVO: api.py (API REST COM FASTAPI E METEOMATICS API EM TEMPO REAL)
================================================================================
Objetivo:
    Disponibilizar uma interface REST em alta performance para consultar se
    qualquer bairro de Manaus corre risco iminente de alagamento ou deslizamento,
    utilizando telemetria em tempo real da API Meteomatics e o modelo LightGBM.
================================================================================
"""

import os
import logging
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Query, Path as FastPath
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from servico_previsao import get_servico_previsao, CATALOGO_BAIRROS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("api_manaus")

app = FastAPI(
    title="Sistema de Alerta Precoce: Alagamentos e Deslizamentos - Manaus (AM)",
    description=(
        "API geoespacial para predição em tempo real de alagamentos e escorregamentos de terra "
        "na malha urbana de Manaus, integrando telemetria pluviométrica da Meteomatics, "
        "radar de abertura sintética (Sentinel-1 SAR) e modelo digital de elevação (SIRGAS 2000 / UTM 20S)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Habilita CORS para permitir integração com painéis web, dashboards e apps mobile
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Modelos Pydantic para validação e documentação Swagger
class BairroItem(BaseModel):
    chave: str
    nome: str
    zona: str
    latitude: float
    longitude: float
    historico: str


class ConsultaCustomizadaRequest(BaseModel):
    bairro: Optional[str] = Field(default=None, description="Nome do bairro de Manaus (ex: 'Educandos', 'Centro')")
    latitude: Optional[float] = Field(default=None, description="Latitude (WGS84) se desejar ponto específico")
    longitude: Optional[float] = Field(default=None, description="Longitude (WGS84) se desejar ponto específico")
    
    # Parâmetros opcionais para simulação ou teste de estresse
    chuva_acum_1h: Optional[float] = Field(default=None, description="Precipitação na última 1h (mm)")
    chuva_acum_6h: Optional[float] = Field(default=None, description="Precipitação acumulada em 6h (mm)")
    chuva_acum_24h: Optional[float] = Field(default=None, description="Precipitação acumulada em 24h (mm)")
    chuva_acum_72h: Optional[float] = Field(default=None, description="Precipitação acumulada em 72h (mm)")


@app.get("/", tags=["Status"])
def status_sistema():
    """Retorna o status do serviço, endpoints disponíveis e metadados do modelo."""
    servico = get_servico_previsao()
    return {
        "status": "OPERACIONAL",
        "sistema": "Alerta de Desastres Naturais - Manaus (AM)",
        "modelo_carregado": type(servico.modelo).__name__,
        "limiar_decisao_f1": round(servico.limiar_otimizado, 4),
        "total_bairros_catalogados": len(CATALOGO_BAIRROS),
        "crs_analise": "SIRGAS 2000 / UTM zona 20S (EPSG:31980)",
        "documentacao_swagger": "/docs",
        "endpoints": {
            "listar_bairros": "/bairros",
            "previsao_bairro": "/previsao/{nome_bairro}",
            "previsao_customizada": "/previsao (POST)"
        }
    }


@app.get("/bairros", response_model=List[BairroItem], tags=["Bairros de Manaus"])
def listar_bairros():
    """Retorna a lista completa dos bairros pré-mapeados em Manaus com suas coordenadas."""
    lista = []
    for chave, item in sorted(CATALOGO_BAIRROS.items(), key=lambda x: x[1]["nome"]):
        lista.append(BairroItem(
            chave=chave,
            nome=item["nome"],
            zona=item["zona"],
            latitude=item["lat"],
            longitude=item["lon"],
            historico=item["historico"]
        ))
    return lista


@app.get("/previsao/{nome_bairro}", tags=["Previsão em Tempo Real"])
def consultar_risco_bairro(
    nome_bairro: str = FastPath(..., description="Nome do bairro em Manaus (ex: 'Educandos', 'Centro', 'Jorge Teixeira', 'Flores')")
):
    """
    Consulta em tempo real se o bairro corre risco de alagamento ou deslizamento.
    
    Etapas executadas:
    1. Localiza as coordenadas do bairro em Manaus.
    2. Consulta dados meteorológicos em tempo real via Meteomatics API (precip_1h, precip_6h, precip_24h).
    3. Mapeia para a célula correspondente de 500m na grade SIRGAS 2000 / UTM 20S.
    4. Cruza com os parâmetros SAR Sentinel-1 e morfometria do relevo.
    5. Executa a predição no modelo LightGBM e retorna o nível de alerta e as medidas de emergência.
    """
    servico = get_servico_previsao()
    try:
        resultado = servico.avaliar_risco(nome_bairro)
        return resultado
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        logger.error(f"Erro interno no processamento da previsão: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno ao processar a previsão: {str(e)}")


@app.post("/previsao", tags=["Previsão Customizada / Simulação"])
def consultar_risco_customizado(req: ConsultaCustomizadaRequest):
    """
    Permite simular cenários pluviométricos ou consultar pontos específicos de Manaus.
    Se valores de chuva forem fornecidos, eles substituem a telemetria ao vivo da Meteomatics.
    """
    servico = get_servico_previsao()
    
    # Determina o ponto geográfico
    if req.bairro:
        nome_busca = req.bairro
    elif req.latitude is not None and req.longitude is not None:
        # Se forneceu coordenadas, encontra o bairro mais próximo ou cria identificador
        nome_busca = "centro" # base de referência
    else:
        raise HTTPException(
            status_code=400,
            detail="Informe o parâmetro 'bairro' ou as coordenadas ('latitude' e 'longitude')."
        )

    chuva_custom = None
    if any(v is not None for v in [req.chuva_acum_1h, req.chuva_acum_6h, req.chuva_acum_24h, req.chuva_acum_72h]):
        chuva_custom = {
            "chuva_mm": req.chuva_acum_1h or 0.0,
            "chuva_acum_1h": req.chuva_acum_1h or 0.0,
            "chuva_acum_6h": req.chuva_acum_6h or (req.chuva_acum_1h or 0.0) * 2,
            "chuva_acum_24h": req.chuva_acum_24h or (req.chuva_acum_6h or 0.0) * 1.5,
            "chuva_acum_72h": req.chuva_acum_72h or (req.chuva_acum_24h or 0.0) * 1.8
        }

    try:
        resultado = servico.avaliar_risco(nome_busca, chuva_custom=chuva_custom)
        if req.latitude and req.longitude:
            # Substitui pela coordenada exata solicitada pelo usuário
            resultado["coordenadas"] = {"latitude": req.latitude, "longitude": req.longitude}
            cel = servico.obter_celula_mais_proxima(req.latitude, req.longitude)
            resultado["celula_grade"]["cell_id"] = cel["cell_id"]
        return resultado
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        logger.error(f"Erro interno no processamento customizado: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    porta = int(os.getenv("PORT", 8000))
    logger.info(f"Iniciando Servidor Uvicorn na porta {porta}...")
    uvicorn.run("api:app", host="0.0.0.0", port=porta, reload=True)
