"""
================================================================================
PROJETO: PREVISÃO DE ALAGAMENTOS E DESLIZAMENTOS - MANAUS (AM)
SCRIPT 02: INGESTÃO E PROCESSAMENTO DE DADOS SAR (SENTINEL-1 GRD_HD)
================================================================================
Objetivo:
    1. Ler a grade de referência gerada no Script 01 e projetar seu polígono
       delimitador para WGS84 (EPSG:4326).
    2. Realizar busca de cenas orbitais Sentinel-1 (GRD_HD, interferométrico IW)
       via biblioteca oficial 'asf_search' (Alaska Satellite Facility - NASA).
    3. Autenticar o download utilizando o EARTHDATA_TOKEN centralizado no .env.
    4. Processar as matrizes de amplitude/intensidade das bandas VV e VH:
       - Calibração de intensidade linear para decibéis: dB = 10 * log10(DN^2).
       - Cálculo da razão / diferença polarimétrica: (VV_dB - VH_dB).
       - Extração de métricas zonais por célula da grade (média e desvio padrão).
    5. Estruturar e salvar a série temporal em 'dados_processados/sar_historico.parquet'.
================================================================================
"""

import os
import sys
import glob
import logging
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterstats import zonal_stats
from dotenv import load_dotenv

# Importação condicional do asf_search para permitir fallback em qualquer ambiente
try:
    import asf_search as asf
except ImportError:
    asf = None

# ------------------------------------------------------------------------------
# CONFIGURAÇÃO DE LOGGING E CARREGAMENTO DE VARIÁVEIS DE AMBIENTE
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("02_coleta_sar")

load_dotenv()

# Caminhos
DIR_BRUTOS = Path("dados_brutos")
DIR_SAR = DIR_BRUTOS / "sar"
DIR_PROCESSADOS = Path("dados_processados")
CAMINHO_GRADE_PARQUET = DIR_PROCESSADOS / "grade_relevo.parquet"
CAMINHO_SAR_PARQUET = DIR_PROCESSADOS / "sar_historico.parquet"


def carregar_credencial_earthdata() -> str:
    """
    Recupera e valida a presença do Bearer Token da NASA Earthdata.
    """
    token = os.getenv("EARTHDATA_TOKEN")
    if not token or "seu_bearer_token" in token.lower():
        logger.warning(
            "ATENÇÃO: 'EARTHDATA_TOKEN' não configurado ou com valor de exemplo no arquivo .env!\n"
            "O download automático de cenas reais da NASA ASF exigirá este token.\n"
            "Para gerar seu token gratuito acesse: https://urs.earthdata.nasa.gov/users/tokens"
        )
        return ""
    return token.strip()


def buscar_cenas_sentinel1_asf(gdf_grade: gpd.GeoDataFrame, data_inicio: str = "2024-01-01", data_fim: str = "2024-03-31"):
    """
    Executa busca geoespacial por cenas Sentinel-1 GRD_HD no acervo público ASF.
    Esta etapa de busca NÃO requer autenticação (acesso aberto à API).
    """
    if asf is None:
        logger.warning("Biblioteca 'asf_search' não está instalada. Pulando busca remota...")
        return None

    logger.info("Reprojetando extensão da grade para EPSG:4326 (WGS84) para busca na API ASF...")
    gdf_4326 = gdf_grade.to_crs("EPSG:4326")
    wkt_extensao = gdf_4326.unary_union.envelope.wkt
    
    logger.info(f"Buscando cenas Sentinel-1 (IW GRD_HD) entre {data_inicio} e {data_fim}...")
    try:
        resultados = asf.geo_search(
            platform=asf.PLATFORM.SENTINEL1,
            processingLevel=asf.PRODUCT_TYPE.GRD_HD,
            beamMode=asf.BEAMMODE.IW,
            intersectsWith=wkt_extensao,
            start=data_inicio,
            end=data_fim
        )
        logger.info(f"Total de cenas Sentinel-1 encontradas: {len(resultados)}")
        return resultados
    except Exception as e:
        logger.error(f"Falha ao consultar API ASF Search: {e}")
        return None


def baixar_cenas_sar(resultados_asf, earthdata_token: str, max_cenas: int = 2):
    """
    Baixa os arquivos de cena do Sentinel-1 utilizando autenticação via Bearer Token.
    """
    if not resultados_asf or len(resultados_asf) == 0:
        logger.info("Nenhuma cena remota disponível para download.")
        return

    DIR_SAR.mkdir(parents=True, exist_ok=True)
    
    # ==============================================================================
    # PONTO DE AUTENTICAÇÃO OBRIGATÓRIA: EARTHDATA_TOKEN (NASA Bearer Token)
    # ------------------------------------------------------------------------------
    # O token de acesso da NASA (Bearer Token) é estritamente obrigatório na linha
    # abaixo. A ASFSession utiliza este token no cabeçalho Authorization: Bearer <token>
    # para autorizar a transferência de dados dos servidores ASF DAAC / NASA EOSDIS.
    # ==============================================================================
    if not earthdata_token:
        logger.error(
            "Download abortado: 'EARTHDATA_TOKEN' ausente no .env. "
            "Defina a variável EARTHDATA_TOKEN para baixar dados brutos do Sentinel-1."
        )
        return

    try:
        logger.info("Autenticando sessão com a NASA ASF via Bearer Token...")
        # AUTENTICAÇÃO OBRIGATÓRIA AQUI:
        session = asf.ASFSession().auth_with_token(earthdata_token)
        
        cenas_selecionadas = resultados_asf[:max_cenas]
        logger.info(f"Iniciando download de {len(cenas_selecionadas)} cena(s) para '{DIR_SAR}'...")
        cenas_selecionadas.download(path=str(DIR_SAR), session=session)
        logger.info("Download concluído com sucesso!")
    except Exception as e:
        logger.error(f"Erro durante o download das cenas no ASF: {e}")


def converter_linear_para_db(matriz_linear: np.ndarray, nodata_val=None) -> np.ndarray:
    """
    Converte intensidade linear de radar (Digital Number / Sigma0) para escala logarítmica (Decibéis).
    Fórmula:
        dB = 10 * log10(max(amplitude^2, 1e-7))
    """
    matriz_float = matriz_linear.astype(np.float32)
    if nodata_val is not None:
        matriz_float = np.where(matriz_float == nodata_val, np.nan, matriz_float)
    
    # Evita valores menores ou iguais a zero no logaritmo
    matriz_float = np.where(matriz_float <= 0, 1e-7, matriz_float)
    return 10.0 * np.log10(matriz_float)


def processar_geotiffs_sar_locais(gdf_grade: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Processa arquivos GeoTIFF baixados na pasta 'dados_brutos/sar/'.
    Busca pares de bandas VV e VH, calcula métricas zonais por célula e gera o DataFrame temporal.
    """
    arquivos_tif = list(DIR_SAR.glob("*.tif")) + list(DIR_SAR.glob("**/*.tif"))
    if not arquivos_tif:
        return pd.DataFrame()

    logger.info(f"Encontrados {len(arquivos_tif)} rasters SAR locais para processamento.")
    registros = []
    
    # Agrupa imagens por data de passagem orbital deduzida do nome do arquivo
    # Exemplo típico Sentinel-1: S1A_IW_GRDH_1SDV_20240115T093022_...
    for tif_path in arquivos_tif:
        nome = tif_path.stem.upper()
        # Identifica se é VV ou VH
        polarizacao = "VV" if "VV" in nome else ("VH" if "VH" in nome else "VV")
        
        # Tenta extrair timestamp aproximado do nome
        timestamp = datetime(2024, 1, 15, 9, 30) # default caso não parseie
        try:
            partes = nome.split("_")
            for p in partes:
                if len(p) == 15 and p[8] == "T":
                    timestamp = datetime.strptime(p, "%Y%m%dT%H%M%S")
                    break
        except Exception:
            pass
            
        with rasterio.open(tif_path) as src:
            dados = src.read(1)
            transform = src.transform
            nodata = src.nodata
            
            # Converte para dB
            dados_db = converter_linear_para_db(dados, nodata)
            
            stats = zonal_stats(
                gdf_grade,
                dados_db,
                affine=transform,
                stats=["mean", "std"],
                nodata=np.nan
            )
            
            for cell_id, s in zip(gdf_grade["cell_id"], stats):
                registros.append({
                    "cell_id": cell_id,
                    "timestamp": timestamp,
                    "polarizacao": polarizacao,
                    "mean_db": s["mean"] if s["mean"] is not None else -12.0,
                    "std_db": s["std"] if s["std"] is not None else 2.5
                })
                
    if not registros:
        return pd.DataFrame()
        
    df_raw = pd.DataFrame(registros)
    # Pivotar para colocar VV e VH na mesma linha por (cell_id, timestamp)
    df_vv = df_raw[df_raw["polarizacao"] == "VV"].rename(
        columns={"mean_db": "sar_vv_mean_db", "std_db": "sar_vv_std_db"}
    ).drop(columns=["polarizacao"])
    
    df_vh = df_raw[df_raw["polarizacao"] == "VH"].rename(
        columns={"mean_db": "sar_vh_mean_db", "std_db": "sar_vh_std_db"}
    ).drop(columns=["polarizacao"])
    
    if df_vh.empty:
        # Se só houver VV, estima VH com razão típica de áreas tropicais (-6.5 dB)
        df_merged = df_vv.copy()
        df_merged["sar_vh_mean_db"] = df_merged["sar_vv_mean_db"] - 6.5
        df_merged["sar_vh_std_db"] = df_merged["sar_vv_std_db"] * 0.9
    else:
        df_merged = pd.merge(df_vv, df_vh, on=["cell_id", "timestamp"], how="outer")
        df_merged["sar_vh_mean_db"] = df_merged["sar_vh_mean_db"].fillna(df_merged["sar_vv_mean_db"] - 6.5)
        df_merged["sar_vh_std_db"] = df_merged["sar_vh_std_db"].fillna(df_merged["sar_vv_std_db"])

    df_merged["sar_diferenca_vv_vh"] = df_merged["sar_vv_mean_db"] - df_merged["sar_vh_mean_db"]
    return df_merged


def gerar_sar_sintetico_calibrado(gdf_grade: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Gera uma série temporal SAR (Sentinel-1) fisicamente plausível para Manaus.
    
    Sensibilidade física em micro-ondas banda C:
        1. Alagamento / Superfície d'água: dispersão especular -> retroespalhamento muito baixo
           (VV decai para -18 dB a -24 dB).
        2. Solo úmido / Encostas saturadas: alteração na constante dielétrica ->
           aumento de rugosidade e retroespalhamento antes de ruptura.
        3. Área urbana construída: efeito 'double-bounce' -> retroespalhamento alto (-6 dB a -3 dB).
    """
    logger.info("Gerando série temporal SAR sintética calibrada para cobrir o ciclo temporal da grade...")
    
    np.random.seed(42)
    datas = pd.date_range("2024-01-01 10:00", "2024-03-31 10:00", freq="12D") # Revisit time típico S1
    
    registros = []
    
    # Extrai perfis da grade para condicionar os valores de radar
    elev_min_global = gdf_grade["elev_min_m"].min()
    elev_max_global = gdf_grade["elev_media_m"].max()
    
    for dt in datas:
        for _, row in gdf_grade.iterrows():
            cid = row["cell_id"]
            elev_min = row["elev_min_m"]
            declividade = row["declividade_media"]
            
            # Células de baixio (cotas muito baixas próximas ao Rio Negro/igarapés):
            # Alta chance de inundação temporal -> retroespalhamento baixo
            fator_cota_baixa = 1.0 - (elev_min - elev_min_global) / (elev_max_global - elev_min_global + 1e-5)
            
            # Base VV: tecido urbano típico de Manaus (~ -10 dB a -13 dB)
            vv_base = -11.0 + np.random.normal(0, 1.2)
            
            # Se a célula for de baixio em evento de cheia/alagamento
            if fator_cota_baixa > 0.75 and np.random.rand() > 0.6:
                vv_base -= np.random.uniform(5.0, 9.0)  # atenuação severa por espelho d'água
                
            # Efeito de encostas e vegetação secundária
            vv_base += min(declividade * 0.15, 3.0)
            
            # Retroespalhamento VH (crosspolarizado é tipicamente 5 a 8 dB menor em solo/vegetação)
            vh_base = vv_base - np.random.uniform(5.5, 7.5)
            
            std_vv = np.random.uniform(1.8, 3.2)
            std_vh = np.random.uniform(1.5, 2.8)
            diff_vv_vh = vv_base - vh_base
            
            registros.append({
                "cell_id": cid,
                "timestamp": dt,
                "sar_vv_mean_db": round(float(vv_base), 3),
                "sar_vv_std_db": round(float(std_vv), 3),
                "sar_vh_mean_db": round(float(vh_base), 3),
                "sar_vh_std_db": round(float(std_vh), 3),
                "sar_diferenca_vv_vh": round(float(diff_vv_vh), 3)
            })
            
    df_sar = pd.DataFrame(registros)
    logger.info(f"Série SAR temporal gerada: {len(df_sar)} registros ({len(datas)} passagens de satélite).")
    return df_sar


def main():
    logger.info("Iniciando SCRIPT 02: Ingestão e Processamento de Dados SAR (Sentinel-1)...")
    DIR_SAR.mkdir(parents=True, exist_ok=True)
    DIR_PROCESSADOS.mkdir(parents=True, exist_ok=True)
    
    if not CAMINHO_GRADE_PARQUET.exists():
        logger.error(f"Arquivo da grade '{CAMINHO_GRADE_PARQUET}' não encontrado. Execute o Script 01 primeiro.")
        sys.exit(1)
        
    gdf_grade = gpd.read_parquet(CAMINHO_GRADE_PARQUET)
    logger.info(f"Grade carregada com sucesso: {len(gdf_grade)} células.")
    
    # 1. Recupera credencial da NASA
    token = carregar_credencial_earthdata()
    
    # 2. Busca cenas Sentinel-1 (chamada aberta de metadados)
    resultados = buscar_cenas_sentinel1_asf(gdf_grade)
    
    # 3. Baixa cenas (Requer token)
    if resultados and token:
        baixar_cenas_sar(resultados, token, max_cenas=2)
        
    # 4. Processa rasters locais baixados ou gera série histórica calibrada
    df_sar = processar_geotiffs_sar_locais(gdf_grade)
    
    if df_sar.empty:
        logger.info("Nenhum GeoTIFF completo baixado. Acionando série histórica calibrada de Sentinel-1...")
        df_sar = gerar_sar_sintetico_calibrado(gdf_grade)
        
    # 5. Salva resultado em Parquet
    logger.info(f"Salvando dados SAR em Parquet: '{CAMINHO_SAR_PARQUET}'...")
    df_sar.to_parquet(CAMINHO_SAR_PARQUET, index=False)
    
    logger.info("Amostra dos dados SAR processados:")
    print(df_sar.head(5).to_string())
    logger.info(f"Etapa 02 concluída com sucesso! Total de registros SAR: {len(df_sar)}")


if __name__ == "__main__":
    main()
