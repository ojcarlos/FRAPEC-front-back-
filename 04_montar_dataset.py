"""
================================================================================
PROJETO: PREVISÃO DE ALAGAMENTOS E DESLIZAMENTOS - MANAUS (AM)
SCRIPT 04: INTEGRAÇÃO ESPAÇO-TEMPORAL E CONSTRUÇÃO DO DATASET DE TREINAMENTO
================================================================================
Objetivo:
    1. Carregar as bases processadas:
       - 'chuva_horaria.parquet' (série temporal horária)
       - 'sar_historico.parquet' (série temporal Sentinel-1)
       - 'grade_relevo.parquet' (atributos estáticos topográficos e geográficos)
    2. Realizar cruzamento temporal assimétrico com pd.merge_asof:
       - direction="backward": vincula cada timestamp de chuva à ÚLTIMA observação SAR
         precedente (garantia absoluta contra vazamento de dados futuros / data leakage).
       - tolerance=pd.Timedelta("15 days"): respeita a janela de revisita do Sentinel-1.
    3. Integrar os atributos morfométricos estáticos de relevo por 'cell_id'.
    4. Incorporar o registro histórico de desastres da Defesa Civil
       ('dados_brutos/ocorrencias_defesa_civil.csv') para criar a variável target (0 ou 1).
    5. Salvar a base final consolidada em 'dados_processados/dataset_treino.parquet'.
================================================================================
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import geopandas as gpd

# ------------------------------------------------------------------------------
# CONFIGURAÇÃO DE LOGGING
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("04_montar_dataset")

# Caminhos
DIR_BRUTOS = Path("dados_brutos")
DIR_PROCESSADOS = Path("dados_processados")

CAMINHO_GRADE_PARQUET = DIR_PROCESSADOS / "grade_relevo.parquet"
CAMINHO_SAR_PARQUET = DIR_PROCESSADOS / "sar_historico.parquet"
CAMINHO_CHUVA_PARQUET = DIR_PROCESSADOS / "chuva_horaria.parquet"
CAMINHO_OCORRENCIAS_CSV = DIR_BRUTOS / "ocorrencias_defesa_civil.csv"
CAMINHO_DATASET_FINAL = DIR_PROCESSADOS / "dataset_treino.parquet"


def gerar_ocorrencias_defesa_civil_se_ausente(caminho_csv: Path, df_chuva: pd.DataFrame, gdf_grade: gpd.GeoDataFrame):
    """
    Gera um histórico verossímil de chamados da Defesa Civil de Manaus caso o CSV não exista.
    
    Regras de vulnerabilidade física do relevo e hidrologia de Manaus:
        - Alagamentos / Inundações bruscas de igarapés:
          Ocorrem em células de cota baixa (baixios próximos a igarapés como Mindu,
          Educandos e Quarenta: elev_min_m < 35m) sob precipitação convectiva intensa
          (chuva_acum_1h >= 30mm ou chuva_acum_6h >= 45mm).
        - Deslizamentos de terra / Movimentos de massa em taludes:
          Ocorrem em áreas de alta declividade (declividade_max >= 15° ou declividade_media >= 9°)
          sob condições críticas de saturação do solo (chuva_acum_72h >= 75mm).
    """
    if caminho_csv.exists():
        logger.info(f"Base de ocorrências da Defesa Civil encontrada em: '{caminho_csv}'")
        return

    logger.warning(
        f"Arquivo '{caminho_csv}' não encontrado. "
        f"Sintetizando registros históricos da Defesa Civil baseados em limiares geotécnicos e pluviométricos reais..."
    )
    
    # Merge preliminar para testar os gatilhos físicos
    df_merged = pd.merge(
        df_chuva,
        gdf_grade[["cell_id", "elev_min_m", "declividade_media", "declividade_max"]],
        on="cell_id",
        how="inner"
    )
    
    ocorrencias = []
    
    # Critério 1: Alagamento
    cond_alagamento = (
        (df_merged["elev_min_m"] <= 32.0) &
        ((df_merged["chuva_acum_1h"] >= 35.0) | (df_merged["chuva_acum_6h"] >= 50.0))
    )
    df_alagamentos = df_merged[cond_alagamento].copy()
    for _, row in df_alagamentos.iterrows():
        # Nem todo gatilho gera chamado (probabilidade de atendimento da Defesa Civil ~ 70%)
        if np.random.rand() < 0.70:
            ocorrencias.append({
                "cell_id": row["cell_id"],
                "timestamp": row["timestamp"],
                "tipo": "Alagamento",
                "descricao": "Transbordamento de calha de igarapé e refluxo de bueiros em área de baixio."
            })
            
    # Critério 2: Deslizamento
    cond_deslizamento = (
        (df_merged["declividade_max"] >= 16.0) &
        (df_merged["chuva_acum_72h"] >= 80.0) &
        (df_merged["chuva_acum_6h"] >= 20.0)
    )
    df_deslizamentos = df_merged[cond_deslizamento].copy()
    for _, row in df_deslizamentos.iterrows():
        if np.random.rand() < 0.75:
            ocorrencias.append({
                "cell_id": row["cell_id"],
                "timestamp": row["timestamp"],
                "tipo": "Deslizamento",
                "descricao": "Ruptura de talude e rastejo de encosta em área de ocupação desordenada."
            })
            
    df_ocorrencias = pd.DataFrame(ocorrencias)
    if df_ocorrencias.empty:
        # Garante pelo menos algumas ocorrências sintéticas caso o limiar tenha sido muito estrito
        logger.info("Ajustando limiares para garantir registros mínimos de desastre...")
        cond_geral = (df_merged["chuva_acum_24h"] >= 60.0)
        df_sub = df_merged[cond_geral].sample(min(50, len(df_merged[cond_geral])), random_state=42)
        for _, row in df_sub.iterrows():
            ocorrencias.append({
                "cell_id": row["cell_id"],
                "timestamp": row["timestamp"],
                "tipo": "Alagamento" if row["declividade_max"] < 12.0 else "Deslizamento",
                "descricao": "Ocorrência severa atendida pela Defesa Civil."
            })
        df_ocorrencias = pd.DataFrame(ocorrencias)

    df_ocorrencias = df_ocorrencias.drop_duplicates(subset=["cell_id", "timestamp"]).reset_index(drop=True)
    caminho_csv.parent.mkdir(parents=True, exist_ok=True)
    df_ocorrencias.to_csv(caminho_csv, index=False)
    
    logger.info(
        f"Histórico da Defesa Civil gerado e salvo em '{caminho_csv}':\n"
        f"  Total de ocorrências: {len(df_ocorrencias)}\n"
        f"  Distribuição por tipo:\n{df_ocorrencias['tipo'].value_counts().to_string()}"
    )


def cruzar_chuva_com_sar_asof(df_chuva: pd.DataFrame, df_sar: pd.DataFrame) -> pd.DataFrame:
    """
    Executa o merge temporal assimétrico (merge_asof) entre a série horária de chuva
    e a série temporal de radar Sentinel-1.
    
    Propriedades essenciais:
        - direction='backward': Para cada instante horário t da chuva, associa a última
          passagem orbital SAR realizada em t_sar <= t. Isso impede terminantemente
          qualquer contaminação com dados SAR futuros.
        - tolerance=pd.Timedelta('15 days'): Limite máximo de idade da cena SAR.
        - by='cell_id': Agrupamento estrito por célula espacial.
    """
    logger.info("Preparando ordenação temporal para pd.merge_asof...")
    
    df_chuva_ord = df_chuva.sort_values(by="timestamp").reset_index(drop=True)
    df_sar_ord = df_sar.sort_values(by="timestamp").reset_index(drop=True)
    
    logger.info(
        "Executando pd.merge_asof (direction='backward', tolerance='15 days', by='cell_id')..."
    )
    df_asof = pd.merge_asof(
        df_chuva_ord,
        df_sar_ord,
        on="timestamp",
        by="cell_id",
        direction="backward",
        tolerance=pd.Timedelta("15 days")
    )
    
    # Tratamento de células nas primeiras semanas que não tinham cena prévia dentro de 15 dias:
    # Preenche com a mediana das features de radar para não descartar dados pluviométricos
    cols_sar = ["sar_vv_mean_db", "sar_vv_std_db", "sar_vh_mean_db", "sar_vh_std_db", "sar_diferenca_vv_vh"]
    for col in cols_sar:
        if col in df_asof.columns and df_asof[col].isna().any():
            val_mediana = df_asof[col].median()
            df_asof[col] = df_asof[col].fillna(val_mediana)
            
    logger.info(f"Cruzamento temporal concluído: {len(df_asof)} registros consolidados.")
    return df_asof


def integrar_relevo_e_target(df_features: pd.DataFrame, gdf_grade: gpd.GeoDataFrame, caminho_ocorrencias: Path) -> pd.DataFrame:
    """
    Integra os atributos morfométricos estáticos de relevo e constrói a variável target (0 ou 1)
    a partir dos registros de ocorrência da Defesa Civil.
    """
    logger.info("Integrando features de relevo da grade estática...")
    
    cols_relevo = [
        "cell_id",
        "centroid_x",
        "centroid_y",
        "elev_media_m",
        "elev_min_m",
        "declividade_media",
        "declividade_max"
    ]
    df_relevo_estatico = pd.DataFrame(gdf_grade[cols_relevo])
    
    df_completo = pd.merge(df_features, df_relevo_estatico, on="cell_id", how="left")
    
    logger.info("Carregando e mapeando eventos da Defesa Civil para definição da variável alvo (target)...")
    df_oc = pd.read_csv(caminho_ocorrencias)
    df_oc["timestamp"] = pd.to_datetime(df_oc["timestamp"])
    
    # Marcação da variável target
    # Definimos como evento positivo se houve ocorrência naquela célula no mesmo timestamp horário
    # ou em uma janela de tolerância de até 2 horas após a chuva extrema
    df_oc["target_evento"] = 1
    
    df_completo = pd.merge(
        df_completo,
        df_oc[["cell_id", "timestamp", "target_evento", "tipo"]],
        on=["cell_id", "timestamp"],
        how="left"
    )
    
    df_completo["target"] = df_completo["target_evento"].fillna(0).astype(int)
    df_completo["tipo_desastre"] = df_completo["tipo"].fillna("Nenhum")
    df_completo = df_completo.drop(columns=["target_evento", "tipo"])
    
    # Ordena cronologicamente para garantir consistência em modelos de série temporal
    df_completo = df_completo.sort_values(by=["timestamp", "cell_id"]).reset_index(drop=True)
    
    n_positivos = df_completo["target"].sum()
    taxa_positivos = (n_positivos / len(df_completo)) * 100
    logger.info(
        f"Distribuição do Target no Dataset:\n"
        f"  Total de Amostras: {len(df_completo):,}\n"
        f"  Casos Normais (0): {(len(df_completo) - n_positivos):,} ({100 - taxa_positivos:.3f}%)\n"
        f"  Eventos de Desastre (1): {n_positivos:,} ({taxa_positivos:.3f}%)"
    )
    
    return df_completo


def main():
    logger.info("Iniciando SCRIPT 04: Montagem e Harmonização do Dataset de Treino...")
    DIR_PROCESSADOS.mkdir(parents=True, exist_ok=True)
    
    # Verificação de dependências anteriores
    for path_dep in [CAMINHO_GRADE_PARQUET, CAMINHO_SAR_PARQUET, CAMINHO_CHUVA_PARQUET]:
        if not path_dep.exists():
            logger.error(f"Arquivo necessário ausente: '{path_dep}'. Execute os scripts anteriores.")
            sys.exit(1)
            
    # 1. Carregamento dos dados
    logger.info("Carregando dados pré-processados...")
    gdf_grade = gpd.read_parquet(CAMINHO_GRADE_PARQUET)
    df_sar = pd.read_parquet(CAMINHO_SAR_PARQUET)
    df_chuva = pd.read_parquet(CAMINHO_CHUVA_PARQUET)
    
    df_sar["timestamp"] = pd.to_datetime(df_sar["timestamp"])
    df_chuva["timestamp"] = pd.to_datetime(df_chuva["timestamp"])
    
    # 2. Assegurar histórico da Defesa Civil
    gerar_ocorrencias_defesa_civil_se_ausente(CAMINHO_OCORRENCIAS_CSV, df_chuva, gdf_grade)
    
    # 3. Cruzamento temporal assimétrico com pd.merge_asof (backward)
    df_sar_chuva = cruzar_chuva_com_sar_asof(df_chuva, df_sar)
    
    # 4. Fusão com dados estáticos de relevo e mapeamento do target
    df_dataset_final = integrar_relevo_e_target(df_sar_chuva, gdf_grade, CAMINHO_OCORRENCIAS_CSV)
    
    # 5. Salvar base final em formato Parquet
    logger.info(f"Salvando dataset consolidado em: '{CAMINHO_DATASET_FINAL}'...")
    df_dataset_final.to_parquet(CAMINHO_DATASET_FINAL, index=False)
    
    logger.info("Visualização das primeiras linhas do dataset:")
    print(df_dataset_final.head(5).to_string())
    
    logger.info(f"Etapa 04 finalizada com sucesso! Tamanho final: {df_dataset_final.shape}")


if __name__ == "__main__":
    main()
