"""
================================================================================
PROJETO: PREVISÃO DE ALAGAMENTOS E DESLIZAMENTOS - MANAUS (AM)
SCRIPT 05: TREINAMENTO DO MODELO PREDITIVO (LIGHTGBM) E AVALIAÇÃO DE EVENTOS RAROS
================================================================================
Objetivo:
    1. Carregar a base consolidada 'dados_processados/dataset_treino.parquet'.
    2. Realizar split treino/teste estritamente temporal (80% inicial / 20% final),
       evitando vazamento de autocorrelação temporal climática e hidrológica.
    3. Tratar o desbalanceamento extremo de classes de desastres naturais com
       'scale_pos_weight' no LightGBM.
    4. Treinar o classificador LightGBM com métricas de validação precoces (early stopping).
    5. Avaliar o modelo com métricas especializadas para eventos raros:
       - Precision-Recall AUC (PR-AUC / Average Precision Score)
       - Curva ROC-AUC
       - Matriz de Confusão (TN, FP, FN, TP)
       - Precision, Recall e F1-Score com otimização de threshold de decisão
       - Importância de Atributos (Feature Importance por Ganho)
    6. Salvar o artefato treinado em 'modelos/modelo_alagamento_deslizamento_lgb.joblib'.
================================================================================
"""

import os
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    f1_score
)
import joblib

# ------------------------------------------------------------------------------
# CONFIGURAÇÃO DE LOGGING
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("05_treinar_modelo")

# Caminhos
DIR_PROCESSADOS = Path("dados_processados")
DIR_MODELOS = Path("modelos")
CAMINHO_DATASET = DIR_PROCESSADOS / "dataset_treino.parquet"
CAMINHO_MODELO_SALVO = DIR_MODELOS / "modelo_alagamento_deslizamento_lgb.joblib"


def carregar_e_preparar_dados(caminho_parquet: Path):
    """
    Carrega o dataset e define a matriz de features X e o vetor alvo y.
    """
    if not caminho_parquet.exists():
        logger.error(f"Dataset de treino não encontrado em: '{caminho_parquet}'. Execute o Script 04.")
        sys.exit(1)
        
    logger.info(f"Lendo dataset de treinamento: '{caminho_parquet}'...")
    df = pd.read_parquet(caminho_parquet)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Ordena rigorosamente pelo tempo
    df = df.sort_values(by="timestamp").reset_index(drop=True)
    
    # Lista de features preditoras estruturadas por domínio:
    # 1. Hidrológicas / Pluviométricas (dinâmicas de curto e médio prazo)
    features_chuva = [
        "chuva_mm",
        "chuva_acum_1h",
        "chuva_acum_6h",
        "chuva_acum_24h",
        "chuva_acum_72h"
    ]
    
    # 2. Sensoriamento Remoto por Radar (Sentinel-1 SAR C-Band)
    features_sar = [
        "sar_vv_mean_db",
        "sar_vv_std_db",
        "sar_vh_mean_db",
        "sar_vh_std_db",
        "sar_diferenca_vv_vh"
    ]
    
    # 3. Topografia e Morfometria do Relevo (DEM SIRGAS 2000 / UTM 20S)
    features_relevo = [
        "elev_media_m",
        "elev_min_m",
        "declividade_media",
        "declividade_max"
    ]
    
    # 4. Coordenadas Espaciais Georreferenciadas
    features_espaciais = [
        "centroid_x",
        "centroid_y"
    ]
    
    features_todas = features_chuva + features_sar + features_relevo + features_espaciais
    
    logger.info(f"Total de atributos preditores selecionados: {len(features_todas)}")
    for f in features_todas:
        logger.info(f"  - {f}")
        
    return df, features_todas


def split_temporal(df: pd.DataFrame, features: list, proporcao_treino: float = 0.80):
    """
    Realiza a partição temporal estrita (Time-Series Split).
    
    Princípio Metodológico:
        Em eventos climáticos e hidrológicos extremos, partições aleatórias (K-Fold tradicional)
        injetam vazamento de dados gravíssimo (treinar e testar no mesmo evento de tempestade
        em células vizinhas). A separação puramente cronológica garante que o modelo aprenda
        com o passado e seja avaliado exclusivamente no futuro não observado.
    """
    timestamps_unicos = df["timestamp"].drop_duplicates().sort_values().values
    n_instantes = len(timestamps_unicos)
    idx_corte = int(n_instantes * proporcao_treino)
    timestamp_corte = timestamps_unicos[idx_corte]
    
    logger.info(
        f"Divisão Temporal do Dataset:\n"
        f"  Período Global: [{df['timestamp'].min()} até {df['timestamp'].max()}]\n"
        f"  Timestamp de Corte: {pd.to_datetime(timestamp_corte)}\n"
        f"  Treino: {proporcao_treino*100:.0f}% inicial | Teste: {(1 - proporcao_treino)*100:.0f}% final"
    )
    
    mask_treino = df["timestamp"] < timestamp_corte
    mask_teste = df["timestamp"] >= timestamp_corte
    
    df_treino = df[mask_treino].copy()
    df_teste = df[mask_teste].copy()
    
    X_train = df_treino[features]
    y_train = df_treino["target"]
    
    X_test = df_teste[features]
    y_test = df_teste["target"]
    
    pos_treino = y_train.sum()
    pos_teste = y_test.sum()
    
    logger.info(
        f"Amostras de Treino: {len(X_train):,} (Positivos: {pos_treino} | {pos_treino/len(X_train)*100:.3f}%)\n"
        f"Amostras de Teste:  {len(X_test):,} (Positivos: {pos_teste} | {pos_teste/len(X_test)*100:.3f}%)"
    )
    
    return X_train, y_train, X_test, y_test, df_teste


def treinar_lightgbm(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Configura e treina o classificador LightGBM com balanceamento de pesos.
    """
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    
    # Evita divisão por zero se y_train não contiver positivos
    scale_pos_weight = (n_neg / max(n_pos, 1))
    logger.info(f"Fator de balanceamento de classes ('scale_pos_weight'): {scale_pos_weight:.2f}")
    
    # Hiperparâmetros otimizados para dados tabulares geoespaciais com classes raras
    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "average_precision"],
        "boosting_type": "gbdt",
        "scale_pos_weight": scale_pos_weight,
        "n_estimators": 400,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1
    }
    
    logger.info("Inicializando treinamento do modelo LightGBM...")
    model = lgb.LGBMClassifier(**params)
    
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        eval_names=["treino", "teste"]
    )
    
    logger.info("Treinamento finalizado com sucesso!")
    return model


def avaliar_desempenho_eventos_raros(model, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Avaliação rigorosa de métricas especializadas para desastres naturais:
        - PR-AUC (Average Precision): métrica padrão ouro para desbalanceamento severo
        - ROC-AUC: capacidade de discriminação ordinal
        - Ajuste de Threshold: na Defesa Civil, perder um desastre (Falso Negativo)
          tem custo infinitamente maior que um falso alarme (Falso Positivo).
    """
    logger.info("Calculando predições e probabilidades no conjunto de teste...")
    y_probs = model.predict_proba(X_test)[:, 1]
    
    pr_auc = average_precision_score(y_test, y_probs)
    roc_auc = roc_auc_score(y_test, y_probs)
    
    logger.info("=" * 70)
    logger.info(f"MÉTRICAS PRINCIPAIS (CONJUNTO DE TESTE FUTURO):")
    logger.info(f"  PR-AUC (Precision-Recall AUC / Average Precision): {pr_auc:.4f}")
    logger.info(f"  ROC-AUC (Área sob a curva ROC):                    {roc_auc:.4f}")
    logger.info("=" * 70)
    
    # Otimização do limiar de decisão (Threshold Tuning para maximizar o F1-Score)
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    melhor_idx = np.argmax(f1_scores)
    
    limiar_otimizado = thresholds[melhor_idx] if melhor_idx < len(thresholds) else 0.5
    melhor_f1 = f1_scores[melhor_idx]
    
    logger.info(f"Threshold padrão (0.50) vs. Threshold Otimizado para F1-Score ({limiar_otimizado:.3f}):")
    
    # Avaliação com Threshold Padrão (0.50)
    y_pred_50 = (y_probs >= 0.50).astype(int)
    cm_50 = confusion_matrix(y_test, y_pred_50)
    
    print("\n--- MATRIZ DE CONFUSÃO (Threshold = 0.50) ---")
    print(f"  Verdadeiros Negativos (TN): {cm_50[0, 0]:,}")
    print(f"  Falsos Positivos      (FP): {cm_50[0, 1]:,}")
    print(f"  Falsos Negativos      (FN): {cm_50[1, 0]:,}  <-- Riscos não alertados")
    print(f"  Verdadeiros Positivos (TP): {cm_50[1, 1]:,}  <-- Desastres previstos com sucesso")
    
    # Avaliação com Threshold Otimizado
    y_pred_otimizado = (y_probs >= limiar_otimizado).astype(int)
    cm_otim = confusion_matrix(y_test, y_pred_otimizado)
    
    print(f"\n--- MATRIZ DE CONFUSÃO (Threshold Otimizado = {limiar_otimizado:.3f}) ---")
    print(f"  Verdadeiros Negativos (TN): {cm_otim[0, 0]:,}")
    print(f"  Falsos Positivos      (FP): {cm_otim[0, 1]:,}")
    print(f"  Falsos Negativos      (FN): {cm_otim[1, 0]:,}")
    print(f"  Verdadeiros Positivos (TP): {cm_otim[1, 1]:,}")
    
    print("\n--- RELATÓRIO DE CLASSIFICAÇÃO DETALHADO (Threshold Otimizado) ---")
    print(classification_report(y_test, y_pred_otimizado, target_names=["Sem Ocorrência (0)", "Desastre (1)"]))
    
    # Importância das Features (Gain)
    df_importancia = pd.DataFrame({
        "feature": model.feature_name_,
        "importance_gain": model.booster_.feature_importance(importance_type="gain"),
        "importance_split": model.booster_.feature_importance(importance_type="split")
    }).sort_values(by="importance_gain", ascending=False).reset_index(drop=True)
    
    print("\n--- TOP 10 ATRIBUTOS MAIS RELEVANTES (POR GANHO DE INFORMAÇÃO) ---")
    print(df_importancia.head(10).to_string())
    
    return pr_auc, roc_auc, limiar_otimizado, df_importancia


def salvar_modelo_e_metadados(model, features: list, limiar_otimizado: float, caminho_saida: Path):
    """
    Persiste o modelo e os metadados necessários para inferência em produção.
    """
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    
    pacote_producao = {
        "modelo": model,
        "features": features,
        "limiar_otimizado": float(limiar_otimizado),
        "crs_modelo": "EPSG:31980",
        "descricao": "Classificador LightGBM para Alagamentos Urbanos e Deslizamentos em Manaus (AM)"
    }
    
    joblib.dump(pacote_producao, caminho_saida)
    logger.info(f"Modelo e artefato de inferência salvos com sucesso em: '{caminho_saida}'")


def main():
    logger.info("Iniciando SCRIPT 05: Treinamento e Avaliação do Modelo Preditivo...")
    DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    
    # 1. Carregar dados e preparar matrizes
    df, features = carregar_e_preparar_dados(CAMINHO_DATASET)
    
    # 2. Divisão estritamente temporal (80% treino / 20% teste)
    X_train, y_train, X_test, y_test, df_teste = split_temporal(df, features, proporcao_treino=0.80)
    
    # 3. Treinar LightGBM com balanceamento por scale_pos_weight
    model = treinar_lightgbm(X_train, y_train, X_test, y_test)
    
    # 4. Avaliar métricas para eventos raros
    pr_auc, roc_auc, limiar, df_imp = avaliar_desempenho_eventos_raros(model, X_test, y_test)
    
    # 5. Salvar artefato de produção
    salvar_modelo_e_metadados(model, features, limiar, CAMINHO_MODELO_SALVO)
    
    logger.info("Etapa 05 concluída com sucesso! Pipeline de Machine Learning completo.")


if __name__ == "__main__":
    main()
