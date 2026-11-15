"""
================================================================================
PROJETO: PREVISÃO DE ALAGAMENTOS E DESLIZAMENTOS - MANAUS (AM)
SCRIPT 01: GERAÇÃO DA GRADE ESPACIAL REGULAR E EXTRAÇÃO DE ATRIBUTOS DE RELEVO
================================================================================
Objetivo:
    1. Projetar a Bounding Box urbana de Manaus de WGS84 (EPSG:4326) para
       SIRGAS 2000 / UTM zona 20S (EPSG:31980).
    2. Gerar uma grade regular de células de 500m x 500m com identificador 'cell_id'.
    3. Ler ou sintetizar o Modelo Digital de Elevação (DEM) em UTM 20S.
    4. Calcular a declividade (slope) em graus através do gradiente numérico (np.gradient).
    5. Extrair estatísticas zonais (rasterstats): elev_media_m, elev_min_m,
       declividade_media e declividade_max por célula.
    6. Salvar a grade estruturada em 'dados_processados/grade_relevo.parquet'.
================================================================================
"""

import os
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, Point
import rasterio
from rasterio.transform import from_origin
from rasterstats import zonal_stats
from pyproj import Transformer

# ------------------------------------------------------------------------------
# CONFIGURAÇÃO DE LOGGING
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("01_grade_relevo")

# ------------------------------------------------------------------------------
# CONSTANTES E PARÂMETROS GEOGRÁFICOS
# ------------------------------------------------------------------------------
# Bounding Box oficial solicitada para a mancha urbana de Manaus (WGS84)
BBOX_WGS84 = {
    "min_lon": -60.15,
    "min_lat": -3.16,
    "max_lon": -59.80,
    "max_lat": -2.90
}

# Sistemas de Referência de Coordenadas (CRS)
CRS_GEOGRAFICO = "EPSG:4326"   # WGS84 Geográfico
CRS_METRICO = "EPSG:31980"      # SIRGAS 2000 / UTM zona 20S (Manaus - AM)

TAMANHO_CELULA_METROS = 500.0   # Resolução da grade: 500m x 500m

# Caminhos de arquivos
DIR_BRUTOS = Path("dados_brutos")
DIR_PROCESSADOS = Path("dados_processados")
CAMINHO_DEM_TIF = DIR_BRUTOS / "dem_manaus_utm.tif"
CAMINHO_GRADE_PARQUET = DIR_PROCESSADOS / "grade_relevo.parquet"


def criar_diretorios():
    """Garante a existência das pastas necessárias para o pipeline."""
    DIR_BRUTOS.mkdir(parents=True, exist_ok=True)
    DIR_PROCESSADOS.mkdir(parents=True, exist_ok=True)
    logger.info("Diretórios de dados verificados/criados com sucesso.")


def reprojetar_bbox_para_utm(bbox_wgs84: dict, crs_origem: str = CRS_GEOGRAFICO, crs_destino: str = CRS_METRICO) -> tuple:
    """
    Converte as coordenadas da Bounding Box de WGS84 para UTM zona 20S (SIRGAS 2000).

    Retorna:
        tuple: (min_x, min_y, max_x, max_y) em metros no EPSG:31980.
    """
    transformer = Transformer.from_crs(crs_origem, crs_destino, always_xy=True)
    
    # Converte os quatro cantos para obter a extensão delimitadora correta em projeção métrica
    xs = [bbox_wgs84["min_lon"], bbox_wgs84["max_lon"], bbox_wgs84["min_lon"], bbox_wgs84["max_lon"]]
    ys = [bbox_wgs84["min_lat"], bbox_wgs84["min_lat"], bbox_wgs84["max_lat"], bbox_wgs84["max_lat"]]
    
    utm_xs, utm_ys = transformer.transform(xs, ys)
    
    min_x, max_x = min(utm_xs), max(utm_xs)
    min_y, max_y = min(utm_ys), max(utm_ys)
    
    logger.info(
        f"Extensão convertida para {crs_destino} (UTM):\n"
        f"  X: [{min_x:,.2f} m, {max_x:,.2f} m] (Largura: {(max_x - min_x)/1000:.2f} km)\n"
        f"  Y: [{min_y:,.2f} m, {max_y:,.2f} m] (Altura:  {(max_y - min_y)/1000:.2f} km)"
    )
    return min_x, min_y, max_x, max_y


def gerar_grade_regular(bounds: tuple, cell_size: float = TAMANHO_CELULA_METROS, crs: str = CRS_METRICO) -> gpd.GeoDataFrame:
    """
    Gera uma malha vetorial regular (grid) com células de cell_size x cell_size metros.
    
    Atribui a cada célula:
        - cell_id: Identificador alfanumérico único ('cell_00000', etc.)
        - centroid_x: Coordenada métrica X do centroide (EPSG:31980)
        - centroid_y: Coordenada métrica Y do centroide (EPSG:31980)
        - geometry: Polígono retangular da célula
    """
    min_x, min_y, max_x, max_y = bounds
    
    # Cria os passos nas direções X e Y
    x_coords = np.arange(min_x, max_x, cell_size)
    y_coords = np.arange(min_y, max_y, cell_size)
    
    poligonos = []
    cell_ids = []
    centroides_x = []
    centroides_y = []
    
    idx = 0
    for x in x_coords:
        for y in y_coords:
            p = box(x, y, x + cell_size, y + cell_size)
            poligonos.append(p)
            cell_ids.append(f"cell_{idx:05d}")
            centroides_x.append(x + cell_size / 2.0)
            centroides_y.append(y + cell_size / 2.0)
            idx += 1
            
    gdf = gpd.GeoDataFrame(
        {
            "cell_id": cell_ids,
            "centroid_x": centroides_x,
            "centroid_y": centroides_y
        },
        geometry=poligonos,
        crs=crs
    )
    
    logger.info(f"Grade regular criada com sucesso: {len(gdf)} células de {cell_size:.0f}m x {cell_size:.0f}m.")
    return gdf


def gerar_dem_sintetico_se_necessario(caminho_tif: Path, bounds: tuple, crs: str = CRS_METRICO, resolucao: float = 30.0):
    """
    Gera um raster DEM sintético em GeoTIFF caso o arquivo real não exista localmente.
    
    Características do relevo de Manaus (Formação Alter do Chão):
        - Cotas baixas próximas ao Rio Negro e igarapés: ~15m a 30m.
        - Platôs estruturais e tabuleiros: ~60m a 110m.
        - Encostas e vertentes com alta dissecação, propensas a deslizamentos.
    """
    if caminho_tif.exists():
        logger.info(f"Arquivo DEM existente localizado em: {caminho_tif}")
        return

    logger.warning(
        f"DEM não encontrado em '{caminho_tif}'. "
        f"Gerando DEM sintético realista de Manaus (30m de resolução) para testes..."
    )
    
    min_x, min_y, max_x, max_y = bounds
    
    # Adiciona uma margem de segurança ao redor da grade
    margem = 1000.0  # 1 km
    min_x -= margem
    min_y -= margem
    max_x += margem
    max_y += margem
    
    cols = int(np.ceil((max_x - min_x) / resolucao))
    rows = int(np.ceil((max_y - min_y) / resolucao))
    
    # Cria uma malha de coordenadas 2D
    x = np.linspace(min_x, max_x, cols)
    y = np.linspace(max_y, min_y, rows)  # Y decresce de cima para baixo
    xx, yy = np.meshgrid(x, y)
    
    # Modelagem sintética do relevo: platôs ondulados e vales de drenagem (igarapés)
    # Relevo base: gradiente suave em direção ao sul (Rio Negro)
    base = 25.0 + 35.0 * ((yy - min_y) / (max_y - min_y))
    
    # Vales de igarapés principais (linhas de fratura / erosão profunda)
    igarapes = -18.0 * np.sin(xx / 3000.0) * np.cos(yy / 2500.0)
    
    # Topografia de tabuleiros e platôs dissecados com vertentes íngremes
    platoes = 40.0 / (1.0 + np.exp(-np.sin(xx / 1800.0 + yy / 2200.0) * 3.5))
    
    # Ruído de micro-relevo
    np.random.seed(42)
    ruido = np.random.normal(0, 1.2, size=(rows, cols))
    
    elevacao = base + igarapes + platoes + ruido
    elevacao = np.clip(elevacao, 12.0, 125.0).astype(np.float32)
    
    transform = from_origin(min_x, max_y, resolucao, resolucao)
    
    with rasterio.open(
        caminho_tif,
        "w",
        driver="GTiff",
        height=rows,
        width=cols,
        count=1,
        dtype=np.float32,
        crs=crs,
        transform=transform,
        nodata=-9999.0
    ) as dst:
        dst.write(elevacao, 1)
        
    logger.info(
        f"DEM sintético salvo com sucesso em '{caminho_tif}':\n"
        f"  Dimensões: {rows} linhas x {cols} colunas\n"
        f"  Altitude Mínima: {elevacao.min():.2f}m | Máxima: {elevacao.max():.2f}m | Média: {elevacao.mean():.2f}m"
    )


def calcular_declividade_graus(dem_array: np.ndarray, res_x: float, res_y: float) -> np.ndarray:
    """
    Calcula a declividade (slope) em graus a partir de uma matriz de elevação
    usando o gradiente numérico centrado (np.gradient).

    Fórmula:
        slope_rad = arctan( sqrt((dz/dx)^2 + (dz/dy)^2) )
        slope_deg = rad2deg(slope_rad)
    """
    # np.gradient retorna derivadas parciais: [eixo 0 (linhas / Y), eixo 1 (colunas / X)]
    dz_dy, dz_dx = np.gradient(dem_array, res_y, res_x)
    
    grad_magnitude = np.sqrt(dz_dx**2 + dz_dy**2)
    slope_rad = np.arctan(grad_magnitude)
    slope_deg = np.degrees(slope_rad)
    
    return slope_deg.astype(np.float32)


def extrair_atributos_relevo(gdf_grade: gpd.GeoDataFrame, caminho_dem: Path) -> gpd.GeoDataFrame:
    """
    Lê o DEM, deriva a declividade e executa estatísticas zonais para cada célula.
    
    Métricas extraídas:
        - elev_media_m: Elevação média da célula (m)
        - elev_min_m: Cota mínima da célula (m) -> indicativo crítico de baixios sujeitos a inundação
        - declividade_media: Declividade média em graus
        - declividade_max: Declividade máxima em graus -> indicativo crítico de risco de deslizamento
    """
    logger.info(f"Lendo dados de relevo do arquivo: {caminho_dem}")
    
    with rasterio.open(caminho_dem) as src:
        dem_data = src.read(1)
        transform = src.transform
        nodata = src.nodata
        res_x = abs(src.res[0])
        res_y = abs(src.res[1])
        dem_crs = src.crs

    # Tratamento de nodata
    if nodata is not None:
        dem_data = np.where(dem_data == nodata, np.nan, dem_data)
        
    logger.info("Calculando matriz de declividade (slope em graus) via np.gradient...")
    slope_data = calcular_declividade_graus(dem_data, res_x, res_y)
    
    logger.info("Extraindo estatísticas zonais de elevação por célula da grade...")
    stats_elev = zonal_stats(
        gdf_grade,
        dem_data,
        affine=transform,
        stats=["mean", "min"],
        nodata=nodata if nodata is not None else np.nan
    )
    
    logger.info("Extraindo estatísticas zonais de declividade por célula da grade...")
    stats_slope = zonal_stats(
        gdf_grade,
        slope_data,
        affine=transform,
        stats=["mean", "max"],
        nodata=nodata if nodata is not None else np.nan
    )
    
    # Anexa as colunas ao GeoDataFrame
    gdf_resultado = gdf_grade.copy()
    gdf_resultado["elev_media_m"] = [round(s["mean"], 2) if s["mean"] is not None else np.nan for s in stats_elev]
    gdf_resultado["elev_min_m"] = [round(s["min"], 2) if s["min"] is not None else np.nan for s in stats_elev]
    gdf_resultado["declividade_media"] = [round(s["mean"], 2) if s["mean"] is not None else np.nan for s in stats_slope]
    gdf_resultado["declividade_max"] = [round(s["max"], 2) if s["max"] is not None else np.nan for s in stats_slope]
    
    # Preenche eventuais células de borda com a mediana
    for col in ["elev_media_m", "elev_min_m", "declividade_media", "declividade_max"]:
        if gdf_resultado[col].isna().any():
            mediana = gdf_resultado[col].median()
            gdf_resultado[col] = gdf_resultado[col].fillna(mediana)
            logger.warning(f"Valores nulos preenchidos na coluna {col} com a mediana: {mediana:.2f}")

    return gdf_resultado


def main():
    logger.info("Iniciando SCRIPT 01: Processamento da Grade Espacial e Topografia...")
    criar_diretorios()
    
    # 1. Obter extensão em coordenadas métricas SIRGAS 2000 / UTM zona 20S
    bounds_utm = reprojetar_bbox_para_utm(BBOX_WGS84)
    
    # 2. Gerar grade vetorial regular de 500m x 500m
    gdf_grade = gerar_grade_regular(bounds_utm, cell_size=TAMANHO_CELULA_METROS, crs=CRS_METRICO)
    
    # 3. Garantir disponibilidade do DEM
    gerar_dem_sintetico_se_necessario(CAMINHO_DEM_TIF, bounds_utm, crs=CRS_METRICO)
    
    # 4. Extrair estatísticas zonais de relevo e declividade
    gdf_relevo = extrair_atributos_relevo(gdf_grade, CAMINHO_DEM_TIF)
    
    # 5. Salvar resultado em formato Parquet
    logger.info(f"Salvando GeoDataFrame em Parquet: '{CAMINHO_GRADE_PARQUET}'...")
    gdf_relevo.to_parquet(CAMINHO_GRADE_PARQUET, index=False)
    
    logger.info("Resumo descritivo dos atributos de relevo calculados:")
    print(gdf_relevo[["elev_media_m", "elev_min_m", "declividade_media", "declividade_max"]].describe().to_string())
    
    logger.info(f"Etapa 01 concluída com sucesso! Total de células: {len(gdf_relevo)}")


if __name__ == "__main__":
    main()
