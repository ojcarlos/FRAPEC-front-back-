"""
================================================================================
PROJETO: PREVISÃO DE ALAGAMENTOS E DESLIZAMENTOS - MANAUS (AM)
ARQUIVO: consultar_bairro.py (SCRIPT CLIENTE CLI PARA CONSULTA DE RISCO)
================================================================================
Objetivo:
    Script de terminal para consultar o risco iminente de alagamento ou deslizamento
    em qualquer bairro de Manaus, integrando os dados em tempo real da Meteomatics
    e exibindo um relatório formatado para a Defesa Civil e tomadores de decisão.

Uso:
    python consultar_bairro.py --bairro "Educandos"
    python consultar_bairro.py --bairro "Jorge Teixeira"
    python consultar_bairro.py --listar
    python consultar_bairro.py --todos
    python consultar_bairro.py --bairro "Centro" --simular-chuva-extrema
================================================================================
"""

import sys
import argparse
import json
import requests

from servico_previsao import get_servico_previsao, CATALOGO_BAIRROS


def imprimir_cabecalho():
    print("=" * 80)
    print("  SISTEMA DE ALERTA PRECOCE: ALAGAMENTOS E DESLIZAMENTOS - MANAUS (AM)")
    print("  Monitoramento Georreferenciado com Meteomatics API, Sentinel-1 e LightGBM")
    print("=" * 80)


def formatar_relatorio_bairro(resultado: dict):
    """Exibe um relatório estruturado e legível no terminal."""
    bairro = resultado["bairro"]
    zona = resultado["zona_manaus"]
    coord = resultado["coordenadas"]
    av = resultado["avaliacao_risco"]
    met = resultado["dados_meteorologicos"]
    cel = resultado["celula_grade"]
    sar = resultado["radar_sentinel1"]

    # Cores/Símbolos para status de alerta
    nivel = av["nivel_alerta"]
    if nivel == "BAIXO":
        icone = "🟢 [BAIXO]"
    elif nivel == "ATENÇÃO":
        icone = "🟡 [ATENÇÃO]"
    elif nivel == "ALERTA":
        icone = "🟠 [ALERTA]"
    else:
        icone = "🔴 [EMERGÊNCIA MÁXIMA]"

    print("\n" + "-" * 80)
    print(f"📍 BAIRRO: {bairro.upper()} (Zona {zona} de Manaus)")
    print(f"   Coordenadas: Lat {coord['latitude']:.4f}, Lon {coord['longitude']:.4f}")
    print(f"   Célula Grade UTM 20S: {cel['cell_id']} (Cota Min: {cel['elev_min_m']:.1f}m | Declividade Max: {cel['declividade_max_graus']:.1f}°)")
    print("-" * 80)

    print(f"\n🚨 STATUS DE RISCO: {icone}")
    print(f"   Probabilidade de Desastre: {av['probabilidade_desastre_pct']:.2f}% (Limiar Crítico: {av['limiar_decisao_pct']:.2f}%)")
    print(f"   Tipologia Prevista:        {av['ameaca_predita']}")

    print(f"\n🌧️ TELEMETRIA METEOROLÓGICA (Meteomatics / Tempo Real):")
    print(f"   Fonte de Dados:   {met['fonte']}")
    print(f"   Precipitação 1h:  {met['chuva_acum_1h']:.1f} mm")
    print(f"   Acumulado 6h:     {met['chuva_acum_6h']:.1f} mm  (Gatilho de Enxurrada/Alagamento)")
    print(f"   Acumulado 24h:    {met['chuva_acum_24h']:.1f} mm")
    print(f"   Acumulado 72h:    {met['chuva_acum_72h']:.1f} mm  (Gatilho de Saturação de Encosta)")

    print(f"\n🛰️ PARÂMETROS DE RADAR (Sentinel-1 SAR C-Band):")
    print(f"   Retroespalhamento VV: {sar['sar_vv_mean_db']:.2f} dB (Média)")
    print(f"   Retroespalhamento VH: {sar['sar_vh_mean_db']:.2f} dB")
    print(f"   Razão Polarimétrica:  {sar['sar_diferenca_vv_vh']:.2f} dB (Sensível a umidade do solo)")

    print(f"\n🔍 FATORES DETERMINANTES IDENTIFICADOS:")
    for f in av["fatores_determinantes"]:
        print(f"   • {f}")

    print(f"\n📋 RECOMENDAÇÕES DA DEFESA CIVIL:")
    for r in av["recomendacoes_defesa_civil"]:
        print(f"   ⚠️  {r}")

    print(f"\n📖 CONTEXTO HISTÓRICO DO BAIRRO:")
    print(f"   {resultado['historico_bairro']}")
    print("-" * 80)


def listar_bairros_disponiveis():
    imprimir_cabecalho()
    print("\nBairros Urbanos Cadastrados em Manaus (AM):")
    print(f"{'Bairro':<28} | {'Zona':<12} | {'Latitude':<10} | {'Longitude':<10}")
    print("-" * 70)
    for chave, item in sorted(CATALOGO_BAIRROS.items(), key=lambda x: x[1]["nome"]):
        print(f"{item['nome']:<28} | {item['zona']:<12} | {item['lat']:<10.4f} | {item['lon']:<10.4f}")
    print("-" * 70)
    print(f"Total: {len(CATALOGO_BAIRROS)} bairros mapeados na grade espacial.")


def executar_consulta_via_api(url_api: str, nome_bairro: str) -> dict:
    """Tenta consultar via endpoint HTTP da API."""
    try:
        url = f"{url_api.rstrip('/')}/previsao/{nome_bairro}"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 404:
            erro = resp.json().get("detail", "Bairro não encontrado.")
            print(f"❌ Erro 404: {erro}")
            return None
        else:
            print(f"❌ Erro HTTP {resp.status_code} na API: {resp.text}")
            return None
    except requests.exceptions.ConnectionError:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Consulta em Tempo Real de Risco de Alagamentos e Deslizamentos em Bairros de Manaus (AM)"
    )
    parser.add_argument(
        "--bairro", "-b",
        type=str,
        help="Nome do bairro para consulta (ex: 'Educandos', 'Centro', 'Jorge Teixeira', 'Flores')"
    )
    parser.add_argument(
        "--listar", "-l",
        action="store_true",
        help="Lista todos os bairros urbanos mapeados em Manaus"
    )
    parser.add_argument(
        "--todos", "-t",
        action="store_true",
        help="Executa diagnóstico de risco em lote para todos os bairros de Manaus"
    )
    parser.add_argument(
        "--simular-chuva-extrema",
        action="store_true",
        help="Simula cenário de tempestade tropical severa (65mm em 1h, 110mm em 6h, 140mm em 72h)"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000",
        help="URL base da API FastAPI (padrão: http://localhost:8000)"
    )

    args = parser.parse_args()

    if args.listar:
        listar_bairros_disponiveis()
        return

    imprimir_cabecalho()

    # Caso queira consultar todos os bairros em lote
    if args.todos:
        print("\nExecutando diagnóstico de risco em lote para toda a cidade de Manaus...\n")
        servico = get_servico_previsao()
        
        chuva_custom = None
        if args.simular_chuva_extrema:
            print("⚡ MODO SIMULAÇÃO DE TEMPESTADE SEVERA ATIVADO!")
            chuva_custom = {
                "chuva_acum_1h": 65.0,
                "chuva_acum_6h": 110.0,
                "chuva_acum_24h": 125.0,
                "chuva_acum_72h": 140.0
            }

        resultados = []
        for chave, item in sorted(CATALOGO_BAIRROS.items(), key=lambda x: x[1]["nome"]):
            res = servico.avaliar_risco(item["nome"], chuva_custom=chuva_custom)
            resultados.append(res)

        print(f"{'Bairro':<24} | {'Zona':<10} | {'Nível de Alerta':<16} | {'Prob (%)':<8} | {'Ameaça Principal'}")
        print("-" * 95)
        for r in resultados:
            av = r["avaliacao_risco"]
            print(
                f"{r['bairro']:<24} | {r['zona_manaus']:<10} | "
                f"{av['nivel_alerta']:<16} | {av['probabilidade_desastre_pct']:<8.1f} | "
                f"{av['ameaca_predita']}"
            )
        print("-" * 95)
        return

    # Consulta de bairro único
    bairro_alvo = args.bairro
    if not bairro_alvo:
        # Se nenhum argumento for passado, consulta um bairro clássico de alta vulnerabilidade
        print("Nenhum bairro informado via argumento. Consultando bairro de exemplo: 'Educandos'...")
        bairro_alvo = "Educandos"

    chuva_custom = None
    if args.simular_chuva_extrema:
        print("⚡ MODO SIMULAÇÃO DE TEMPESTADE SEVERA ATIVADO (65mm em 1h, 110mm em 6h, 140mm em 72h)!")
        chuva_custom = {
            "chuva_acum_1h": 65.0,
            "chuva_acum_6h": 110.0,
            "chuva_acum_24h": 125.0,
            "chuva_acum_72h": 140.0
        }

    # 1. Tenta consulta pela API se estiver ativa (e sem customização)
    resultado = None
    if not chuva_custom:
        resultado = executar_consulta_via_api(args.api_url, bairro_alvo)

    # 2. Se a API não estiver rodando ou se houver parâmetros simulados, executa diretamente via serviço local
    if resultado is None:
        try:
            servico = get_servico_previsao()
            resultado = servico.avaliar_risco(bairro_alvo, chuva_custom=chuva_custom)
        except ValueError as e:
            print(f"\n❌ Erro: {e}")
            sys.exit(1)

    formatar_relatorio_bairro(resultado)


if __name__ == "__main__":
    main()
