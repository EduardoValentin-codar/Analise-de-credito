# 💳 Simulador de Aprovação de Crédito com Random Forest

https://blank-app-8v9vm72liol.streamlit.app/

Projeto da disciplina de **Mineração de Dados** (UNEX). Site em Streamlit que simula a análise de crédito bancário usando o algoritmo **Random Forest**.


> ⚠️ Resultado meramente simulado, não representa uma decisão bancária real. Todos os dados são fictícios.

## Como funciona
O usuário informa idade, renda mensal, crédito solicitado, score de crédito, tempo de emprego e dívidas mensais. O site retorna:
- a decisão (**aprovado** ou **reprovado**);
- a **probabilidade estimada** de aprovação (corte em 50%);
- os **3 fatores mais importantes**, comparando o valor informado com a mediana dos aprovados.

## Dados
- 500 registros **simulados** (`gerar_dados.py`), com 6 variáveis e o rótulo `aprovado`.
- O rótulo é gerado por uma regra de negócio (score, renda, dívidas, crédito e tempo de emprego) somada a ruído aleatório.
- Cerca de 3% dos valores de renda, tempo de emprego e dívidas são ausentes de propósito, para exercitar o tratamento de dados.

## Modelo
- **Tratamento:** valores ausentes preenchidos com a mediana de cada coluna.
- **Divisão:** 80% treino e 20% teste, estratificada.
- **Algoritmo:** `RandomForestClassifier` (200 árvores, profundidade máxima 8, classes balanceadas).
- **Resultados no teste:** acurácia 82%, precisão 74%, recall 74%, F1-score 74%.
- **Importância das variáveis:** score de crédito (38%), renda mensal (23%), crédito solicitado (12%), dívidas mensais (11%), tempo de emprego (9%), idade (7%).

## Estrutura
| Arquivo | Função |
|---|---|
| `gerar_dados.py` | Gera a base simulada em `data/credito.csv` |
| `treinar_modelo.py` | Trata os dados, treina, avalia e salva o modelo |
| `modelo_credito.joblib` | Modelo treinado |
| `metricas.json` | Métricas e importâncias |
| `app.py` | Site Streamlit |
| `.streamlit/config.toml` | Tema do site |
| `requirements.txt` | Dependências |

## Como executar
```bash
pip install -r requirements.txt
python gerar_dados.py
python treinar_modelo.py
python -m streamlit run app.py
```
O site abre em `http://localhost:8501`. (Opcional: criar um ambiente virtual antes com `python -m venv venv`.)

## Limitações
- Como os dados são simulados e o rótulo vem de uma regra