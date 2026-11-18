"""Gera a base de dados SIMULADA de pessoas com perfis financeiros diferentes."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 500

idade = rng.integers(18, 71, N)
renda = np.clip(rng.lognormal(mean=8.3, sigma=0.55, size=N), 1500, 30000).round(2)
score = np.clip(rng.normal(600, 130, N), 0, 1000).round().astype(int)
tempo_emprego = np.clip(rng.normal((idade - 18) * 0.45, 3, N), 0, 40).round(1)
tempo_emprego = np.minimum(tempo_emprego, idade - 16)
dividas = (renda * rng.uniform(0, 0.7, N)).round(2)
credito = (renda * rng.uniform(1, 15, N)).round(2)

# Regra "oculta" que gera o rótulo (com ruído, para o modelo não ficar perfeito)
pontos = (
    (score - 500) / 250 * 2.2
    + (renda - 4000) / 4000 * 1.2
    - dividas / renda * 3.0
    - credito / (renda * 12) * 1.5
    + np.minimum(tempo_emprego, 10) / 10 * 0.8
    + rng.normal(0, 0.6, N)
)
aprovado = (pontos > 0.3).astype(int)

df = pd.DataFrame({
    "idade": idade,
    "renda_mensal": renda,
    "valor_credito": credito,
    "score_credito": score,
    "tempo_emprego": tempo_emprego,
    "dividas_mensais": dividas,
    "aprovado": aprovado,
})

# Introduz valores ausentes (~3%) para exercitar o tratamento de dados
for col in ["renda_mensal", "tempo_emprego", "dividas_mensais"]:
    df.loc[rng.random(N) < 0.03, col] = np.nan

df.to_csv("data/credito.csv", index=False)
print(f"{len(df)} registros salvos em data/credito.csv")
print(f"Aprovados: {df['aprovado'].mean():.0%} | Ausentes:\n{df.isna().sum()[df.isna().sum() > 0]}")
