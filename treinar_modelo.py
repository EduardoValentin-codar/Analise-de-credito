"""Trata os dados, treina o Random Forest, avalia e salva o modelo."""
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

FEATURES = ["idade", "renda_mensal", "valor_credito",
            "score_credito", "tempo_emprego", "dividas_mensais"]

df = pd.read_csv("data/credito.csv")

# Tratamento: valores ausentes preenchidos com a mediana de cada coluna
medianas = df[FEATURES].median()
df[FEATURES] = df[FEATURES].fillna(medianas)

X, y = df[FEATURES], df["aprovado"]
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

modelo = RandomForestClassifier(n_estimators=200, max_depth=8,
                                random_state=42, class_weight="balanced")
modelo.fit(X_treino, y_treino)

pred = modelo.predict(X_teste)
metricas = {
    "acuracia": accuracy_score(y_teste, pred),
    "precisao": precision_score(y_teste, pred),
    "recall": recall_score(y_teste, pred),
    "f1": f1_score(y_teste, pred),
}
for nome, valor in metricas.items():
    print(f"{nome:>9}: {valor:.3f}")

importancias = dict(zip(FEATURES, modelo.feature_importances_.round(4).tolist()))
print("Importâncias:", importancias)

joblib.dump({"modelo": modelo, "medianas": medianas.to_dict(), "features": FEATURES},
            "modelo_credito.joblib")
with open("metricas.json", "w", encoding="utf-8") as f:
    json.dump({"metricas": metricas, "importancias": importancias}, f, indent=2)
print("Modelo salvo em modelo_credito.joblib")
