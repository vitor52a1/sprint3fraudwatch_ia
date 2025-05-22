from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
import joblib
import ast  # Para converter strings em arrays
import json

# Carregar o dataset
dataframe = pd.read_csv('dados_dentista_reduzido_10000.csv')

# Função para converter strings de embeddings em arrays
def converter_embedding(embedding_str):
    embedding_str = embedding_str.strip('[]')
    embedding_str = ','.join(embedding_str.split())
    return np.array(ast.literal_eval(f"[{embedding_str}]"))

# Aplicar a conversão para todas as colunas de embeddings
colunas_embedding = ['sintomas_embedding', 'diagnostico_embedding', 'procedimento_embedding', 'lista_de_insumos_embedding']
for coluna in colunas_embedding:
    dataframe[coluna] = dataframe[coluna].apply(converter_embedding)

# Combinar os embeddings em um único array de entrada X
X = np.hstack([
    np.vstack(dataframe['sintomas_embedding']),
    np.vstack(dataframe['diagnostico_embedding']),
    np.vstack(dataframe['procedimento_embedding']),
    np.vstack(dataframe['lista_de_insumos_embedding'])
])

# Variável alvo
y = dataframe['chance_de_fraude'].values  # Supondo que a coluna 'chance_de_fraude' existe

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo
modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_train, y_train)

# Fazer previsões
y_pred = modelo.predict(X_test)

# Calcular métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

# Mostrar as métricas no terminal
print(f"Acurácia: {accuracy:.4f}")
print(f"Precisão: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Salvar o modelo
joblib.dump(modelo, 'modelo_fraude.pkl')

# Salvar as métricas em um arquivo JSON
metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1
}

with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

print("Modelo e métricas salvos com sucesso.")
