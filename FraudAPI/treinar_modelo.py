from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import joblib
import ast  # Para converter strings em arrays

# Carregar o dataset
dataframe = pd.read_csv('dados_dentista_reduzido_10000.csv')

# Função para converter strings de embeddings em arrays
def converter_embedding(embedding_str):
    # Remover colchetes e espaços extras
    embedding_str = embedding_str.strip('[]')
    # Substituir múltiplos espaços por uma única vírgula
    embedding_str = ','.join(embedding_str.split())
    # Converter a string em uma lista de números
    return np.array(ast.literal_eval(f"[{embedding_str}]"))

# Aplicar a conversão para todas as colunas de embeddings
colunas_embedding = ['sintomas_embedding', 'diagnostico_embedding', 'procedimento_embedding', 'lista_de_insumos_embedding']
for coluna in colunas_embedding:
    dataframe[coluna] = dataframe[coluna].apply(converter_embedding)

# Combinar os embeddings
X = np.hstack([
    np.vstack(dataframe['sintomas_embedding']),
    np.vstack(dataframe['diagnostico_embedding']),
    np.vstack(dataframe['procedimento_embedding']),
    np.vstack(dataframe['lista_de_insumos_embedding'])
])
y = dataframe['chance_de_fraude'].values  # Supondo que a coluna 'chance_de_fraude' já existe no dataset

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo
modelo = LogisticRegression()
modelo.fit(X_train, y_train)

# Avaliar o modelo
y_pred = modelo.predict(X_test)
print(f"Acurácia do modelo: {accuracy_score(y_test, y_pred)}")

# Salvar o modelo (usando joblib)
joblib.dump(modelo, 'modelo_fraude.pkl')