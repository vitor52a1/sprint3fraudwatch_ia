from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Definir o dispositivo (CPU ou GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Caminho para salvar o modelo
model_path = 'modelo_fraude.pth'

# Definir a arquitetura do modelo
class FraudPredictor(nn.Module):
    def __init__(self, input_size):
        super(FraudPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Carregar o dataset reduzido
dataframe = pd.read_csv('dados_dentista_reduzido_10000.csv')
print("Dataset carregado com sucesso!")
print(dataframe.head())

# Verificar os valores de chance_de_fraude
print("Valores mínimos e máximos de chance_de_fraude:")
print(f"Mínimo: {dataframe['chance_de_fraude'].min()}")
print(f"Máximo: {dataframe['chance_de_fraude'].max()}")

# Criar um mapeamento de texto para embeddings
text_to_embedding = {}
for col in ['sintomas', 'diagnostico', 'procedimento', 'lista_de_insumos']:
    texts = dataframe[col].unique()
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(texts).toarray()
    for text, embedding in zip(texts, embeddings):
        text_to_embedding[text] = embedding

# Preparar os dados para treinamento
X = np.array([np.concatenate([
    text_to_embedding[row['sintomas']],
    text_to_embedding[row['diagnostico']],
    text_to_embedding[row['procedimento']],
    text_to_embedding[row['lista_de_insumos']]
]) for _, row in dataframe.iterrows()])

# Verificar o número de features
input_size = X.shape[1]
print(f"Número de features: {input_size}")

# Normalizar chance_de_fraude (assumindo percentual de 0 a 100)
y = dataframe['chance_de_fraude'].values / 100

# Converter para tensores
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)

# Instanciar o modelo
predictor = FraudPredictor(input_size).to(device)

# Treinamento
if os.path.exists(model_path):
    print("Carregando modelo pré-treinado...")
    predictor.load_state_dict(torch.load(model_path))
    predictor.eval()
else:
    print("Treinando o modelo...")
    optimizer = torch.optim.Adam(predictor.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = predictor(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    torch.save(predictor.state_dict(), model_path)
    print(f"Modelo treinado e salvo em {model_path}")
    predictor.eval()

# Rota da API para previsão
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Extrair os textos do JSON
    sintomas = data['sintomas']
    diagnostico = data['diagnostico']
    procedimento = data['procedimento']
    lista_de_insumos = data['lista_de_insumos']

    # Converter textos em embeddings
    sintomas_embedding = text_to_embedding.get(sintomas, np.zeros(input_size // 4))
    diagnostico_embedding = text_to_embedding.get(diagnostico, np.zeros(input_size // 4))
    procedimento_embedding = text_to_embedding.get(procedimento, np.zeros(input_size // 4))
    lista_de_insumos_embedding = text_to_embedding.get(lista_de_insumos, np.zeros(input_size // 4))

    # Concatenar os embeddings
    input_data = np.concatenate([
        sintomas_embedding,
        diagnostico_embedding,
        procedimento_embedding,
        lista_de_insumos_embedding
    ])

    # Verificar o tamanho dos dados de entrada
    print(f"Tamanho dos dados de entrada: {input_data.shape}")

    # Converter para tensor
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)

    # Fazer a previsão
    with torch.no_grad():
        prediction = predictor(input_tensor).item()

    return jsonify({'chance_de_fraude': prediction * 100})  # Retornar em percentual

if __name__ == '__main__':
    app.run(debug=True)