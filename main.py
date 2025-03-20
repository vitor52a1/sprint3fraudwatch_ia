from fastapi import FastAPI
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from pydantic import BaseModel
import joblib
import numpy as np

# Defina o modelo de entrada usando Pydantic
class InputData(BaseModel):
    sintomas: str
    diagnostico: str
    procedimento: str
    lista_de_insumos: str

# Inicialize o FastAPI
app = FastAPI()

# Carregar o modelo treinado
modelo = joblib.load('modelo_fraude.pkl')

# Carregar o tokenizer e o modelo BERT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased").to(device)

# Função para gerar embeddings
def gerar_embeddings_lote(textos):
    inputs = tokenizer(textos, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return embeddings

# Rota da API
@app.post("/predict")
async def predict(data: InputData):
    # Gerar embeddings para os dados de entrada
    embeddings_sintomas = gerar_embeddings_lote([data.sintomas])
    embeddings_diagnostico = gerar_embeddings_lote([data.diagnostico])
    embeddings_procedimento = gerar_embeddings_lote([data.procedimento])
    embeddings_insumos = gerar_embeddings_lote([data.lista_de_insumos])

    # Combinar os embeddings
    X = np.hstack([embeddings_sintomas, embeddings_diagnostico, embeddings_procedimento, embeddings_insumos])

    # Prever a chance de fraude usando o modelo treinado
    chance_de_fraude = modelo.predict_proba(X)[0][1]  # Probabilidade da classe 1 (fraude)

    return {"chance_de_fraude": float(chance_de_fraude)}

# Rota de teste para verificar se a API está funcionando
@app.get("/")
async def root():
    return {"message": "API de Previsão de Fraude está funcionando!"}