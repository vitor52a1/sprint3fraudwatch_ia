from fastapi import FastAPI
from pydantic import BaseModel
import json
import os
import torch
from transformers import AutoTokenizer, AutoModel
import joblib
import numpy as np

app = FastAPI()

class InputData(BaseModel):
    sintomas: str
    diagnostico: str
    procedimento: str
    lista_de_insumos: str

# Arquivo onde serão salvos os dados do POST /predict
METRICS_FILE = "metrics.json"

# Carregar modelo e tokenizer (supondo que estejam configurados)
modelo = joblib.load('modelo_fraude.pkl')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased").to(device)

def gerar_embeddings_lote(textos):
    inputs = tokenizer(textos, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return embeddings

def salvar_metrics(data):
    # Lê as métricas atuais ou cria uma lista vazia se arquivo não existir
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "r") as f:
            metrics = json.load(f)
    else:
        metrics = []

    # Adiciona os novos dados
    metrics.append(data)

    # Salva novamente no arquivo
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)

@app.post("/predict")
async def predict(data: InputData):
    embeddings_sintomas = gerar_embeddings_lote([data.sintomas])
    embeddings_diagnostico = gerar_embeddings_lote([data.diagnostico])
    embeddings_procedimento = gerar_embeddings_lote([data.procedimento])
    embeddings_insumos = gerar_embeddings_lote([data.lista_de_insumos])

    X = np.hstack([embeddings_sintomas, embeddings_diagnostico, embeddings_procedimento, embeddings_insumos])

    chance_de_fraude = modelo.predict_proba(X)[0][1]

    # Prepare dados para salvar (entrada + resultado)
    dados_para_salvar = data.dict()
    dados_para_salvar["chance_de_fraude"] = float(chance_de_fraude)

    salvar_metrics(dados_para_salvar)

    return {"chance_de_fraude": float(chance_de_fraude)}

@app.get("/metrics")
async def get_metrics():
    if not os.path.exists(METRICS_FILE):
        return {"message": "Nenhum dado armazenado ainda."}

    with open(METRICS_FILE, "r") as f:
        metrics = json.load(f)

    return {"total_requests": len(metrics), "dados": metrics}

@app.get("/")
async def root():
    return {"message": "API de Previsão de Fraude está funcionando!"}
