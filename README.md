Link do github da Sprint 1 2: https://github.com/yagoluucas/sprint2_fraudwatch_ia

Link github da Sprint 3: https://github.com/vitor52a1/sprint3fraudwatch_ia

Link do vídeo: https://youtu.be/Zbomiqwn3Lo

Link do dataset: https://drive.google.com/file/d/1JRJReJmWt_1qTdtrkjpq4-KrsPHaBo97/view?usp=sharing

RM553542 Luiz Otávio - 2tdspr /
RM553483 Vitor de Melo - 2tdspr /
RM553748 Mauricio Pereira - 2tdspc

# API de Detecção de Fraudes em Procedimentos Odontológicos

Este repositório contém uma API desenvolvida com FastAPI para detectar possíveis fraudes em procedimentos odontológicos. A API utiliza embeddings gerados a partir de um modelo BERT para analisar os dados fornecidos e retornar a probabilidade de fraude.

## Tecnologias Utilizadas

- **FastAPI**: Para criar a API.
- **Pandas**: Para manipulação de dados.
- **Torch**: Para trabalhar com PyTorch e carregar o modelo BERT.
- **Transformers**: Para carregar o tokenizer e o modelo BERT.
- **Pydantic**: Para definir modelos de dados e validação.
- **Joblib**: Para carregar o modelo treinado.
- **NumPy**: Para manipulação de arrays numéricos.
- **Uvicorn**: Para rodar o servidor FastAPI.

## Instalação


1. Instale as dependências:
   ```sh
   pip install fastapi pandas torch transformers pydantic joblib numpy uvicorn
   ```

2. Anexe o arquivo **dados_dentista_reduzido_10000.csv** à pasta do projeto Python. (Devido ao tamanho do dataset colocamos ele no Google Drive, ele não está incluído no repositório.)

## Execução da API

3. Inicie o servidor FastAPI pelo terminal do Python:
   ```sh
   uvicorn main:app --reload
   ```

4. Acesse a documentação da API no navegador:
   ```
   http://127.0.0.1:8000/docs
   ```

## Endpoints 
- GET / Testa se a API está online
- POST /predict Envia os dados para prever chance de fraude
- GET /metrics Retorna dados e métricas armazenadas das requisições feitas.

## Uso da API no Postman

1. **Testar API online**

   - Método: `GET`  
   - URL: `http://127.0.0.1:8000/`  

2. **Fazer previsão**

   - Método: `POST`  
   - URL: `http://127.0.0.1:8000/predict`  
   - Body → raw → JSON:

```json
{
  "sintomas": "Dor de dente leve",
  "diagnostico": "Cárie pequena",
  "procedimento": "Restauração simples",
  "lista_de_insumos": "Resina composta, anestesia local"
}
```
Resposta exemplo

```json{
  "chance_de_fraude": 0.31
}
```
3. **Consultar métricas e histórico das requisições**
   - Método: `GET`  
   - URL: `http://127.0.0.1:8000/metrics`  

Resposta exemplo
```json
{
  "total_requisicoes": 3,
  "media_chance_de_fraude": 0.42,
  "detalhes": [
    {
      "sintomas": "Dor de dente leve",
      "diagnostico": "Cárie pequena",
      "procedimento": "Restauração simples",
      "lista_de_insumos": "Resina composta, anestesia local",
      "chance_de_fraude": 0.5
    }
  ]
}
```

# Reflexão e Avaliação do Projeto de API de Previsão de Fraude

## 1. Demonstração da Versão Final do Projeto

O projeto consiste em uma API de previsão de fraude baseada em dados, utilizando um modelo treinado e embeddings de linguagem natural extraídos via BERT. A API está implementada em FastAPI, permitindo receber dados via POST, fazer previsões e armazenar as informações de uso e métricas localmente para acompanhamento. Também há endpoints para consultar métricas e verificar o status da API.

As funcionalidades principais incluem:

- Recebimento de dados via JSON para análise (sintomas, diagnóstico, procedimento, lista de insumos).
- Geração de embeddings para representar os dados.
- Predição da probabilidade de fraude com um modelo treinado.
- Armazenamento das predições para geração de métricas.
- Endpoints para consulta das métricas e status da API.
- Utilização local via Postman para testes e desenvolvimento.

## 2. Autocrítica (Reflexão sobre o processo de desenvolvimento)

### O que funcionou:

- Implementação bem-sucedida do modelo de predição dentro da API.
- Integração do BERT para geração de embeddings, permitindo tratar dados complexos.
- Armazenamento dinâmico dos dados e métricas gerados durante as requisições.
- Uso do FastAPI, que facilitou a criação de endpoints e documentação automática.
- Testes locais via Postman para validar as respostas da API.

### O que não funcionou ou desafios:

- Inicialmente houve dificuldade em integrar o processamento dos dados de forma eficiente; nas primeiras tentativas, o CSV gerado era enorme, dificultando os testes devido ao tempo de compilação que ficava inviável.
- Ausência de autenticação e controle de acesso, deixando a API vulnerável.
- Implementação diretamente na aplicação sem integrações adequadas, o que dificultou a execução e exigiu rodar localmente para evitar transtornos no produto final.
- Falta de monitoramento e logs detalhados para facilitar a manutenção em produção.

### O que foi aprendido:

- A importância de escolher a arquitetura correta para persistência dos dados (arquivo vs banco de dados).
- Como trabalhar com embeddings e modelos pré-treinados.
- Utilização do FastAPI para criar APIs REST de forma rápida e eficiente.
- Como estruturar e organizar endpoints para funcionalidades distintas.

### O que faria diferente:

- Planejar desde o início um banco de dados para métricas e histórico, garantindo escalabilidade.
- Implementar autenticação básica para proteger os endpoints.
- Criar testes automatizados para garantir a qualidade do código.
- Investir em monitoramento e logging desde a fase inicial.

## 3. Planos para o futuro do projeto

- Migrar o armazenamento de métricas para um banco de dados como MongoDB para maior eficiência, por exemplo.
- Adicionar autenticação e autorização para proteger o acesso à API.
- Implementar monitoramento com ferramentas como Prometheus e dashboards visuais (Grafana).
- Adicionar explicações interpretáveis para as predições, aumentando a transparência.
- Expandir a API para suportar múltiplos modelos e versões, com controle de versão.
- Otimizar o processamento e permitir predições em lote para maior desempenho.

