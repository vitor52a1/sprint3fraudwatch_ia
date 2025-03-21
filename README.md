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

## Uso da API no Postman

1. Abra o Postman.
2. Utilize a URL:
   ```
   http://127.0.0.1:8000/predict
   ```
3. Escolha o método **POST**.
4. No **Body**, selecione **raw** e defina o formato **JSON**.
5. Envie um JSON no seguinte formato:
   ```json
   {
       "sintomas": "Dor de dente leve",
       "diagnostico": "Cárie pequena",
       "procedimento": "Restauração simples",
       "lista_de_insumos": "Resina composta, anestesia local"
   }
   ```

6. A resposta será algo semelhante a:
   ```json
   {
       "chance_de_fraude": 0.3107525730183178
   }
   ```



Este projeto está licenciado sob a MIT License - veja o arquivo LICENSE para mais detalhes.


