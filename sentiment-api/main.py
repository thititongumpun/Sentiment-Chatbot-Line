from joblib import load
from fastapi import FastAPI, HTTPException
from serviceType import service_type
from Models.SentimentData import SentimentData
from Service.init_csv import initial_csv
from fastapi.logger import logger

app = FastAPI()

vector = load("vectors.joblib")
model = load("model.joblib")

@app.get("/")
async def read_root():
    return {"message: Sentiment-Api"}

@app.get("/predict")
async def get_predict(sentimentText: str):
  guard = service_type(sentimentText)
  text = [sentimentText]
  vec = vector.transform(text)
  prediction = model.predict(vec)  
  return {"Sentiment" : sentimentText, "Predict": prediction[0], "Service Type": guard}

@app.post("/data", response_model=SentimentData, status_code=200)
async def get_data(sentiment: SentimentData) -> SentimentData:
  logger.info(f"receive data {sentiment}")
  if sentiment is None:
    raise HTTPException(status_code=500, default="Invalid Model")
  data = [sentiment.Sentiment, sentiment.SentimentText]
  await initial_csv(data)
  return sentiment

# from pydantic import BaseModel
# import tensorflow as tf
# load_model = tf.keras.models.load_model
# import numpy as np
# class Data(BaseModel):
#     text: str

# def loadModel():
#     global predict_model

#     predict_model = load_model('./Algorithm/lstm.h5')

# loadModel()

# async def predict(text):
#     clean = re.sub(r'[^ก-๙]', " ", text)
#     test_word = word_tokenize(clean)
#     test_word = [w.lower() for w in test_word]
#     test_ls = train_word_tokenizer.texts_to_sequences(test_word)
#     print(test_word)
#     if [] in test_ls:
#       test_ls = list(filter(None, test_ls))
      
#     test_ls = np.array(test_ls).reshape(1, len(test_ls))
  
#     x = padding_doc(test_ls, max_length)
    
#     pred = predict_model.predict(x)
  
  
#   return pred

# @app.post('/getclass/')
# async def get_class(data: Data):
#     category, confidence = await predict(data)
#     res = {'class': category, 'confidence':confidence}
#     return {'results': res}
