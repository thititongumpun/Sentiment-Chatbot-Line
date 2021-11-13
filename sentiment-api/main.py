from joblib import load
from fastapi import FastAPI, HTTPException
from serviceType import service_type
from Models.SentimentData import SentimentData
from Service.init_csv import initial_csv
import logging
from fastapi.logger import logger
from pythainlp import word_tokenize
import re
from utils import cleansing
import numpy as np
import tensorflow as tf
import pandas as pd
from logging.config import dictConfig
import logging
from Models.logger import LogConfig
load_model = tf.keras.models.load_model

dictConfig(LogConfig().dict())
logger = logging.getLogger("sentiment-api")

app = FastAPI()

gunicorn_logger = logging.getLogger('gunicorn.error')
logger.handlers = gunicorn_logger.handlers
if __name__ != "main":
    logger.setLevel(gunicorn_logger.level)
else:
    logger.setLevel(logging.DEBUG)

vector = load("./Algorithm/vectors.joblib")
model = load("./Algorithm/logistic.joblib")

def loadModel():
  global predict_model
  predict_model = load_model('./Algorithm/lstm.h5')

loadModel()

df = pd.read_csv('./data/data.csv', sep=',').drop_duplicates(subset=['Sentiment', 'SentimentText'], keep=False)
df = df.reset_index(drop=True)
df['Sentiment'] = df['Sentiment'].map({0: 'Negative', 1: 'Neutral'})
sentiment = df.SentimentText.values
category = df.Sentiment.values
unique_category = list(set(category))
cleaned_words, temp = cleansing.cleaning(sentiment)
max_length = cleansing.max_length(temp)
print('...cleansing data...')
logger.info('...cleansing data...')
cleaned_words, temp = cleansing.cleaning(sentiment)
print('...done...')
logger.info('...done...')
predict_word_tokenizer = cleansing.create_tokenizer(cleaned_words)
encoded_doc = cleansing.encoding_doc(predict_word_tokenizer, cleaned_words)
padded_doc = cleansing.padding_doc(encoded_doc, max_length)

def predictLSTM(text):
  clean = re.sub(r'[^ก-๙]', " ", text)
  test_word = word_tokenize(clean)
  test_word = [w.lower() for w in test_word]
  test_ls = predict_word_tokenizer.texts_to_sequences(test_word)
  print(test_word)
  if [] in test_ls:
    test_ls = list(filter(None, test_ls))
  test_ls = np.array(test_ls).reshape(1, len(test_ls))
  x = cleansing.padding_doc(test_ls, max_length)
  pred = predict_model.predict(x)

  return pred

def get_final_output(pred, classes):
  predictions = pred[0]
  classes = np.array(classes)
  ids = np.argsort(-predictions)
  classes = classes[ids]
  
  predictions = -np.sort(-predictions)
  
  for i in range(pred.shape[1]):
    print("%s has confidence = %s" % (classes[i], (predictions[i])))
    return classes[i], predictions[i]

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

@app.post('/predict-lstm')
async def get_lstm_predict(sentimentText: str):
  guard = service_type(sentimentText)
  pred = predictLSTM(sentimentText)
  res = get_final_output(pred, unique_category)
  return {"Sentiment" : sentimentText, "Predict": res[0], "Service Type": guard}

@app.post("/data", response_model=SentimentData, status_code=200)
async def get_data(sentiment: SentimentData) -> SentimentData:
  logger.info(f"receive data {sentiment}")
  if sentiment is None:
    raise HTTPException(status_code=500, default="Invalid Model")
  data = [sentiment.Sentiment, sentiment.SentimentText]
  await initial_csv(data)
  return sentiment