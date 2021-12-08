from joblib import load
from fastapi import FastAPI, HTTPException
from serviceType import service_type
from Models.SentimentData import SentimentData
from Service.init_csv import initial_csv
import logging
from fastapi.logger import logger
from pythainlp import word_tokenize
from pythainlp.util import normalize
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

logger.info("....load models....")
vector = load("./Algorithm/vectors.joblib")
lr_tf_vector = load('./Algorithm/lr_tf_vectors.joblib')
model = load("./Algorithm/logistic.joblib")
nb_vector = load('./Algorithm/nb_vectors.joblib')
nb_model = load('./Algorithm/naive.joblib')
nb_tf_vector = load('./Algorithm/nb_tf_vectors.joblib')

def loadModel():
  global predict_model
  predict_model = load_model('./Algorithm/lstm.h5')

loadModel()
logger.info('....done....')

df = pd.read_csv('./data/dataandpythai_V2.csv', sep=',').drop_duplicates(subset=['Sentiment', 'SentimentText'], keep=False)
df = df.reset_index(drop=True)
df['Sentiment'] = df['Sentiment'].map({0: 'Negative', 1: 'Neutral'})
sentiment = df.SentimentText.values
category = df.Sentiment.values
unique_category = list(set(category))
logger.info('...cleansing data...')
cleaned_words, temp = cleansing.cleaning(sentiment)
logger.info('...done...')
max_length = cleansing.max_length(temp)
predict_word_tokenizer = cleansing.create_tokenizer(cleaned_words)
encoded_doc = cleansing.encoding_doc(predict_word_tokenizer, cleaned_words)
padded_doc = cleansing.padding_doc(encoded_doc, max_length)
logger.info('api ready....')

def predictLSTM(text):
  clean = normalize(text)
  clean = re.sub(r'[^ก-๙]', "", clean)
  test_word = word_tokenize(clean, engine='attacut')
  test_word = [w.lower() for w in test_word]
  test_ls = predict_word_tokenizer.texts_to_sequences(test_word)
  logger.info(test_word)
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
    logger.info("%s has confidence = %s" % (classes[i], (predictions[i])))
    return classes[i], predictions[i]

@app.get("/")
async def read_root():
    return {"message: Sentiment-Api"}

@app.get("/predict")
async def get_predict(sentimentText: str):
  logger.info(word_tokenize(sentimentText, engine='attacut'))
  guard = service_type(sentimentText)
  text = [sentimentText]
  vec = vector.transform(text)
  tf_idf_vec = lr_tf_vector.transform(vec)
  prediction = model.predict(tf_idf_vec)  
  data = [prediction[0], sentimentText, 'logistic']
  await initial_csv(data)
  return {"Sentiment" : sentimentText, "Predict": prediction[0], "Service Type": guard}

@app.post("/predict-nb")
async def get_predict(sentimentText: str):
  logger.info(word_tokenize(sentimentText, engine='attacut'))
  guard = service_type(sentimentText)
  text = [sentimentText]
  vec = nb_vector.transform(text)
  tf_idf_vec = nb_tf_vector.transform(vec)
  prediction = nb_model.predict(tf_idf_vec)  
  data = [prediction[0], sentimentText, 'naivebayes']
  await initial_csv(data)
  return {"Sentiment" : sentimentText, "Predict": prediction[0], "Service Type": guard}

@app.post('/predict-lstm')
async def get_lstm_predict(sentimentText: str):
  guard = service_type(sentimentText)
  pred = predictLSTM(sentimentText)
  res = get_final_output(pred, unique_category)
  data = [res[0], sentimentText, 'lstm']
  await initial_csv(data)
  return {"Sentiment" : sentimentText, "Predict": res[0], "Service Type": guard}

@app.post("/data", response_model=SentimentData, status_code=200)
async def get_data(sentiment: SentimentData) -> SentimentData:
  logger.info(f"receive data {sentiment}")
  if sentiment is None:
    raise HTTPException(status_code=500, default="Invalid Model")
  data = [sentiment.Sentiment, sentiment.SentimentText]
  await initial_csv(data)
  return sentiment