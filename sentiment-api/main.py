from joblib import load
from fastapi import FastAPI, HTTPException
from serviceType import service_type
from Service.init_csv import initial_csv
import logging
from fastapi.logger import logger
from pythainlp.util import normalize
from utils import cleansing, predict_output
import numpy as np
import tensorflow as tf
import pandas as pd
from logging.config import dictConfig
from Models.logger import LogConfig
from Models.SentimentData import SentimentData
from sklearn.utils import resample, shuffle
from pythainlp.ulmfit import *

load_model = tf.keras.models.load_model

dictConfig(LogConfig().dict())
logger = logging.getLogger("sentiment-api")

app = FastAPI()
logger.info('Starting application')

logger.info("....load models....")
def loadModel():
  global predict_model, vector, lr_tf_vector, model, nb_vector, nb_model, nb_tf_vector
  predict_model = load_model('./Algorithm/lstm.h5')
  vector = load("./Algorithm/vectors.joblib")
  lr_tf_vector = load('./Algorithm/lr_tf_vectors.joblib')
  model = load("./Algorithm/logistic.joblib")
  nb_vector = load('./Algorithm/nb_vectors.joblib')
  nb_model = load('./Algorithm/naive.joblib')
  nb_tf_vector = load('./Algorithm/nb_tf_vectors.joblib')
loadModel()
logger.info('....done....')

logger.info('...initial data...')
df = pd.read_csv('./data/dataandpythai_V2.csv', sep=',').drop_duplicates(subset=['Sentiment', 'SentimentText'], keep=False)
df = df.reset_index(drop=True)
df['Sentiment'] = df['Sentiment'].map({0: 'Negative', 1: 'Neutral'})
df['SentimentText'] = df['SentimentText'].apply(cleansing.text_process)
nan_value = float("NaN")
df.replace("", nan_value, inplace=True)
df.dropna(subset = ["SentimentText"], inplace=True)
logger.info('....done....')

logger.info('...cleansing data...')
max_length=484
unique_category = ['Neutral', 'Negative']
# unique_category = ['Negative', 'Neutral']
predict_word_tokenizer = cleansing.create_tokenizer(df['SentimentText'])
logger.info('...done...')
logger.info('api ready....')

def predictLSTM(text):
  text = normalize(text)
  test_word = process_thai(text,
                            pre_rules=[replace_rep_after, fix_html, rm_useless_spaces],
                            post_rules=[ungroup_emoji,
                            replace_wrep_post_nonum,
                            remove_space]
                          )
  test_ls = predict_word_tokenizer.texts_to_sequences(test_word)
  logger.info(test_word)
  if [] in test_ls:
    test_ls = list(filter(None, test_ls))
  test_ls = np.array(test_ls).reshape(1, len(test_ls))
  x = cleansing.padding_doc(test_ls, max_length)
  pred = predict_model.predict(x)

  return pred

@app.get("/")
async def read_root():
    return {"message: Sentiment-Api"}

@app.get("/predict")
async def get_predict(sentimentText: str):
  sentimentText = normalize(sentimentText)
  sentimentTextToken = cleansing.cleansingTextToken(sentimentText)
  logger.info(sentimentTextToken)
  guard = service_type(sentimentText)
  text = [sentimentText]
  vec = vector.transform(text)
  tf_idf_vec = lr_tf_vector.transform(vec)
  prediction = model.predict(tf_idf_vec)
  data = [prediction[0], sentimentText, 'logistic']
  await initial_csv(data)
  return {"Sentiment" : sentimentText, "Tokenize": sentimentTextToken, "Predict": prediction[0], "Service Type": guard}

@app.post("/predict-nb")
async def get_predict(sentimentText: str):
  sentimentText = normalize(sentimentText)
  sentimentTextToken = cleansing.cleansingTextToken(sentimentText)
  logger.info(sentimentTextToken)
  guard = service_type(sentimentText)
  text = [sentimentText]
  vec = nb_vector.transform(text)
  tf_idf_vec = nb_tf_vector.transform(vec)
  prediction = nb_model.predict(tf_idf_vec)
  data = [prediction[0], sentimentText, 'naivebayes']
  await initial_csv(data)
  return {"Sentiment" : sentimentText, "Tokenize": sentimentTextToken, "Predict": prediction[0], "Service Type": guard}

@app.post('/predict-lstm')
async def get_lstm_predict(sentimentText: str):
  sentimentText = normalize(sentimentText)
  sentimentTextToken = cleansing.cleansingTextToken(sentimentText)
  guard = service_type(sentimentText)
  pred = predictLSTM(sentimentText)
  res = predict_output.get_final_output(pred, unique_category)
  confidence = predict_output.get_output_confident_type(pred, unique_category)
  confidenceSubType = predict_output.get_output_confident_opposite_type(pred, unique_category)
  data = [res[0], sentimentText, 'lstm']
  await initial_csv(data)
  return {"Sentiment" : sentimentText, "Tokenize": sentimentTextToken, "Predict": res[0], "Service Type": guard, "Confidence": confidence, "ConfidenceSubType": confidenceSubType}

@app.post("/data", response_model=SentimentData, status_code=200)
async def get_data(sentiment: SentimentData) -> SentimentData:
  logger.info(f"receive data {sentiment}")
  if sentiment is None:
    raise HTTPException(status_code=500, default="Invalid Model")
  data = [sentiment.Sentiment, sentiment.SentimentText]
  await initial_csv(data)
  return sentiment