import pandas as pd
from pythainlp.ulmfit import *
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Embedding, Dropout, BatchNormalization, SpatialDropout1D
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report

data = pd.read_csv('./dataandpythai_V2.csv', sep=',')
data = data.drop_duplicates(subset=['SentimentText', 'Sentiment'], keep=False)
data = data.reset_index(drop=True)
data = data[['SentimentText','Sentiment']]
data['Sentiment'] = data['Sentiment'].map({0: 'Negative', 1: 'Neutral'})

def text_process(text):
    words = re.sub(r'[^ก-๙]', '', text)
    words = process_thai(words)
    words = ' '.join(word for word in words)
    return words
data['SentimentText'] = data['SentimentText'].apply(text_process)
nan_value = float("NaN")
data.replace("", nan_value, inplace=True)
data.dropna(subset = ["SentimentText"], inplace=True)

max_length = 484
def create_tokenizer(words, filters = ''):
    token = Tokenizer(filters=filters)
    token.fit_on_texts(words)
    return token
def encoding_doc(token, words):
    return(token.texts_to_sequences(words))
def padding_doc(encoded_doc, max_length):
    return(pad_sequences(encoded_doc, maxlen = max_length, padding = "post"))
def one_hot(encode):
    oh = OneHotEncoder(sparse = False)
    return(oh.fit_transform(encode))

train_word_tokenizer = create_tokenizer(data['SentimentText'])
vocab_size = len(train_word_tokenizer.word_index) + 1
encoded_doc = encoding_doc(train_word_tokenizer, data['SentimentText'])

unique_category = list(set(data['Sentiment']))
unique_category
output_tokenizer = create_tokenizer(unique_category)
encoded_output = encoding_doc(output_tokenizer, data['Sentiment'])
output_one_hot = one_hot(encoded_output)
padded_doc = padding_doc(encoded_doc, max_length)

X_train, X_test, Y_train, Y_test = train_test_split(padded_doc, output_one_hot, test_size = 0.20, random_state = 42, shuffle=True, stratify=output_one_hot)
num_classes = len(unique_category)

adam = Adam(learning_rate=0.0001)
def create_model(vocab_size, max_length):
  model = Sequential()
  model.add(Embedding(vocab_size, 128, input_length = max_length, trainable = True))
  model.add(Bidirectional(LSTM(196), merge_mode='concat'))
  model.add(Dense(128, activation = "relu"))
  model.add(Dropout(0.4))
  model.add(BatchNormalization())
  model.add(Dense(num_classes, activation = "softmax"))
  return model

model = create_model(vocab_size, max_length)
model.compile(loss = "categorical_crossentropy", optimizer = adam, metrics = ["accuracy"])

from time import time
start = time()
EPOCHS = 15
BS = 50
filename = './lstm.h5'
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
hist = model.fit(X_train, Y_train, epochs = EPOCHS, batch_size = BS, validation_data = (X_test, Y_test), callbacks = [checkpoint, es])
print(time()-start)