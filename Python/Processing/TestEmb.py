import pandas as pd
from nltk.corpus import stopwords
from textblob import Word
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import joblib

modelPath = "./static/homepage/emb_model.h5"
tokenizerPath = "./static/homepage/emb_tokenizer.pkl"
scalarPath = "./static/homepage/emb_scaler.pkl"
filePath = "F:/IndustryProject/PickleDataCountV8"

model = load_model(modelPath)
tokenizer = joblib.load(tokenizerPath)
scaler = joblib.load(scalarPath)
df = pd.read_pickle(filePath)


inputVectors = tokenizer.texts_to_sequences(df['Text'])
inputVectors = pad_sequences(inputVectors, padding='post', maxlen=8906)
output = model.predict(inputVectors)
df['Predicted'] = scaler.inverse_transform(output)

df[['Path', 'WordCount', 'CharCount', 'StopWords', 'Sentence', 'Predicted']].to_csv("F:/IndustryProject/TestingEmb.csv")