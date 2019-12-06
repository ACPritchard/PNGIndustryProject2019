import pandas as pd
from nltk.corpus import stopwords
from textblob import Word
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import joblib

modelPath = "./static/homepage/CountVector.h5"
vectorsPath = "./static/homepage/CV.pkl"
scalarPath = "./static/homepage/scaler.pkl"
filePath = "F:/IndustryProject/PickleDataCountV8"

model = load_model(modelPath)
vectors = joblib.load(vectorsPath)
scaler = joblib.load(scalarPath)
df = pd.read_pickle(filePath)

inputVectors = vectors.transform(df['Text'])
output = model.predict(inputVectors)
df['Predicted'] = scaler.inverse_transform(output)

df[['Path', 'WordCount', 'CharCount', 'StopWords', 'Sentence', 'Predicted']].to_csv("F:/IndustryProject/TestingCountVec.csv")