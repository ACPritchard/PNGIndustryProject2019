import pandas as pd
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import joblib

modelPath = "F:/IndustryProject/word2vec_model.h5"
tokenizerPath = "F:/IndustryProject/word2vec_tokenizer.pkl"
scalarPath = "F:/IndustryProject/word2vec_scaler.pkl"
filePath = "F:/IndustryProject/PickleDataCountV8"

model = load_model(modelPath)
tokenizer = joblib.load(tokenizerPath)
scaler = joblib.load(scalarPath)
df = pd.read_pickle(filePath)


inputVectors = tokenizer.texts_to_sequences(df['Text'])
inputVectors = pad_sequences(inputVectors, padding='post', maxlen=8906)
output = model.predict(inputVectors)
df['Predicted'] = scaler.inverse_transform(output)

df[['Path', 'WordCount', 'CharCount', 'StopWords', 'Sentence', 'Predicted']].to_csv("F:/IndustryProject/TestingWord2Vec.csv")