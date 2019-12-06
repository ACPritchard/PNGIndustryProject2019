import pandas as pd
from nltk.corpus import stopwords
from textblob import Word
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import joblib

def text_preprocessing(text):
    #text = request.POST['info']
    # import stop words
    stop = stopwords.words('english')
    # setup input data
    data = {'Text': []}
    data['Text'].append(text)
    df = pd.DataFrame(data)
    # lower case all characters
    df['Text'] = df['Text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    # remove punctuation
    df['Text'] = df['Text'].str.replace('[^\w\s]', '')
    # remove stopwords
    df['Text'] = df['Text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    # lemmatize remaing words
    df['Text'] = df['Text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

    # result = CountVecModel(df['Text'])
    # result = EmbeddingsModel(df['Text'])
    result = Word2VecModel(df['Text'])

    formatedText = "The AI predicts a sentence of {:.2f} months".format(result)
    return formatedText

def CountVecModel(input_data):
    # path strings to the needed files
    modelPath = "./static/homepage/CountVector.h5"
    vectorsPath = "./static/homepage/CV.pkl"
    scalarPath = "./static/homepage/scaler.pkl"

    # load the trained neural network
    model = load_model(modelPath)
    # load the CountVectors from the training
    vectors = joblib.load(vectorsPath)
    # load the min-max scaler that was used to scaler the targer variable in training so we can invert it
    scaler = joblib.load(scalarPath)
    inputVectors = vectors.transform(input_data)

    output = model.predict(inputVectors)
    output = scaler.inverse_transform(output)
    return output[0][0]

def EmbeddingsModel(input_data):
    # path strings to the needed files
    modelPath = "./static/homepage/emb_model.h5"
    tokenizerPath = "./static/homepage/emb_tokenizer.pkl"
    scalarPath = "./static/homepage/emb_scaler.pkl"

    model = load_model(modelPath)
    tokenizer = joblib.load(tokenizerPath)
    scaler = joblib.load(scalarPath)

    inputVectors = tokenizer.texts_to_sequences(input_data)
    inputVectors = pad_sequences(inputVectors, padding='post', maxlen=8906)
    output = model.predict(inputVectors)
    output = scaler.inverse_transform(output)
    return output[0][0]

def Word2VecModel(input_data):
    # path strings to the needed files
    modelPath = "./static/homepage/Word2Vec_model.h5"
    tokenizerPath = "./static/homepage/Word2Vec_tokenizer.pkl"
    scalarPath = "./static/homepage/Word2Vec_scaler.pkl"

    model = load_model(modelPath)
    tokenizer = joblib.load(tokenizerPath)
    scaler = joblib.load(scalarPath)

    inputVectors = tokenizer.texts_to_sequences(input_data)
    inputVectors = pad_sequences(inputVectors, padding='post', maxlen=8906)
    output = model.predict(inputVectors)
    output = scaler.inverse_transform(output)
    return output[0][0]