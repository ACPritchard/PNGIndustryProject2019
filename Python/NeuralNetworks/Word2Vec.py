import matplotlib.pyplot as plt
plt.style.use('ggplot')

def plot_history(history, plt_path):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(plt_path)
    plt.show()

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from keras.models import Sequential
from keras import layers
from keras.layers import LeakyReLU
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer

filePath = Path("F:/IndustryProject/PickleDataCountV8")
scalerPath = Path("F:/IndustryProject/word2vec_scaler.pkl")
tokenizerPath = Path("F:/IndustryProject/word2vec_tokenizer.pkl")
word2vecPath = Path("F:/IndustryProject/Word2Vec_df.pkl")
plotPath = "F:/IndustryProject/Word2Vec_test.png"
modelPath = "F:/IndustryProject/word2vec_model.h5"

epochNum = 20
loss_func = "mean_squared_error"
layer1_size = 64
layer2_size = 32
emb_dim = 300
maxlen = 8906

df = pd.read_pickle(filePath)
text = df['Text'].values

tokenizer = Tokenizer(num_words=50000)
tokenizer.fit_on_texts(text)
joblib.dump(tokenizer, tokenizerPath)
vocab_size = len(tokenizer.word_index) + 1

embedding_matrix = np.zeros((vocab_size, emb_dim))
word2vec_df= pd.read_pickle(word2vecPath)
word2vec_list = word2vec_df['Word'].tolist()
word_index = tokenizer.word_index

for word in word_index:
    #print(word in word2vec_df['Word'])
    if word in word2vec_list:
        idx = word_index[word]
        #print(idx)
        embedding_matrix[idx] = np.array(
            word2vec_df[word2vec_df['Word'] == word]['Vector'].tolist(), dtype=np.float32)[:emb_dim]

nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))

text = df['Text'].values
y = df['Sentence'].values
y = y.reshape((len(y), 1))

scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(y)
joblib.dump(scaler, scalerPath)

y_normalized = scaler.transform(y)
textTrain, textTest, y_train, y_test = train_test_split(text, y_normalized, test_size=0.10, random_state=1000)

X_train = tokenizer.texts_to_sequences(textTrain)
X_test = tokenizer.texts_to_sequences(textTest)
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
#print(embedding_matrix[:,0])

model = Sequential()
model.add(layers.Embedding(vocab_size, emb_dim,
                           #weights=[embedding_matrix],
                           input_length=maxlen,
                           trainable=True))
model.add(layers.Conv1D(filters=128, kernel_size=4, activation='relu'))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(layer1_size, kernel_initializer='he_normal'))
model.add(LeakyReLU())
model.add(layers.Dense(layer2_size, kernel_initializer='he_normal'))
model.add(LeakyReLU())
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss=loss_func,
              metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train,
                    epochs=epochNum,
                    verbose=False,
                    validation_data=(X_test, y_test),
                    batch_size=10)
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
model.save(modelPath)
plot_history(history, "F:/IndustryProject/Word2VecConv_{}-epoch.png".format(epochNum))