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

from pathlib import Path
import pandas as pd

from keras.models import Sequential
from keras import layers
from keras.layers import LeakyReLU
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
import joblib


filePath = Path("F:/IndustryProject/PickleDataCountV8")
scalerPath = Path("F:/IndustryProject/emb_scaler.pkl")
tokenizerPath = Path("F:/IndustryProject/emb_tokenizer.pkl")
epochNum = 100
loss_func = "mean_squared_error"
layer1_size = 512
layer2_size = 256

df = pd.read_pickle(filePath)

text = df['Text'].values
y = df['Sentence'].values
y = y.reshape((len(y), 1))

scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(y)
joblib.dump(scaler, scalerPath)
print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))

y_normalized = scaler.transform(y)


textTrain, textTest, y_train, y_test = train_test_split(text, y_normalized, test_size=0.10, random_state=1000)

tokenizer = Tokenizer(num_words=50000)
tokenizer.fit_on_texts(textTrain)
joblib.dump(tokenizer, tokenizerPath)

X_train = tokenizer.texts_to_sequences(textTrain)
X_test = tokenizer.texts_to_sequences(textTest)

vocab_size = len(tokenizer.word_index) + 1

print(textTrain[2])
print(X_train[2])
print(vocab_size)

maxlen = 8906 #This puts ~95% of all cases without trimming words

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

print(X_train[0, :])

embedding_dim = 512

model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size,
                           output_dim=embedding_dim,
                           input_length=maxlen))
model.add(layers.GlobalMaxPool1D())

# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(32, activation='relu'))
# model.add(layers.Dense(16, activation='relu'))
# model.add(layers.Dense(8, activation='relu'))

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
model.save("F:/IndustryProject/emb_model.h5")
plot_history(history, "F:/IndustryProject/emb_{}e_{}-loss_{}-layer1_{}-layer2".format(epochNum, loss_func, layer1_size, layer2_size))