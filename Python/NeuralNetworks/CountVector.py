from keras.models import Sequential
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from keras.utils import plot_model
import joblib

import matplotlib.pyplot as plt
plt.style.use('ggplot')

def plot_history(history, plt_name):
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
    plt.savefig(plt_name)
    plt.show()

filePath = Path("F:/IndustryProject/PickleDataCountV8")
saveModel = Path("F:/IndustryProject/CV.pkl")
scalerPath = Path("F:/IndustryProject/scaler.pkl")

epochsNum = 100
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

vectorizer = CountVectorizer()
vectorizer.fit(textTrain)
X_train = vectorizer.transform(textTrain)
X_test  = vectorizer.transform(textTest)
joblib.dump(vectorizer, saveModel)

# classifier = LogisticRegression()
# classifier.fit(X_train, y_train)
# score = classifier.score(X_test, y_test)
#
# print("Accuracy:", score)

input_dim = X_train.shape[1]  # Number of features
print(input_dim)
model = Sequential()
model.add(layers.Dense(layer1_size, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(layer2_size, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(
    loss=loss_func,
    optimizer='adam',
    metrics=['accuracy'])
model.summary()


history = model.fit(
    X_train, y_train,
    epochs=epochsNum,
    verbose=False,
    validation_data=(X_test, y_test),
    batch_size=10)

loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
model.save("F:/IndustryProject/CountVector.h5")
plot_model(model, to_file='model.png')

plot_history(history, "F:/IndustryProject/countVec_{}e_{}-loss_{}-layer1_{}-layer2".format(epochsNum, loss_func, layer1_size, layer2_size))