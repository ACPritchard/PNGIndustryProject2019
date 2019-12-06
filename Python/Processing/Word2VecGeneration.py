import gensim
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer

model = gensim.models.KeyedVectors.load_word2vec_format('F:/IndustryProject/GoogleNews-vectors-negative300.bin', binary=True, limit=500000)

df = pd.read_pickle("F:/IndustryProject/PickleDataCountV8")
text = df['Text'].values

tokenizer = Tokenizer(num_words=50000)
tokenizer.fit_on_texts(text)

words = tokenizer.word_index
vocab_size = len(words) + 1

word2vec = {'Word':[], 'Vector':[]}

for word in words:
    if word in model.vocab:
        word2vec['Word'].append(word)
        word2vec['Vector'].append(model.word_vec(word))
#print(word2vec['year'])
df_word2vec = pd.DataFrame(word2vec)
print(df_word2vec.head())
df_word2vec.to_pickle("F:/IndustryProject/Word2Vec_df.pkl")
#print(len(word2vec))
print(vocab_size)