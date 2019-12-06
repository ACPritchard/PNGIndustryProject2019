import urllib.request, os
import nltk
#nltk.download('stopwords')
#nltk.download('punkt')
from pathlib import Path
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk import word_tokenize
import json

corpus = []
stopWords = set(stopwords.words('english'))

for year in range(1974, 2020):
    folderPath = Path("F:/IndustryProject/DataV4/{}/".format(year))
    for path in folderPath.iterdir():
        soup = BeautifulSoup(open(path), "html.parser")
        corpus.append(soup.get_text().lower())

corpus_tokens = []

for text in corpus:
    tokens = word_tokenize(text, language='english')
    filtered_tokens = [w for w in tokens if not w in stopWords]
    corpus_tokens.append(filtered_tokens)

#testTokenize = word_tokenize(corpus[0], language='english')
print(len(corpus_tokens))

unique_tokens = set(())
i = 0
for token_list in corpus_tokens:
    unique_tokens.update(set(token_list))

tokensList = []

for token in unique_tokens:
    tokensList.append(token)

with open(Path("F:/IndustryProject/testJson.json"),'w') as file:
    json.dump(tokensList,file)