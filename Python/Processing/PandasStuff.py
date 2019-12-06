import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk import word_tokenize
from textblob import Word

stop = stopwords.words('english')
root = Path("F:/IndustryProject")
picklePath = root / 'PickleData'
countsPath = root / 'PickleDataCounts'
lowerCasePath = root / 'PickleDataCountsV2'
alphaPath = root / 'PickleDataCountsV3'
alphaCSVPath = root / 'PickleDataCountsV3.csv'
betaCSVPath = root / 'sentence.csv'
stopWordPath = root / 'PickleDataCountsV4'
uncommonWordPath = root / 'PickleDataCountsV5'
frequentWordPath = root / 'PickleDataCountsV6'
lemmatizedPath = root / 'PickleDataCountsV7'
sentencePath = root / 'PickleDataCountV8'

data = {'Path':[], 'Text':[]}

# df = pd.read_pickle(alphaPath)
# df_csv = df[['Path', 'WordCount', 'CharCount', 'StopWords']]
# df_csv.to_csv(alphaCSVPath)

df = pd.read_pickle(lemmatizedPath)
csv_df = pd.read_csv(alphaCSVPath)

print(df.head())
print(csv_df.head())

df['Sentence'] = csv_df['Sentence']

print(df.head())

df = df[df.Sentence != '*']

print(df.head())
print(len(df.Sentence))

df = df[pd.isnull(df.Sentence).__invert__()]
print(df.head())
print(len(df.Sentence))

df.to_pickle(sentencePath)
# df_csv = df[['Path', 'WordCount', 'CharCount', 'StopWords', 'Sentence']]
# df_csv.to_csv(betaCSVPath)

# #### Create DataFrame ####
# for year in range(1974, 2020):
#     folderPath = Path("F:/IndustryProject/DataV4/{}/".format(year))
#     for path in folderPath.iterdir():
#         soup = BeautifulSoup(open(path), "html.parser")
#         data['Path'].append(path)
#         data['Text'].append(soup.get_text())
#         #print(soup.get_text)
# df = pd.DataFrame(data)
# df.to_pickle(picklePath)

# df = pd.read_pickle(picklePath)
#
# #### Word Count ####
# df['WordCount'] = df['Text'].apply(lambda x: len(str(x).split(" ")))
# print(df[['Path','WordCount']].head())
#
# #### Character Count (Including Spaces) ####
# df['CharCount'] = df['Text'].str.len()
# print(df[['Path','WordCount','CharCount']].head())
#
# #### Stop Word Count ####
# df['StopWords'] = df['Text'].apply(lambda x: len([x for x in x.split() if x in stop]))
# print(df[['Path','StopWords']].head())
# df.to_pickle(countsPath)

# #### Lower Case the text ####
# df = pd.read_pickle(countsPath)
# print(df[['Path','Text']].head())
# df['Text'] = df['Text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
# print(df[['Path','Text']].head())
# df.to_pickle(lowerCasePath)


# #### Remove characters that are not Alphanumeric ####
# df = pd.read_pickle(lowerCasePath)
# print(df[['Path','Text']].head())
# df['Text'] = df['Text'].str.replace('[^\w\s]','')
# print(df[['Path','Text']].head())
# df.to_pickle(alphaPath)


# #### Remove Stop Words ####
# df = pd.read_pickle(alphaPath)
# print(df[['Path','Text']].head())
# df['Text'] = df['Text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
# print(df[['Path','Text']].head())
# df.to_pickle(stopWordPath)

# #### Remove Uncommon Words ####
# df = pd.read_pickle(stopWordPath)
#
# freq = pd.Series(' '.join(df['Text']).split()).value_counts()
# freq = freq[freq == 1]
# uncommonWords = freq.keys()
# print(uncommonWords)
#
#
# print(df[['Path','Text']].head())
# df['Text'] = df['Text'].apply(lambda x: " ".join(x for x in x.split() if x not in uncommonWords))
# print(df[['Path','Text']].head())
#
# df.to_pickle(uncommonWordPath)

# #### Remove Common Words ####
# df = pd.read_pickle(uncommonWordPath)
#
# freq = pd.Series(' '.join(df['Text']).split()).value_counts()[:10] #5 chosen as this removes 'state', 'court', 'v', 'sentence' and 'case'. Next was 'deceased'
# print(freq)
# freq = list(freq.index)
# print(df[['Path','Text']].head())
# df['Text'] = df['Text'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
# print(df[['Path','Text']].head())
#
# df.to_pickle(frequentWordPath)

# #### Tokenize and Lemmatization ####
# df = pd.read_pickle(frequentWordPath)
#
# print(df[['Path','Text']].head())
# df['Text'] = df['Text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
# print(df[['Path','Text']].head())
#
# df.to_pickle(lemmatizedPath)

