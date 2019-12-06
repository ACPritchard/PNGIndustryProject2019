import matplotlib.pyplot as plt
plt.style.use('ggplot')
from pathlib import Path
import pandas as pd

df = pd.read_csv("F:/IndustryProject/TestingWord2Vec.csv")
results = pd.DataFrame()
results['Actual'] = df['Sentence']
results['Predicted'] = df['Predicted']

print(results.describe())

results.boxplot()
plt.savefig("F:/IndustryProject/Word2Vec_PredictedVsActual_boxplot.png")

results.hist()
plt.savefig("F:/IndustryProject/Word2Vec_PredictedVsActual_hist.png")

