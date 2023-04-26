import pandas as pd
import matplotlib.pyplot as plt

df= pd.read_csv(r"data1.csv")
print(df)
treino = df.sample(frac=0.8)
teste = df.drop(treino.index)

plt.scatter(df.iloc[:,0],df.iloc[:,1])
plt.show()
