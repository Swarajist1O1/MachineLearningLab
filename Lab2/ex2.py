import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/rashida048/Datasets/master/StudentsPerformance.csv")

print("Before:", df.shape)

df_clean = df.dropna()

print("After:", df_clean.shape)