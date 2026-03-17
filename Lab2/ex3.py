import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/rashida048/Datasets/master/StudentsPerformance.csv")

df.fillna(df.mean(numeric_only=True), inplace=True)

df.fillna("Unknown", inplace=True)

print(df.isnull().sum())