import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/rashida048/Datasets/master/StudentsPerformance.csv")

print(df.dtypes)

df["math score"] = df["math score"].astype(float)

print(df.dtypes)