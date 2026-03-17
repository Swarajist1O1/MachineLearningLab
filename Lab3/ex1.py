import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/rashida048/Datasets/master/StudentsPerformance.csv")

print(df.head())

num = df.select_dtypes('number')

print("Mean:\n", num.mean())
print("Median:\n", num.median())
print("Mode:\n", num.mode().iloc[0])

print("Min:\n", num.min())
print("Max:\n", num.max())
print("Sum:\n", num.sum())
print("Var:\n", num.var())
print("Std:\n", num.std())