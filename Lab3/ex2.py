import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/rashida048/Datasets/master/StudentsPerformance.csv")

num = df.select_dtypes('number')

print("Q1:\n", num.quantile(0.25))
print("Q2:\n", num.quantile(0.50))
print("Q3:\n", num.quantile(0.75))