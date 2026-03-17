import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/rashida048/Datasets/master/StudentsPerformance.csv")

df.info()

print("\nRows, Columns:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())