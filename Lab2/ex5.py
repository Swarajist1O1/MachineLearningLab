import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/rashida048/Datasets/master/StudentsPerformance.csv")

df.rename(columns={
    "math score": "Math",
    "reading score": "Reading",
    "writing score": "Writing"
}, inplace=True)

print(df.head())