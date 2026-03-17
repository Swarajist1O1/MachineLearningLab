import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/rashida048/Datasets/master/StudentsPerformance.csv")

num = df.select_dtypes('number')

print("Correlation:\n", num.corr())
print("\nCovariance:\n", num.cov())