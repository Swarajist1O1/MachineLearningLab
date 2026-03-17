import pandas as pd

data = {"Name":["A","B","C"], "Marks":[85,90,78]}
df = pd.DataFrame(data)

print(df)
print("\nData Types:\n", df.dtypes)