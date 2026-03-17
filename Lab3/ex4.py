import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("https://raw.githubusercontent.com/rashida048/Datasets/master/StudentsPerformance.csv")

num = df.select_dtypes('number')

# Histograms
num.hist()
plt.show()

# Boxplot
plt.boxplot([num[c] for c in num.columns], labels=num.columns)
plt.show()