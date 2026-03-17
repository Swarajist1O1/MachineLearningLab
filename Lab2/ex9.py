import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("https://raw.githubusercontent.com/rashida048/Datasets/master/StudentsPerformance.csv")

plt.boxplot(df["math score"])
plt.title("Math Score Boxplot")
plt.show()