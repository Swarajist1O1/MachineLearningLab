import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("https://raw.githubusercontent.com/rashida048/Datasets/master/StudentsPerformance.csv")

df[["math score","reading score","writing score"]].head(20).plot()
plt.title("Score Trends")
plt.show()

df["gender"].value_counts().plot(kind="bar")
plt.title("Gender Count")
plt.show()