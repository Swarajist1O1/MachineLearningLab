import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_curve
from sklearn.preprocessing import label_binarize

sns.set(style="whitegrid")

# 1) Load dataset
df = pd.read_csv("Lab7/iris.data", header=None)
df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# 2) Hyperparameter variation
estimators = [5, 10, 50, 100]
depths = [2, 3, 5, None]

heatmap_data = []

for n in estimators:
    row = []
    for d in depths:
        model = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        row.append(accuracy_score(y_test, y_pred))
    heatmap_data.append(row)

# Heatmap for accuracy
plt.figure(figsize=(8,5))
sns.heatmap(heatmap_data, annot=True, cmap="viridis",
            xticklabels=depths, yticklabels=estimators)
plt.xlabel("max_depth")
plt.ylabel("n_estimators")
plt.title("Accuracy Heatmap (Hyperparameter Tuning)")
plt.show()

# 3) Final model
model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 4) Evaluation

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix (Seaborn Heatmap)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=np.unique(y),
            yticklabels=np.unique(y))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Precision-Recall Curve
y_test_bin = label_binarize(y_test, classes=np.unique(y))
y_score = model.predict_proba(X_test)

plt.figure(figsize=(7,5))
for i in range(y_test_bin.shape[1]):
    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
    sns.lineplot(x=recall, y=precision, label=f"Class {i}")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()