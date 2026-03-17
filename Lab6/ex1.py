import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_curve, PrecisionRecallDisplay

# 1) Load the dataset and split it [cite: 4]
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

# 2) Train a K-Nearest Neighbour classifier 
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# 3) Predict class labels for the test data [cite: 7]
y_pred = knn.predict(X_test)

# 4) Evaluate the model [cite: 8]
print("--- Experiment 1: KNN Results ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}") # [cite: 9]
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred)) # [cite: 10]
print("\nClassification Report:\n", classification_report(y_test, y_pred)) # [cite: 11]

# Precision-Recall Curve [cite: 12]
PrecisionRecallDisplay.from_estimator(knn, X_test, y_test)
plt.title("KNN Precision-Recall Curve")
plt.savefig('knn_precision_recall.png')
print("Plot saved as knn_precision_recall.png")