import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split as t
from sklearn.svm import SVC
from sklearn.metrics import *
from sklearn.decomposition import PCA

X,y=load_breast_cancer(return_X_y=True)
Xtr,Xte,ytr,yte=t(X,y,test_size=.3,random_state=42,stratify=y)

m=SVC(kernel='linear',probability=True).fit(Xtr,ytr)
yp=m.predict(Xte); yp2=m.predict_proba(Xte)[:,1]

print("SVM - Breast Cancer")
print("Acc:",accuracy_score(yte,yp))
print(confusion_matrix(yte,yp))
print(classification_report(yte,yp))

p,r,_=precision_recall_curve(yte,yp2)
plt.plot(r,p);plt.xlabel("Recall");plt.ylabel("Precision");plt.show()