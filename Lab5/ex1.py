import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split as t
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import *

X,y=load_breast_cancer(return_X_y=True)
Xtr,Xte,ytr,yte=t(X,y,test_size=.2,random_state=42)

m=GaussianNB().fit(Xtr,ytr)
yp=m.predict(Xte); yp2=m.predict_proba(Xte)[:,1]

print("Naive Bayes - Breast Cancer")
print("Acc:",accuracy_score(yte,yp))
print(confusion_matrix(yte,yp))

p,r,_=precision_recall_curve(yte,yp2)
plt.plot(r,p);plt.xlabel("Recall");plt.ylabel("Precision");plt.show()