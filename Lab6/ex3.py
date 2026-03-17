import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split as t
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler

X,y=load_breast_cancer(return_X_y=True)
X=StandardScaler().fit_transform(X)
Xtr,Xte,ytr,yte=t(X,y,test_size=.3,random_state=42,stratify=y)
s=SVC(kernel='linear',probability=True).fit(Xtr,ytr)
k=KNeighborsClassifier(5).fit(Xtr,ytr)
print("SVM:",accuracy_score(yte,s.predict(Xte)))
print("KNN:",accuracy_score(yte,k.predict(Xte)))
m=['SVM','KNN']
tr=[s.score(Xtr,ytr),k.score(Xtr,ytr)]
te=[s.score(Xte,yte),k.score(Xte,yte)]
x=np.arange(2)
plt.bar(x-.2,tr,.4);plt.bar(x+.2,te,.4)
plt.xticks(x,m);plt.show()
f1,t1,_=roc_curve(yte,s.predict_proba(Xte)[:,1])
f2,t2,_=roc_curve(yte,k.predict_proba(Xte)[:,1])
plt.plot(f1,t1,label='SVM');plt.plot(f2,t2,label='KNN')
plt.legend();plt.show()
print("SVM\n",confusion_matrix(yte,s.predict(Xte)))
print("KNN\n",confusion_matrix(yte,k.predict(Xte)))