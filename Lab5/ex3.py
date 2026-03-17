import numpy as n, matplotlib.pyplot as p
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split as t
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler

X,y=load_breast_cancer(return_X_y=True)
X=StandardScaler().fit_transform(X)
a,b,c,d=t(X,y,test_size=.3,random_state=42,stratify=y)

nb=GaussianNB().fit(a,c)
dt=DecisionTreeClassifier().fit(a,c)

m=['NB','DT']
tr=[nb.score(a,c),dt.score(a,c)]
te=[nb.score(b,d),dt.score(b,d)]
x=n.arange(2)
p.bar(x-.2,tr,.4,label='Train');p.bar(x+.2,te,.4,label='Test')
p.xticks(x,m);p.legend();p.show()

f1,t1,_=roc_curve(d,nb.predict_proba(b)[:,1])
f2,t2,_=roc_curve(d,dt.predict_proba(b)[:,1])
p.plot(f1,t1,label='NB');p.plot(f2,t2,label='DT')
p.legend();p.show()

print("NB\n",confusion_matrix(d,nb.predict(b)))
print("DT\n",confusion_matrix(d,dt.predict(b)))