import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from sklearn.preprocessing import label_binarize

df = pd.read_csv("iris.data",header=None)
X,y = df.iloc[:,:-1],df.iloc[:,-1]
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=.3,random_state=42)

# Hyperparams
for n in [5,10,50,100]:
    for d in [2,3,5,None]:
        print(n,d,accuracy_score(yte,
            RandomForestClassifier(n,d,random_state=42).fit(Xtr,ytr).predict(Xte)))

# Final model
m = RandomForestClassifier(100,None,42).fit(Xtr,ytr)
yp = m.predict(Xte)

print("\nAcc:",accuracy_score(yte,yp))
print("\nCM:\n",confusion_matrix(yte,yp))
print("\nReport:\n",classification_report(yte,yp))

# CM plot
cm = confusion_matrix(yte,yp)
plt.imshow(cm); plt.colorbar(); plt.title("CM"); plt.show()

# PR curve
yt,ys = label_binarize(yte,np.unique(y)), m.predict_proba(Xte)
for i in range(yt.shape[1]):
    p,r,_ = precision_recall_curve(yt[:,i],ys[:,i])
    plt.plot(r,p)
plt.title("PR"); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.show()