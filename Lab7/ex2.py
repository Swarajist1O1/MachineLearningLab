import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import *
from sklearn.preprocessing import label_binarize

d = load_iris()
X,y,cn = d.data,d.target,d.target_names
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=.5,random_state=42)

H=[]
for n in [10,50,100]:
    H.append([accuracy_score(yte,
        AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3),
            n_estimators=n,
            learning_rate=lr,
            random_state=42
        ).fit(Xtr,ytr).predict(Xte)) for lr in [0.1,0.5,1.0]])

plt.imshow(H,cmap='magma'); plt.colorbar()
plt.xticks(range(3),[0.1,0.5,1.0]); plt.yticks(range(3),[10,50,100])
plt.xlabel("LR"); plt.ylabel("n"); plt.title("Heatmap"); plt.show()

m = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=100,
    learning_rate=1.0,
    random_state=42
).fit(Xtr,ytr)

yp = m.predict(Xte)


print("Acc:",accuracy_score(yte,yp))

cm = confusion_matrix(yte,yp)
plt.imshow(cm); plt.colorbar()
plt.xticks(range(3),cn); plt.yticks(range(3),cn)
plt.title("CM"); plt.xlabel("Pred"); plt.ylabel("Actual"); plt.show()

print("\nReport:\n",classification_report(yte,yp))

yt = label_binarize(yte,classes=[0,1,2])
ys = m.predict_proba(Xte)

for i in range(3):
    p,r,_ = precision_recall_curve(yt[:,i],ys[:,i])
    plt.plot(r,p,label=cn[i])

plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("PR"); plt.legend(); plt.show()

fig,ax = plt.subplots(1,2,figsize=(16,6),dpi=150)

plot_tree(m.estimators_[0],
          feature_names=d.feature_names,
          class_names=cn,
          filled=True,rounded=True,precision=2,ax=ax[0])

plot_tree(m.estimators_[-1],
          feature_names=d.feature_names,
          class_names=cn,
          filled=True,rounded=True,precision=2,ax=ax[1])

ax[0].set_title("Tree 0")
ax[1].set_title("Tree Last")

plt.tight_layout(); plt.show()