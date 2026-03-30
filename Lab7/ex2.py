import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import *
from sklearn.preprocessing import label_binarize

d = load_iris()
Xtr,Xte,ytr,yte = train_test_split(d.data,d.target,test_size=.5,random_state=42)

# Heatmap
lr, n = [0.1,0.5,1.0], [10,50,100]
H = [[accuracy_score(yte, AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=3),
        n_estimators=i, learning_rate=j, random_state=42
    ).fit(Xtr,ytr).predict(Xte)) for j in lr] for i in n]

plt.imshow(H); plt.colorbar()
plt.xticks(range(3),lr); plt.yticks(range(3),n)
plt.xlabel("LR"); plt.ylabel("n"); plt.title("Heatmap"); plt.show()

# Final model
m = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),100,1.0,random_state=42).fit(Xtr,ytr)
yp = m.predict(Xte)

print("Acc:",accuracy_score(yte,yp))
print("\nReport:\n",classification_report(yte,yp))

# Confusion matrix
cm = confusion_matrix(yte,yp)
plt.imshow(cm); plt.colorbar()
plt.xticks(range(3),d.target_names); plt.yticks(range(3),d.target_names)
plt.title("CM"); plt.show()

# PR curve
yt, ys = label_binarize(yte,[0,1,2]), m.predict_proba(Xte)
for i in range(3):
    p,r,_ = precision_recall_curve(yt[:,i],ys[:,i])
    plt.plot(r,p,label=d.target_names[i])
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend(); plt.title("PR"); plt.show()

# Trees
fig,ax = plt.subplots(1,2,figsize=(14,5))
for i,t in enumerate([0,-1]):
    plot_tree(m.estimators_[t], feature_names=d.feature_names,
              class_names=d.target_names, filled=True, ax=ax[i])
    ax[i].set_title(f"Tree {t if t==0 else 'Last'}")
plt.show()