import pandas as pd, matplotlib.pyplot as plt, numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *

df = pd.read_csv('Lab4/advertising.csv'); print(df.info())
df = df.drop(['Ad Topic Line','City','Country','Timestamp'],1).fillna(df.mean())

plt.imshow(df.corr(),cmap='coolwarm'); plt.colorbar(); plt.title('Correlation'); plt.show()

X,y = df.drop('Clicked on Ad',1),df['Clicked on Ad']
X = StandardScaler().fit_transform(X)
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=.2,random_state=42)

m = LogisticRegression().fit(Xtr,ytr)
cv = cross_val_score(m,X,y,cv=KFold(5,1,42))
print("CV:",cv,"\nMean:",cv.mean())

yp = m.predict(Xte)
print("\nReport:\n",classification_report(yte,yp))

cm = confusion_matrix(yte,yp)
plt.imshow(cm); plt.colorbar(); plt.title('CM')
for i in range(len(cm)):
    for j in range(len(cm)):
        plt.text(j,i,cm[i,j],ha='center',va='center')
plt.show()

p = m.predict_proba(Xte)[:,1]
fpr,tpr,_ = roc_curve(yte,p)
plt.plot(fpr,tpr,label=f'AUC={auc(fpr,tpr):.2f}')
plt.legend(); plt.show()

pd.DataFrame({'A':yte,'P':yp}).reset_index(drop=1).head(20)\
.plot(kind='bar',figsize=(10,5)); plt.title('Actual vs Pred'); plt.show()