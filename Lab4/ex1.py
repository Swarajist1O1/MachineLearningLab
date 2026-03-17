import pandas as pd, matplotlib.pyplot as plt, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("https://raw.githubusercontent.com/Yashappin/Machine-Learning/master/TvMarketing.csv")
print("Data:\n",df.head(),"\n\nStats:\n",df.describe())

plt.scatter(df['TV'],df['Sales']); plt.title('TV vs Sales'); plt.show()

X,y = df[['TV']],df['Sales']
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=.2,random_state=42)
print("Train:",Xtr.shape,"Test:",Xte.shape)

m = LinearRegression().fit(Xtr,ytr)
print("b0:",m.intercept_,"\nb1:",m.coef_[0])

plt.scatter(Xtr,ytr); plt.plot(Xtr,m.predict(Xtr),'r'); plt.show()

yp = m.predict(Xte)
print("\nActual vs Pred:\n",pd.DataFrame({'A':yte,'P':yp}).head())

print("\nRMSE:",np.sqrt(mean_squared_error(yte,yp)),
      "\nR2:",r2_score(yte,yp))