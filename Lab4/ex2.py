import pandas as pd, matplotlib.pyplot as plt, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv('Lab4/data.csv'); print(df.info())

plt.imshow(df[['Volume','Weight','CO2']].corr(),cmap='coolwarm')
plt.colorbar(); plt.title('Correlation'); plt.show()

fig,ax = plt.subplots(1,3,figsize=(15,5))
for i,c in enumerate(['Volume','Weight','CO2']):
    ax[i].boxplot(df[c]); ax[i].set_title(c)
plt.show()

pd.plotting.scatter_matrix(df[['Volume','Weight','CO2']],figsize=(8,8))
plt.show()

X,y = df[['Volume','Weight']],df['CO2']
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=.2,random_state=42)

m = LinearRegression().fit(Xtr,ytr)
print("W:",m.coef_,"\nI:",m.intercept_)

yp = m.predict(Xte)
plt.plot(yte.values,label='True'); plt.plot(yp,label='Pred')
plt.legend(); plt.title('True vs Pred'); plt.show()

print("MAE:",mean_absolute_error(yte,yp),
      "\nMSE:",mean_squared_error(yte,yp),
      "\nRMSE:",np.sqrt(mean_squared_error(yte,yp)))