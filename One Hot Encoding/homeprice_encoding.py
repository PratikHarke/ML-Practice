# %%
import pandas as pd
import numpy as np
df = pd.read_csv('homeprices.csv') 
# %%
df
# %%
dummies = pd.get_dummies(df.town,dtype=int)
# %%
merged = pd.concat([df,dummies],axis='columns')
# %%
merged
# %%
final = merged.drop(['town','west windsor'],axis='columns')
# %%
final
# %%
from sklearn.linear_model import LinearRegression
model = LinearRegression()
# %%
X = final.drop('price',axis='columns')
y = final.price
# %%
model.fit(X,y)
# %%
model.predict([[2800,0,1]])
# %%
model.score(X,y)
# %%
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
# %%
dfle = df.copy()
# %%
dfle.town = le.fit_transform(dfle.town)
dfle
# %%
X=dfle[['town','area']].values
X
# %%
y=dfle.price
y
# %%
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(
    [('town', OneHotEncoder(), [0])],
    remainder='passthrough'
)

X = ct.fit_transform(X)
X
# %%
X=X[:,1:]
X
# %%
model.fit(X,y)
# %%
model.predict([[1,0,2800]])
# %%
model.predict([[0,1,3400]])
# %%
