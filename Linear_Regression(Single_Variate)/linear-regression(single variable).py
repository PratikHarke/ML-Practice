#%%
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import linear_model

#%%
df =pd.read_csv("homeprices.csv")
df
# %%
% matplotlib inline
plt.xlabel('area')
plt.ylabel('price')
plt.scatter(df.area,df.price,color='black',marker='*')
plt.show()
# %%
reg = linear_model.LinearRegression()
reg.fit(df[['area']],df.price)
# %%
reg.predict([[2550]])
# %%
## Determines the slope of line - m
reg.coef_
# %%
## Returns the y-intercept of line - b
reg.intercept_
# %%
135.78767123*2550+180616.4383561651
# %%
d = pd.read_csv("areas.csv")
d 
# %%
reg.predict(d)
# %%
d['prices']=reg.predict(d)
d
# %%
plt.xlabel('area')
plt.ylabel('price')
plt.scatter(df.area,df.price,color='black',marker='*')
plt.plot(df.area,reg.predict(df[['area']]),color='blue')
plt.show()
# %%

c= pd.read_csv("per_capita.csv")
c 
# %%
plt.xlabel('year')
plt.ylabel('per capita income')
plt.scatter(c.year,c['per_capita_income'],color='green')
plt.show()
# %%
reg = linear_model.LinearRegression()
reg.fit(c[['year']],c[['per_capita_income']])

# %%
reg.predict([[2030]])

# %%
reg.coef_
# %%
reg.intercept_
# %%
plt.xlabel('year')
plt.ylabel('per capita income')
plt.scatter(c.year,c['per_capita_income'],color='green')
plt.plot(c.year,reg.predict(c[['year']]),color='blue')
plt.show()
# %%