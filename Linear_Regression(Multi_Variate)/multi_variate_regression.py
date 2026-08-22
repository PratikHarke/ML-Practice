#%%
import pandas as pd 
import numpy as np  
import matplotlib.pyplot as plt
from sklearn import linear_model
# %%
df = pd.read_csv('homeprices_multi.csv')
df
# %%
## Equation for determining price of a house - 
# price = m1 * area + m2 * bedrooms + m3 * age + b  

import math
med_bedrooms = math.floor(df.bedrooms.median())
med_bedrooms

# %%
df.bedrooms = df.bedrooms.fillna(med_bedrooms)

# %%
reg = linear_model.LinearRegression()
reg.fit(df[['area','bedrooms','age']],df['price'])

# %%
reg.coef_
# %%
reg.intercept_
# %%
reg.predict([[3000,3,40]])
# %%
reg.predict([[3500,5,2]])
# %%
