# %%
from statistics import kde

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %%
from sklearn.datasets import fetch_california_housing
# %%
df = fetch_california_housing()

dataset = pd.DataFrame(df.data)
# %%
dataset.columns = df.feature_names
# %%
dataset.head()
# %%
#Independent and dependent variables
X=dataset
y=df.target
# %%
y
# %%
#train test split
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)
# %%
X_train
# %%
# Standardizing the dataset
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# %%
# Scaling the training data
X_train = scaler.fit_transform(X_train)
# %%
# Scaling the test data
X_test = scaler.transform(X_test)
# %%
# Inverse transforming the scaled data
scaler.inverse_transform(X_train)
# %%
from sklearn.linear_model import LinearRegression
# %%
#Cross validation
from sklearn.model_selection import cross_val_score
# %%
regression=LinearRegression()
regression.fit(X_train,y_train)
# %%
mse = cross_val_score(regression,X_train,y_train,scoring='neg_mean_squared_error',cv=5)
# %%
np.mean(mse)
# %%
#Predicting the model
reg_pred = regression.predict(X_test)
# %%
reg_pred
# %%
import seaborn as sns 
sns.displot(reg_pred-y_test,kind='kde')
# %%
from sklearn.metrics import r2_score
r2_score(reg_pred,y_test)
# %%
# RIDGE REGRESSION
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

# %%
ridge_regressor = Ridge()
# %%
ridge_regressor
# %%
parameters={'alpha':[1,2,3,9,10,15,20,30,40,50,60,70,80,90]}
ridgecv = GridSearchCV(ridge_regressor,parameters,scoring = 'neg_mean_squared_error',cv=5)
ridgecv.fit(X_train,y_train) 
# %%
print(ridgecv.best_params_)

# %%
print(ridgecv.best_score_)
# %%
ridge_pred = ridgecv.predict(X_test)
# %%
import seaborn as sns 
sns.displot(ridge_pred-y_test,kind='kde')
# %%
score = r2_score(ridge_pred,y_test)
# %%
score
# %%
# LASSO REGRESSION
from sklearn.linear_model import Lasso


# %%
lasso=Lasso()
# %%
parameters={'alpha':[1,2,3,9,10,15,20,30,40,50,60,70,80,90]}
lassocv = GridSearchCV(lasso,parameters,scoring = 'neg_mean_squared_error',cv=5)
lassocv.fit(X_train,y_train) 
# %%
print(lassocv.best_params_)
print(lassocv.best_score_)
# %%
