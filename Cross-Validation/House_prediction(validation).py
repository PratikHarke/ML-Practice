#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline 
# %%
## California Housing Dataset
# %%
from sklearn.datasets import fetch_california_housing
california_df = fetch_california_housing()
# %%
california_df
# %%
X= pd.DataFrame(california_df.data, columns=california_df.feature_names)
y=california_df.target
# %%
## Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# %%
X.head()
# %%
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
# %%
regressor.fit(X_train,y_train)
# %%
y_pred = regressor.predict(X_test)

# %%
y_pred
# %%
from sklearn.metrics import r2_score
# %%
score  = r2_score(y_test, y_pred)
# %%
score
# %%
## Hyperparameter Tuning
parameter ={
    'criterion': ['squared_error','friedman_mse', 'absolute_error', 'poisson'],
    'splitter' : ['best', 'random'],
    'max_depth' : [1,2,3,4,5,6,7,8,9,10,11,12,13,14],
    'max_features' :['auto' , 'sqrt' , 'log2']
}
# %%
regressor = DecisionTreeRegressor()
# %%
from sklearn.model_selection import GridSearchCV
regressorcv = GridSearchCV(regressor, param_grid = parameter, scoring='r2', cv=5)
# %%
regressorcv.fit(X_train,y_train)
# %%
regressorcv.best_score_
# %%
y_pred = regressorcv.predict(X_test)
# %%
r2_score(y_pred, y_test)
# %%
