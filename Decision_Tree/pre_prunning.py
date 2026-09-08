# %%
import numpy as np
import seaborn as sns   
import pandas as pd 
import matplotlib.pyplot as plt
# %%
from sklearn.datasets import load_iris
# %%
iris = load_iris()
# %%
iris.target
# %%
iris.data
# %%
import seaborn as sns
# %%
df = sns.load_dataset('iris')
# %%
df.head()
# %%
X=df.iloc[:,:-1]
y=iris.target
# %%    
X,y
# %%
## Train-Test Split
from sklearn.model_selection import train_test_split

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# %%
X_train
# %%
from sklearn.tree import DecisionTreeClassifier
# %%
parameter = {'criterion': ['gini', 'entropy', 'log_loss'], 'splitter' : ['best', 'random'], 'max_depth' : [1,2,3,4,5] , 'max_features' :['auto' , 'sqrt' , 'log2']}
# %%
from sklearn.model_selection import GridSearchCV
# %%
treeModel = DecisionTreeClassifier(max_depth = 3)
cv = GridSearchCV(treeModel, param_grid = parameter, scoring='accuracy', cv=5)
# %%
cv.fit(X_train,y_train)
# %%
cv.best_params_
# %%
cv.best_score_
# %%
y_pred = cv.predict(X_test)
# %%
from sklearn.metrics import accuracy_score, classification_report
# %%
score = accuracy_score(y_pred,y_test)
# %%
score
# %%
report = classification_report(y_pred,y_test)
# %%
report
# %%
