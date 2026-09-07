# %%
import seaborn as sns
import pandas as pd 
import numpy as np
#  %%
df = sns.load_dataset('iris')
df.head()
# %%
df['species'].unique()
# %%
df.isnull().sum()
# %%
df['species']!='setosa'
# %%
df = df[df['species']!='setosa']
# %%
df
# %%
df['species'] = df['species'].map({'versicolor':0,'virginica':1})
# %%
df
# %%
##Split dataset into dependent and independent features
X = df.iloc[:,:-1]
Y = df.iloc[:,-1]
# %%
from sklearn.model_selection import train_test_split
# %%
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=42)
# %%
from sklearn.linear_model import LogisticRegression
# %%
classifier = LogisticRegression()
# %%
from sklearn.model_selection import GridSearchCV
parameter = {'penalty':['l1','l2','elasticnet','none'],'C':[1,2,3,5,9,34,45,56,79],'max_iter':[100,200,300]}
# %%
classifier_regressor = GridSearchCV(classifier,param_grid = parameter, scoring='accuracy',cv=5)
# %%
classifier_regressor.fit(X_train,Y_train)
# %%
print(classifier_regressor.best_params_)
# %%
print(classifier_regressor.best_score_)
# %%
y_pred = classifier_regressor.predict(X_test)
# %%
## Accuracy Score Calaculation
from sklearn.metrics import accuracy_score, classification_report

# %%
score = accuracy_score(Y_test,y_pred)
print(score)
# %%
print(classification_report(Y_test,y_pred))
# %%
## EDA
sns.pairplot(df, hue='species')
# %%
df.corr()
# %%
