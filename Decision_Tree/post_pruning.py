# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# %%
from sklearn.datasets import load_iris

# %%
iris =load_iris()

# %%
iris
# %%
iris.data
# %%
iris.target
# %%
df=sns.load_dataset('iris')
# %%
df.head()
# %%
X=df.iloc[:,:-1]
y=iris.target
# %%
X,y
# %%
## Split the dataset
from sklearn.model_selection import train_test_split
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
X_train
# %%
from sklearn.tree import DecisionTreeClassifier
# %%
## Post-Pruning Demonstration
treeModel = DecisionTreeClassifier(max_depth=2)
# %%
treeModel.fit(X_train,y_train)
# %%
from sklearn import tree
# %%
plt.figure(figsize=(14,9))
# %%
tree.plot_tree(treeModel,filled=True)
plt.show()
# %%
y_pred = treeModel.predict(X_test)
# %%
y_pred

# %%
from sklearn.metrics import accuracy_score, classification_report
# %%
score = accuracy_score(y_test,y_pred)
# %%
print(score)
# %%
repo = classification_report(y_test,y_pred)
# %%
print(repo)
# %%
