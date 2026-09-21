# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %%
from sklearn.datasets import fetch_california_housing
from autoviz.AutoViz_Class import AutoViz_Class
# %%
data = fetch_california_housing()
# %%
df = pd.DataFrame(
    data.data,
    columns=data.feature_names
)
# %%
df["MedHouseVal"] = data.target
# %%
filename = r"D:\ML-practice\EDA\california_housing.csv"
df.to_csv(filename, index=False)
# %%
AV = AutoViz_Class()
dft = AV.AutoViz(
    filename=filename,
    sep=",",
    depVar="MedHouseVal",
    verbose=2
)
# %%
# %%
