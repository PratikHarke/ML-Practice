#%%
import seaborn as sns
df = sns.load_dataset("titanic")

# %%
import sweetviz as sv
report = sv.analyze(df)
report.show_html()
# %%
