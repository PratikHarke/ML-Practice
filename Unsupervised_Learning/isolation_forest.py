#%%
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
# %%
df  = pd.read_csv('healthcare.csv')
# %%
features = [
    "age",
    "bmi",
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "glucose",
    "cholesterol",
    "oxygen_saturation",
    "hospital_visits"
]
# %%
X = df[features]
# %%

## Scaling The Data

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# %%
model = IsolationForest(
    n_estimators=200,
    contamination=0.17,
    random_state=42
)
# %%
model.fit(X_scaled)
# %%
df['prediction'] = model.predict(X_scaled)
# %%
df["anomaly"] = df["prediction"].apply(
    lambda x: 1 if x == -1 else 0
)
# %%

## Perform PCA

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]
# %%
# %%

normal = df[df["anomaly"] == 0]
anomalies = df[df["anomaly"] == 1]

plt.figure(figsize=(10, 7))

plt.scatter(
    normal["glucose"],
    normal["heart_rate"],
    s=40,
    alpha=0.7,
    label="Normal"
)

plt.scatter(
    anomalies["glucose"],
    anomalies["heart_rate"],
    s=120,
    marker="X",
    label="Anomaly"
)

plt.xlabel("Glucose")
plt.ylabel("Heart Rate")

plt.title("Isolation Forest - Healthcare Anomaly Detection")

plt.legend()
plt.grid(alpha=0.2)

plt.tight_layout()
plt.show()

# %%
