import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)

n = 300
temperature = 15 + 10 * np.random.randn(n)
humidity = 60 + 15 * np.random.randn(n)
co2 = 400 + 30 * np.random.randn(n)

rainfall = (
    1.2 * humidity
    - 0.8 * temperature
    + 0.05 * (co2 - 400)
    + 10 * np.random.randn(n)
)

df = pd.DataFrame({
    "Temperature": temperature,
    "Humidity": humidity,
    "CO2": co2,
    "Rainfall": rainfall
})

df.head()

X = df[["Temperature", "Humidity", "CO2"]]
y = df["Rainfall"]

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=1)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

y_pred_ridge = ridge.predict(X_test)

print("Linear Regression:")
print("MSE:", mean_squared_error(y_test, y_pred_lr))
print("R2 Score:", r2_score(y_test, y_pred_lr))

print("\nRidge Regression:")
print("MSE:", mean_squared_error(y_test, y_pred_ridge))
print("R2 Score:", r2_score(y_test, y_pred_ridge))

plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred_ridge)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Rainfall")
plt.ylabel("Predicted Rainfall")
plt.title("Ridge Regression: Actual vs Predicted")
plt.show()

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(6,4))
plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='viridis')
plt.colorbar(label="Rainfall")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Visualization of Climate Data")
plt.show()