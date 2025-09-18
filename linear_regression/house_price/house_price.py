import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.metrics import mean_squared_error

dt = pd.read_csv("data/housing.csv", index_col=0)
X = dt.iloc[0:520, [1, 2, 3, 4, 10]]
Y = dt.price[0:520]
X_test = dt.iloc[-20:, [1, 2, 3, 4, 10]]
Y_test = dt.price[-20:]

plt.scatter(dt.lotsize, dt.price)
plt.show()

lr = linear_model.LinearRegression()
lr.fit(X, Y)
print(lr.intercept_)
print(lr.coef_)

Y_pred = lr.predict(X_test)
for i in range(len(Y_test)):
    print(f"Pred = {Y_pred[i]:0.3f} -> Real = {Y_test.iloc[i]:0.3f} | dis = {Y_test.iloc[i] - Y_pred[i]:0.3f}")

err = mean_squared_error(Y_test, Y_pred)
rmse_err = np.sqrt(err)
print("RMSE =", round(rmse_err, 3))