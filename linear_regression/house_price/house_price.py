import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.metrics import mean_squared_error

dt = pd.read_csv(os.path.join(os.path.dirname(__file__), "data", "housing.csv"), index_col=0)

FEATURE_COLS = ["lotsize", "bedrooms", "bathrms", "stories", "garagepl"]
TRAIN_SIZE = int(0.95*len(dt))
X = dt.loc[:TRAIN_SIZE-1, FEATURE_COLS]
Y = dt.loc[:TRAIN_SIZE-1, "price"]
X_test = dt.loc[TRAIN_SIZE:, FEATURE_COLS]
Y_test = dt.loc[TRAIN_SIZE:, "price"]

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