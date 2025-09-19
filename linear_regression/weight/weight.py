import numpy as np

from sklearn import linear_model

X = np.array([[147,  150, 153, 155, 158, 160, 163, 165, 168, 170, 173, 175, 178, 180]]).T
Y = np.array([[ 49,   50,  51,  52,  54,  56,  58,  59,  60,  72,  63,  64,  66,  67]]).T
X_test = np.array([[183]])
Y_test = 68

lr = linear_model.LinearRegression()
lr.fit(X, Y)
Y_pred = lr.predict(X_test)
print(f"Pred = {Y_pred[0][0]:0.3f} -> Real = {Y_test}")