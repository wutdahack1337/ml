import numpy as np
import matplotlib.pyplot as plt

def LR1(X, Y, epochs, learning_rate, theta0, theta1):
    m = len(X)
    for epoch in range(epochs):
        # print(f"\n=== Epoch {epoch} ===")
        # print(f"theta0 = {theta0:0.9f}, theta1 = {theta1:0.9f}")

        for i in range(0, m):
            h = theta0 + theta1*X[i]
            # print("h:", h, )
            theta0 = theta0 - learning_rate*(h - Y[i])*1
            theta1 = theta1 - learning_rate*(h - Y[i])*X[i]
            # print(f"\ttheta0 = {theta0:0.9f}, theta1 = {theta1:0.9f}")
    return [theta0, theta1]

def LR2(X, Y, epochs, learning_rate, theta0, theta1):
    m = len(X)
    for epoch in range(epochs):
        # print(f"\n=== Epoch {epoch} ===")
        # print(f"theta0 = {theta0:0.9f}, theta1 = {theta1:0.9f}")

        sum_theta0 = 0
        sum_theta1 = 0
        for i in range(0, m):
            h = theta0 + theta1*X[i]
            sum_theta0 += (h - Y[i])*1
            sum_theta1 += (h - Y[i])*X[i]
    
        theta0 = theta0 - learning_rate*sum_theta0
        theta1 = theta1 - learning_rate*sum_theta1
    return [theta0, theta1]

X = np.array([147,  150, 153, 155, 158, 160, 163, 165, 168, 170, 173, 175, 178, 180])
Y = np.array([ 49,   50,  51,  52,  54,  56,  58,  59,  60,  72,  63,  64,  66,  67])
X_test = 183
Y_test = 68

# LR1
    # learning_rate = 0.00003, theta0 = 0, theta1 = 0.5
        # epochs = 2
theta = LR1(X, Y, 2, 0.00003, 0, 0.5)
Y_pred = theta[0] + theta[1]*X_test
print(f"Pred = {Y_pred} -> Real = {Y_test}")
    # epochs = 4
theta = LR1(X, Y, 4, 0.00003, 0, 0.5)
Y_pred = theta[0] + theta[1]*X_test
print(f"Pred = {Y_pred} -> Real = {Y_test}")

    # learning_rate = 0.00002, theta0 = 0, theta1 = 0.5
        # epochs = 2
theta = LR1(X, Y, 2, 0.00002, 0, 0.5)
Y_pred = theta[0] + theta[1]*X_test
print(f"Pred = {Y_pred} -> Real = {Y_test}")
        # epochs = 10
theta = LR1(X, Y, 10, 0.00002, 0, 0.5)
Y_pred = theta[0] + theta[1]*X_test
print(f"Pred = {Y_pred} -> Real = {Y_test}")


# LR2
    # learning_rate = 0.000003, theta0 = 0, theta1 = 0.5
        # epochs = 2
theta = LR2(X, Y, 2, 0.000003, 0, 0.5)
Y_pred = theta[0] + theta[1]*X_test
print(f"Pred = {Y_pred} -> Real = {Y_test}")
        # epochs = 4    
theta = LR2(X, Y, 4, 0.000003, 0, 0.5)
Y_pred = theta[0] + theta[1]*X_test
print(f"Pred = {Y_pred} -> Real = {Y_test}")
    # epochs = 100 
theta = LR2(X, Y, 100, 0.000003, 0, 0.5)
Y_pred = theta[0] + theta[1]*X_test
print(f"Pred = {Y_pred} -> Real = {Y_test}")

    # learning_rate = 0.000002, theta0 = 0, theta1 = 0.5
        # epochs = 2
theta = LR2(X, Y, 2, 0.000002, 0, 0.5)
Y_pred = theta[0] + theta[1]*X_test
print(f"Pred = {Y_pred} -> Real = {Y_test}")
        # epochs = 10
theta = LR2(X, Y, 10, 0.000002, 0, 0.5)
Y_pred = theta[0] + theta[1]*X_test
print(f"Pred = {Y_pred} -> Real = {Y_test}")
    # epochs = 100 
theta = LR2(X, Y, 100, 0.000002, 0, 0.5)
Y_pred = theta[0] + theta[1]*X_test
print(f"Pred = {Y_pred} -> Real = {Y_test}")

plt.plot(X, Y, "bo")
plt.show()