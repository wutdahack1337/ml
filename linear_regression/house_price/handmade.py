import numpy as np
import matplotlib.pyplot as plt

def LR1(X, Y, epochs, learning_rate, theta0, theta1):
    m = len(X)
    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch} ===")
        print(f"theta0 = {theta0:0.3f}, theta1 = {theta1:0.3f}")

        for i in range(0, m):
            h = theta0 + theta1*X[i]
            theta0 = theta0 - learning_rate*(h - Y[i])*1
            theta1 = theta1 - learning_rate*(h - Y[i])*X[i]
            print(f"\ttheta0 = {theta0:0.3f}, theta1 = {theta1:0.3f}")
    return [theta0, theta1]


def Learn_And_Predict(X, Y, X_test, learning_rate, theta0, theta1):
    theta_1_epoch  = LR1(X, Y, 1, learning_rate, theta0, theta1)
    theta_10_epochs = LR1(X, Y, 10, learning_rate, theta0, theta1)

    X_linear = np.array([0, 6])
    Y1_linear = theta_1_epoch[0] + theta_1_epoch[1]*X_linear
    Y10_linear = theta_10_epochs[0] + theta_10_epochs[1]*X_linear
    plt.plot(X_linear, Y1_linear, "r-")
    plt.plot(X_linear, Y10_linear, "g-")

    Y_predict = theta_10_epochs[0] + theta_10_epochs[1]*X_test
    print(f"\n=== Prediction ===")
    for i in range(len(X_test)):
        print(f"\tx = {X_test[i]} => y = {Y_predict[i]:0.3f}")

def Plot(X, Y):
    plt.axis([0, 6, 0, 10])
    plt.xlabel("X")
    plt.ylabel("House Price")
    plt.plot(X, Y, "bo")

    plt.show()


if __name__ == "__main__":
    X = np.array([1, 2, 4])
    Y = np.array([2, 3, 6])

    X_test = np.array([0, 3, 5])
    Learn_And_Predict(X, Y, X_test, 0.1, 0, 1)

    Plot(X, Y)