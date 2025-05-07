import numpy as np
import matplotlib.pyplot as plt

"""
Support vector machine using gradient descent to optimize the hinge loss function
"""

class SVM:
    def __init__(self):
        self.w = None
        self.b = None

    def predict(self, X):
        if self.w is None or self.b is None:
            print("You've not trained the model yet")
            return -1
        return np.sign(np.dot(X, self.w) + self.b)

    def cost(self, w, b, X, y, lambd):
        m = X.shape[0]
        margins = 1 - y * (np.dot(X, w) + b)
        hinge_loss = np.maximum(0, margins)
        cost = np.mean(hinge_loss) + lambd * np.sum(w ** 2)
        return cost

    def fit(self, X, y, lambd=0.01, lr=0.01, epochs=1000):
        m, n = X.shape
        y = 2*y - 1  # Ensure column vector
        w = np.zeros((n, 1))
        b = 0

        for i in range(epochs):
            margin = y * (np.dot(X, w) + b)
            condition = margin < 1

            X_cond = X[condition.flatten()]
            y_cond = y[condition].reshape(-1, 1)

            dW = 2 * lambd * w - (np.dot(X_cond.T, y_cond) / m)
            db = -np.sum(y_cond) / m


            # Parameter update
            w -= lr * dW
            b -= lr * db

            # Optional: monitor cost
            if i % 100 == 0:
                c = self.cost(w, b, X, y, lambd)
                print(f"Epoch {i}, Cost: {c:.4f}")

        self.w = w
        self.b = b
        return w, b