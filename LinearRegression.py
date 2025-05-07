import numpy as np
import matplotlib as plt
import math

"""
Linear Regression with Ordinary Least Square algorithm, plus L1 (LASSO) and L2 (Ridge) regularization
The optimization is programmed in both way, analytical solution (if data is feasible) and gradient descent
Control the regularization mode by the 'reg' parameter in method 'fit'
"""

class Regression:
    def __init__(self):
        self.w_a = None
        self.w_gd = None
        self.reg = 0

    def plot_fit(self, X_train, y_train):
        # Ensure 1D input
        if X_train.shape[1] != 1:
            raise ValueError("plot_fit only supports 1D input features.")

        # Sort X for smoother lines
        X_plot = np.sort(X_train, axis=0)
        m = X_plot.shape[0]

        # Add bias term
        X_with_bias = np.hstack((np.ones((m, 1)), X_plot))

        # Predictions
        y_pred_a = None
        if self.reg != 1:
            y_pred_a = X_with_bias @ self.w_a if self.w_a is not None else None
        y_pred_gd = X_with_bias @ self.w_gd if self.w_gd is not None else None

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.scatter(X_train, y_train, color='blue', label='Training Data')
        if y_pred_a is not None:
            plt.plot(X_plot, y_pred_a, color='green', label='Analytical Solution')
        if y_pred_gd is not None:
            plt.plot(X_plot, y_pred_gd, color='red', linestyle='--', label='Gradient Descent')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title('Regression Fit: Analytical vs Gradient Descent')
        plt.legend()
        plt.grid(True)
        plt.show()


    def random_mini_batches(self, X, y, mini_batch_size):
        m = X.shape[0]
        mini_batches = []

        perm = list(np.random.permutation(m))
        shuffle_X = X[perm, :]
        shuffle_Y = y[perm, :].reshape((m, 1))

        num_mini_batches = math.floor(m / mini_batch_size)
        for k in range(num_mini_batches):
            mini_batch_X = shuffle_X[k * mini_batch_size : (k + 1) * mini_batch_size, : ]
            mini_batch_Y = shuffle_Y[k * mini_batch_size : (k + 1) * mini_batch_size, : ]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        if m % mini_batch_size != 0:
            mini_batch_X = shuffle_X[num_mini_batches * mini_batch_size : ,]
            mini_batch_Y = shuffle_Y[num_mini_batches * mini_batch_size : ,]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        return mini_batches

    def fit(self, X, y, learning_rate=0.01, reg=0, lambd=0.3, mini_batch_size=0, max_iter=1000):
        # Data preprocessing
        self.reg = reg
        X_train = X
        y_train = y

        m, n = X.shape
        
        X = np.hstack((np.ones((m, 1)), X))
        
        if not mini_batch_size:
            mini_batch_size = m

        mini_batches = self.random_mini_batches(X, y, mini_batch_size)

        # Analytical fit
        if reg != 1:
            self.w_a = np.dot(X.T, X)
            if reg == 2:
                self.w_a += lambd * np.identity(n+1)
            self.w_a = np.linalg.inv(self.w_a)
            self.w_a = np.dot(self.w_a, X.T)
            self.w_a = np.dot(self.w_a, y)

            Z = np.dot(X, self.w_a)
            cost = np.sum((Z - y)**2) / (2*m)
            if reg == 2:
                cost += lambd * np.linalg.norm(self.w_a)
            print(f'Cost of analytical solution: {cost}')

        # Gradient descent fit
        self.w_gd = np.zeros((n + 1, 1))

        costs = []
        converge = 0
        for i in range(max_iter):
            if converge:
                break
                    
            for mini_batch in mini_batches:
                X, y = mini_batch
                m = X.shape[0]
                # Forward
                Z = np.dot(X, self.w_gd)
                # Compute cost
                cost = np.sum((Z - y)**2) / (2*m)
                costs.append(cost)
                if reg == 2:
                    cost += lambd * np.linalg.norm(self.w_gd, keepdims=True)**2
                if reg == 1:
                    cost += lambd * np.linalg.norm(self.w_gd, ord=1)
                # Backward
                dw = np.dot((Z - y).T, X).T / m
                if reg == 1:
                    dw += lambd
                if reg == 2:
                    dw += lambd * self.w_gd
                # Update
                self.w_gd -= learning_rate * dw
                # Converge
                if i > 2 and np.abs(costs[-1] - costs[-2]) < 1e-6:
                    print(f'Converge at {i} iteration of cost: {cost}')
                    self.plot_fit(X_train, y_train)
                    converge = 1
                    break
                # Print
                if i % 100 == 0:
                    print(f'Cost after {i} iteration: {cost}')
                    self.plot_fit(X_train, y_train)