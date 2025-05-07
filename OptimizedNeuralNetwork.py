import numpy as np
import math

"""
Neural Network implementation with some optimize techniques:
    + Adam Gradient Descent
    + Mini-Batch 
    + Xavier initializtion

To specify the architecture of the neural network, use parameter 'layer_dims' in method 'fit'
    layer_dims = [n_0, n_1, ..., n_L]
where:
    n_i - number of neurons in i_th layer
    L   - number of layers in the network

Note: X.shape, Y.shape = (number_of_features, number_of_examples)
"""

class NeuralNetwork:
    def __init__(self):
        pass

    def relu(self, Z):
        return np.maximum(0, Z)
    
    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))
    
    def tanh(self, Z):
        return np.tanh(Z)

    def relu_backward(self, dA, Z):
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ

    def sigmoid_backward(self, dA, A):
        return dA * A * (1 - A)

    def tanh_backward(self, dA, A):
        return dA * (1 - A**2)

    def forward_linear(self, A_prev, W, b, activation):
        Z = np.dot(W, A_prev) + b
        if activation == 'relu':
            A = self.relu(Z)
        elif activation == 'sigmoid':
            A = self.sigmoid(Z)
        elif activation == 'tanh':
            A = self.tanh(Z)
        cache = (W, b, Z, A, A_prev)
        return A, cache

    def forward(self, X, parameters, activations):
        A_prev = X
        caches = []
        L = len(parameters) // 2
        for i in range(1, L + 1):
            A, cache = self.forward_linear(A_prev, parameters['W' + str(i)], parameters['b' + str(i)], activations[i - 1])
            caches.append(cache)
            A_prev = A
        return A_prev, caches

    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        cost = -1/m * np.sum(Y * np.log(AL + 1e-8))
        return np.squeeze(cost)

    def backward_linear(self, dA, cache, activation):
        W, b, Z, A, A_prev = cache
        m = A_prev.shape[1]

        if activation == 'relu':
            dZ = self.relu_backward(dA, Z)
        elif activation == 'sigmoid':
            dZ = self.sigmoid_backward(dA, A)
        elif activation == 'tanh':
            dZ = self.tanh_backward(dA, A)

        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(W.T, dZ)
        return dA_prev, dW, db

    def backward(self, AL, Y, activations, caches):
        L = len(caches)
        grads = {}
        dAL = - (np.divide(Y, AL + 1e-8) - np.divide(1 - Y, 1 - AL + 1e-8))

        for i in reversed(range(L)):
            cache = caches[i]
            dA_prev, dW, db = self.backward_linear(dAL, cache, activations[i])
            grads['dW' + str(i + 1)] = dW
            grads['db' + str(i + 1)] = db
            dAL = dA_prev
        return grads

    def init_parameters(self, layer_dims):
        parameters = {}
        for i in range(1, len(layer_dims)):
            parameters['W' + str(i)] = np.random.randn(layer_dims[i], layer_dims[i - 1]) * np.sqrt(2 / layer_dims[i - 1])
            parameters['b' + str(i)] = np.zeros((layer_dims[i], 1))
        return parameters

    def init_adam(self, parameters):
        v, s = {}, {}
        for i in range(1, len(parameters) // 2 + 1):
            v['dW' + str(i)] = np.zeros_like(parameters['W' + str(i)])
            s['dW' + str(i)] = np.zeros_like(parameters['W' + str(i)])
            v['db' + str(i)] = np.zeros_like(parameters['b' + str(i)])
            s['db' + str(i)] = np.zeros_like(parameters['b' + str(i)])
        return v, s

    def update_parameters(self, parameters, grads, v, s, t, learning_rate, b1=0.9, b2=0.999, epsilon=1e-8):
        L = len(parameters) // 2
        for i in range(1, L + 1):
            # Moving averages
            v['dW' + str(i)] = b1 * v['dW' + str(i)] + (1 - b1) * grads['dW' + str(i)]
            v['db' + str(i)] = b1 * v['db' + str(i)] + (1 - b1) * grads['db' + str(i)]

            s['dW' + str(i)] = b2 * s['dW' + str(i)] + (1 - b2) * np.square(grads['dW' + str(i)])
            s['db' + str(i)] = b2 * s['db' + str(i)] + (1 - b2) * np.square(grads['db' + str(i)])

            # Bias correction
            v_corrected_dw = v['dW' + str(i)] / (1 - b1**t)
            v_corrected_db = v['db' + str(i)] / (1 - b1**t)

            s_corrected_dw = s['dW' + str(i)] / (1 - b2**t)
            s_corrected_db = s['db' + str(i)] / (1 - b2**t)

            # Parameter update
            parameters['W' + str(i)] -= learning_rate * v_corrected_dw / (np.sqrt(s_corrected_dw) + epsilon)
            parameters['b' + str(i)] -= learning_rate * v_corrected_db / (np.sqrt(s_corrected_db) + epsilon)

    def mini_batch(self, X, y, mini_batch_size):
        n, m = X.shape
        permutation = list(np.random.permutation(m))
        X_shuffled = X[:, permutation]
        y_shuffled = y[:, permutation]
        num_batches = math.floor(m / mini_batch_size)
        batches = []

        for i in range(num_batches):
            mini_batch_X = X_shuffled[:, i * mini_batch_size:(i + 1) * mini_batch_size]
            mini_batch_Y = y_shuffled[:, i * mini_batch_size:(i + 1) * mini_batch_size]
            batches.append((mini_batch_X, mini_batch_Y))

        if m % mini_batch_size != 0:
            mini_batch_X = X_shuffled[:, num_batches * mini_batch_size:]
            mini_batch_Y = y_shuffled[:, num_batches * mini_batch_size:]
            batches.append((mini_batch_X, mini_batch_Y))

        return batches

    def fit(self, X, y, layer_dims, activations, mini_batch_size, max_iter=1000, b1=0.9, b2=0.999, learning_rate=0.01):
        parameters = self.init_parameters(layer_dims)
        v, s = self.init_adam(parameters)
        batches = self.mini_batch(X, y, mini_batch_size)

        for i in range(1, max_iter + 1):
            cost_total = 0
            for batch_X, batch_y in batches:
                AL, caches = self.forward(batch_X, parameters, activations)
                cost = self.compute_cost(AL, batch_y)
                grads = self.backward(AL, batch_y, activations, caches)
                self.update_parameters(parameters, grads, v, s, i, learning_rate, b1, b2)
                cost_total += cost
            print(f"Cost after iteration {i}: {cost_total:.6f}")

        self.parameters = parameters
        self.layer_dims = layer_dims
        self.activations = activations

    def predict(self, X, y):
        AL, _ = self.forward(X, self.parameters, self.activations)
        cost = self.compute_cost(AL, y)
        predictions = (AL > 0.5).astype(int)
        return cost, predictions