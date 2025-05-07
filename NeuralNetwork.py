import numpy as np

"""
Vanilla Neural Network implementation, using ordinary gradient descent

To specify the architecture of the neural network, use parameter 'layer_dims' in method 'fit'
    layer_dims = [n_0, n_1, ..., n_L]
where:
    n_i - number of neurons in i_th layer
    L   - number of layers in the network

Note: X.shape, Y.shape = (number_of_features, number_of_examples)
"""

class NeuralNetwork:
    def __init__(self):
        self.parameters = None
        self.L = None
        self.cost = None

    def sigmoid(self, Z):
        # Sigmoid function: 1/(e^(-Z) + 1)
        # Arguments:
        #   Z: input vector
        # Return:
        #   1/(np.exp(-Z) + 1)
        return 1/(np.exp(-Z) + 1)
    
    def relu(self, Z):
        # ReLU function: max(0, Z)
        # Arguments:
        #   Z: input vector
        # Return:
        #   return max(0, Z)
        return np.maximum(0, Z)

    def init_parameter(self, layer_dims):
        # Initialize the parameter
        # Arguments:
        #   layer_dims: an array contains number of neurons in each layer (e.g: [2, 3, 2])
        # Return:
        #   parameters: a dictionary contains all the weights and biases
        
        L = len(layer_dims)
        parameters = {}

        for i in range(1, L):
            parameters['W' + str(i)] = np.random.randn(layer_dims[i], layer_dims[i-1]) * 0.01
            parameters['b' + str(i)] = np.zeros((layer_dims[i], 1))

        return parameters
    
    def linear_activation_forward(self, A_prev, W, b, activation):
        # Compute the output of a layer 
        # Arguments:
        #   A_prev: Input from previous layer
        #   W: Weight at current layer
        #   b: Bias at current layer
        #   activation: type of activation function
        # Return:
        #   A: output of the current layer
        #   cache: (Z, A) store for efficient use in the backward pass
        Z = np.dot(W, A_prev) + b

        if activation == 'sigmoid':
            A = self.sigmoid(Z)
        elif activation == 'relu':
            A = self.relu(Z)
        cache = (A_prev, Z, A, W, b)

        return A, cache
    
    def model_forward(self, X):
    # Perform the forward pass of the model
    # Arguments:
    #   X: Input data
    #   parameters: The dictionary contains all the weights and biases to be learned
    # Return:
    #   AL: the output of the model
    #   caches: the list of every layers
        caches = []
        L = self.L - 1
        parameters = self.parameters.copy()
        A = X
        
        # Forward from input layer to L-1 layer (the layer before the output layer)
        for i in range(1, L):
            A_prev = A
            A, cache = self.linear_activation_forward(  A_prev=A_prev, \
                                                        W=parameters['W' + str(i)], \
                                                        b=parameters['b' + str(i)], \
                                                        activation='relu')
            caches.append(cache)
        
        # Forward from L-1 layer to output layer
        A_prev = A
        AL, cache = self.linear_activation_forward( A_prev=A_prev, \
                                                    W=parameters['W' + str(L)], \
                                                    b=parameters['b' + str(L)], \
                                                    activation='sigmoid')
        caches.append(cache)

        return AL, caches
    
    def compute_cost(self, AL, Y):
    # Compute the cost using cross-entropy
    # Argument:
    #   AL: Model's prediction
    #   Y: Ground truth 
    # Return:
    #   cost
        m = Y.shape[1]

        a = Y * np.log(AL)
        cost = -np.sum(a, keepdims=True) / m
        cost = np.squeeze(cost)

        return cost
    
    def relu_backward(self, dA, Z):
    # Compute the derivative of the relu function
    # Arguments:
    #   dA: dJ/dA
    #   Z: input vector
    # Return:
    #   dJ/dZ
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
 
        return dZ

    def sigmoid_backward(self, dA, A):
    # Compute the derivative of the sigmoid function
    # Arguments:
    #   dA: dJ/dA
    #   Z: input vector
    # Return:
    #   dJ/dZ
        return dA * A * (1 - A)


    def linear_activation_backward(self, dA, cache, activation):
    # Compute the backward pass of the current layer
    # Arguments:
    #   dA: Gradient of cost w.r.t activation of the current layer (which is computed by the layer after it)
    #   cache: Z, A of the current layer
    #   activation: type of activation of the current layer
    # Return:
    #   dA_prev: Gradient of cost w.r.t activation of the previous layer
    #   dW: Gradient of cost w.r.t weight of the current layer
    #   db: Gradient of cost w.r.t bias of the current layer
        A_prev, Z, A, W, b = cache
        
        # Compute dZ
        if activation == 'relu':
            dZ = self.relu_backward(dA, Z)
        elif activation == 'sigmoid':
            dZ = self.sigmoid_backward(dA, A)
        
        # Compute dW, db
        dW = np.dot(dZ, A_prev.T)
        db = np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        return dW, db, dA_prev
    
    def model_backward(self, AL, Y, caches):
    # Perform the backward pass of the model
    # Arguments:
    #   AL: Model's predictioc
    #   Y: Ground truth
    #   caches: all the parameters in the previous forward pass
    # Return:
    #   grads: The dictionary that contains all the gradient for every weights and biases
        m = Y.shape[1]
        L = self.L - 1
        grads = {}

        dAL = - Y * (1 / AL) / m 

        # Compute the gradient of the output layer
        cache = caches[L - 1]
        dW, db, dA_prev = self.linear_activation_backward(dA=dAL, cache=cache, activation='sigmoid')
        grads['dW' + str(L)] = dW
        grads['db' + str(L)] = db
        grads['dA' + str(L - 1)] = dA_prev

        # Compute the gradient of the layer L-2 to 0
        for i in reversed(range(L-1)):
            cache = caches[i]
            dW, db, dA_prev = self.linear_activation_backward(dA=grads['dA' + str(i + 1)], cache=cache, activation='relu')
            grads['dW' + str(i + 1)] = dW
            grads['db' + str(i + 1)] = db
            grads['dA' + str(i)] = dA_prev
        
        return grads
    
    def update_parameters(self, grads, learning_rate):
    # Update the weights and biases
    # Arguments:
    #   grads: gradient of every weights and biases
    #   learning_rate
    # Return:
    #   null
        parameters = self.parameters.copy()
        L = self.L - 1
        
        for i in range(1, L + 1):
            parameters['W' + str(i)] -= learning_rate * grads['dW' + str(i)]
            parameters['b' + str(i)] -= learning_rate * grads['db' + str(i)]
        
        return parameters
    
    def fit(self, X, Y, layer_dims, learning_rate=0.01, max_iter=200):
    # Training function of the model
    # Arguments:
    #   X: input matrix (1 example per column)
    #   Y: labels vector
    #   layer_dims: List of number of neurons in each layer (netowrk architecture)
    #   learning_rate
    #   max_iter
        self.cost = []
        self.L = len(layer_dims)
        self.parameters = self.init_parameter(layer_dims)

        for i in range(max_iter):
            AL, caches = self.model_forward(X)
            cost = self.compute_cost(AL, Y)
            self.cost.append(cost)
            grads = self.model_backward(AL, Y, caches)
            self.parameters = self.update_parameters(grads, learning_rate)
            print(f'Cost after iteration {i}: {cost}')