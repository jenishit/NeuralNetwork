import numpy as np
import pandas as pd
from typing import List

class MLP:
    def __init__(self, layers: List[int], learning_rate: float = 0.01, activation: str = 'sigmoid', regularization: str = None, lambda_reg: float = 0.01):
        self.layers = layers
        self.learning_rate = learning_rate
        self.activation = activation
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.weights = []
        self.biases = []
        self.cost_history = []

        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i], layers[i+1]) / np.sqrt(layers[i])
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

    def sigmoid_d(self, x):
        return x * (1 - x)
    
    def tanh(self, x):
        return np.tanh(x)
    
    def tanh_d(self, x):
        return 1 - x**2
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_d(self, x):
        return (x > 0).astype(float)
    
    def activate(self, x):
        if self.activation == 'sigmoid':
            return self.sigmoid(x)
        elif self.activation == 'tanh':
            return self.tanh(x)
        elif self.activation == 'relu':
            return self.relu(x)
        
    def activate_d(self, x):
        if self.activation == 'sigmoid':
            return self.sigmoid_d(x)
        elif self.activation == 'tanh':
            return self.tanh_d(x)
        elif self.activation == 'relu':
            return self.relu_d(x)
        
    def fwd_propagate(self, X):
        self.layer_inputs = [X]
        self.layer_ouputs = [X]

        current_input = X
        for i in range(len(self.weights)):
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            self.layer_inputs.append(z)

            if i < len(self.weights) - 1:
                a = self.activate(z)
            else:
                a = self.sigmoid(z)
            
            self.layer_ouputs.append(a)
            current_input = a
        return current_input
    
    def bwd_propagate(self, X, y):
        m = X.shape[0]

        dW = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]

        output_error = self.layer_ouputs[-1] - y
        delta = output_error * self.sigmoid_d(self.layer_ouputs[-1])

        for i in range(len(self.weights) - 1, -1, -1):
            dW[i] = np.dot(self.layer_ouputs[i].T, delta) / m
            db[i] = np.mean(delta, axis=0, keepdims=True)

            #Regularization
            if self.regularization == 'l2':
                dW[i] += self.lambda_reg * self.weights[i]
            elif self.regularization == 'l1':
                dW[i] += self.lambda_reg * np.sign(self.weights[i])
            
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.activate_d(self.layer_ouputs[i])

        return dW, db
    
    def fit(self, X, y, epochs: int = 1000, verbose: bool = True):
        for epoch in range(epochs):
            output = self.fwd_propagate(X)

            cost  = self.compute_cost(y, output)
            self.cost_history.append(cost)

            dW, db = self.bwd_propagate(X, y)
            
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * dW[i]
                self.biases[i] -= self.learning_rate * db[i]

            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Cost: {cost:.6f}")
            
        return self
    
    def compute_cost(self, y_true, y_pred):
        m = y_true.shape[0]
        cost = -np.mean(y_true * np.log(y_pred + 1e-15) + (1 - y_true) * np.log(1 - y_pred + 1e-15))

        if self.regularization == 'l2':
            l2_cost = sum(np.sum(w**2) for w in self.weights)
            cost += self.lambda_reg * l2_cost / (2 * m)
        elif self.regularization == 'l1':
            l1_cost = sum(np.sum(np.abs(w)) for w in self.weights)
            cost += self.lambda_reg * l1_cost / m
        return cost
    
    def predict(self, X):
        output = self.fwd_propagate(X)
        return (output > 0.5).astype(int)
    
    def probability(self, X):
        return self.fwd_propagate(X)
    
    def plot_training_history(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history)
        plt.title('Training Cost History')
        plt.xlabel('Epoch')
        plt.ylabel('Cost')
        plt.show()

if __name__ == "__main__":
    MLP(layers=[2, 4, 1], learning_rate=1.0, activation='sigmoid')