import numpy as np
from src.activations import sigmoid, sigmoid_derivative, relu, relu_derivative, tanh, tanh_derivative
from src.losses import MSE, BinaryCrossEntropy
from tqdm import tqdm

class NeuralNetwork:
    """Neural Network implementation"""
    
    def __init__(self, layer_sizes, activation='sigmoid', loss='mse'):
        """
        Initialize neural network
        
        Args:
            layer_sizes (list): List of integers for number of neurons in each layer
            activation (str): Activation function to use ('sigmoid', 'relu', or 'tanh')
            loss (str): Loss function to use ('mse' or 'bce')
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        for i in range(self.num_layers - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01)
            self.biases.append(np.random.randn(1, layer_sizes[i+1]) * 0.01)
            
        # Set activation function
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif activation == 'relu':
            self.activation = relu 
            self.activation_derivative = relu_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_derivative = tanh_derivative
        else:
            raise ValueError("Activation must be 'sigmoid', 'relu', or 'tanh'")
            
        # Set loss function
        if loss == 'mse':
            self.loss = MSE()
        elif loss == 'bce':
            self.loss = BinaryCrossEntropy()
        else:
            raise ValueError("Loss must be 'mse' or 'bce'")
            
    def forward(self, X):
        """
        Forward pass through network
        
        Args:
            X (np.ndarray): Input data
            
        Returns:
            list: List of activations for each layer
        """
        activations = [X]
        for i in range(self.num_layers - 1):
            net = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            activations.append(self.activation(net))
        return activations
        
    def backward(self, X, y, learning_rate=0.01):
        """
        Backward pass through network
        
        Args:
            X (np.ndarray): Input data
            y (np.ndarray): Target values
            learning_rate (float): Learning rate for gradient descent
            
        Returns:
            float: Loss value
        """
        m = X.shape[0]
        activations = self.forward(X)
        
        # Calculate initial error
        loss_value = self.loss.calculate(activations[-1], y)
        delta = self.loss.derivative(activations[-1], y)
        
        # Backpropagate error
        for i in range(self.num_layers - 2, -1, -1):
            delta = delta * self.activation_derivative(activations[i+1])
            dW = np.dot(activations[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            
            if i > 0:
                delta = np.dot(delta, self.weights[i].T)
            
            # Update weights and biases
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * db
            
        return loss_value
        
    def train(self, X, y, epochs=100, learning_rate=0.1, batch_size=None):
        """
        Train the neural network
        
        Args:
            X (np.ndarray): Training data
            y (np.ndarray): Target values
            epochs (int): Number of training epochs
            learning_rate (float): Learning rate for gradient descent
            batch_size (int): Batch size for mini-batch gradient descent
            
        Returns:
            list: Training loss history
        """
        history = []
        for epoch in tqdm(range(epochs), desc='Training Progress'):
            loss = self.backward(X, y, learning_rate)
            history.append(loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
                
        return history
        
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X (np.ndarray): Input data
            
        Returns:
            np.ndarray: Predictions
        """
        return self.forward(X)[-1]
