import numpy as np

class Layer:
    """Base class for neural network layers"""
    def forward(self, inputs):
        """Forward pass"""
        raise NotImplementedError
        
    def backward(self, grad):
        """Backward pass"""
        raise NotImplementedError

class Dense(Layer):
    """Fully connected layer"""
    def __init__(self, input_size, output_size):
        """
        Initialize Dense layer
        
        Args:
            input_size (int): Size of input
            output_size (int): Size of output
        """
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))
        self.inputs = None
        
    def forward(self, inputs):
        """
        Forward pass
        
        Args:
            inputs (np.ndarray): Input data
            
        Returns:
            np.ndarray: Output after linear transformation
        """
        self.inputs = inputs
        return np.dot(inputs, self.weights) + self.biases
        
    def backward(self, grad):
        """
        Backward pass
        
        Args:
            grad (np.ndarray): Gradient from next layer
            
        Returns:
            np.ndarray: Gradient with respect to input
        """
        # Gradients with respect to weights, biases, and inputs
        self.weight_gradients = np.dot(self.inputs.T, grad)
        self.bias_gradients = np.sum(grad, axis=0, keepdims=True)
        return np.dot(grad, self.weights.T)

class Activation(Layer):
    """Activation layer"""
    def __init__(self, activation_fn, activation_derivative):
        """
        Initialize Activation layer
        
        Args:
            activation_fn (callable): Activation function
            activation_derivative (callable): Derivative of activation function
        """
        self.activation_fn = activation_fn
        self.activation_derivative = activation_derivative
        self.inputs = None
        
    def forward(self, inputs):
        """
        Forward pass
        
        Args:
            inputs (np.ndarray): Input data
            
        Returns:
            np.ndarray: Activated output
        """
        self.inputs = inputs
        return self.activation_fn(inputs)
        
    def backward(self, grad):
        """
        Backward pass
        
        Args:
            grad (np.ndarray): Gradient from next layer
            
        Returns:
            np.ndarray: Gradient with respect to input
        """
        return grad * self.activation_derivative(self.inputs)
