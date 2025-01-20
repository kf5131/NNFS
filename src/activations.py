import numpy as np


def sigmoid(x):
    """
    Sigmoid activation function.
    
    Args:
        x (np.ndarray): Input array
        
    Returns:
        np.ndarray: Sigmoid of input
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x): 
    """
    Derivative of sigmoid activation function.
    
    Args:
        x (np.ndarray): Input array
        
    Returns:
        np.ndarray: Derivative of sigmoid at input
    """
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x):
    """
    ReLU (Rectified Linear Unit) activation function.
    
    Args:
        x (np.ndarray): Input array
        
    Returns:
        np.ndarray: ReLU of input
    """
    return np.maximum(0, x)


def relu_derivative(x):
    """
    Derivative of ReLU activation function.
    
    Args:
        x (np.ndarray): Input array
        
    Returns:
        np.ndarray: Derivative of ReLU at input
    """
    return np.where(x > 0, 1, 0)


def tanh(x):
    """
    Hyperbolic tangent activation function.
    
    Args:
        x (np.ndarray): Input array
        
    Returns:
        np.ndarray: Tanh of input
    """
    return np.tanh(x)


def tanh_derivative(x):
    """
    Derivative of hyperbolic tangent activation function.
    
    Args:
        x (np.ndarray): Input array
        
    Returns:
        np.ndarray: Derivative of tanh at input
    """
    return 1 - np.tanh(x)**2
