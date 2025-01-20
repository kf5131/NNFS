import numpy as np

class Loss:
    """Base class for loss functions"""
    def calculate(self, predicted, actual):
        """Calculate loss value"""
        raise NotImplementedError
        
    def derivative(self, predicted, actual):
        """Calculate loss derivative"""
        raise NotImplementedError

class MSE(Loss):
    """Mean Squared Error loss"""
    def calculate(self, predicted, actual):
        """
        Calculate MSE loss
        
        Args:
            predicted (np.ndarray): Predicted values
            actual (np.ndarray): Actual values
            
        Returns:
            float: MSE loss value
        """
        return np.mean(np.square(predicted - actual))
        
    def derivative(self, predicted, actual):
        """
        Calculate MSE derivative
        
        Args:
            predicted (np.ndarray): Predicted values
            actual (np.ndarray): Actual values
            
        Returns:
            np.ndarray: Loss gradient
        """
        return 2 * (predicted - actual) / predicted.size

class BinaryCrossEntropy(Loss):
    """Binary Cross Entropy loss"""
    def calculate(self, predicted, actual):
        """
        Calculate BCE loss
        
        Args:
            predicted (np.ndarray): Predicted values (after sigmoid)
            actual (np.ndarray): Actual values
            
        Returns:
            float: BCE loss value
        """
        epsilon = 1e-15  # Small constant to avoid log(0)
        predicted = np.clip(predicted, epsilon, 1 - epsilon)
        return -np.mean(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted))
        
    def derivative(self, predicted, actual):
        """
        Calculate BCE derivative
        
        Args:
            predicted (np.ndarray): Predicted values (after sigmoid)
            actual (np.ndarray): Actual values
            
        Returns:
            np.ndarray: Loss gradient
        """
        epsilon = 1e-15
        predicted = np.clip(predicted, epsilon, 1 - epsilon)
        return -(actual / predicted - (1 - actual) / (1 - predicted)) / predicted.size
