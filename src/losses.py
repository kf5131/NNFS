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

class CategoricalCrossEntropy(Loss):
    """Categorical Cross Entropy loss for multi-class classification"""
    def calculate(self, predicted, actual):
        """
        Calculate CCE loss
        
        Args:
            predicted (np.ndarray): Predicted probabilities for each class
            actual (np.ndarray): One-hot encoded actual values
            
        Returns:
            float: CCE loss value
        """
        epsilon = 1e-15
        predicted = np.clip(predicted, epsilon, 1 - epsilon)
        loss = -np.sum(actual * np.log(predicted)) / predicted.shape[0]
        return loss
        
    def derivative(self, predicted, actual):
        """
        Calculate derivative of categorical cross entropy
        
        Args:
            predicted (np.ndarray): Predicted probabilities for each class
            actual (np.ndarray): One-hot encoded actual values
            
        Returns:
            np.ndarray: Loss gradient
        """
        epsilon = 1e-15
        predicted = np.clip(predicted, epsilon, 1 - epsilon)
        return (predicted - actual) / predicted.shape[0]
