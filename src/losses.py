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
        """Calculate CCE loss"""
        epsilon = 1e-7  # Smaller epsilon
        predicted = np.clip(predicted, epsilon, 1 - epsilon)
        return -np.mean(np.sum(actual * np.log(predicted), axis=1))
        
    def derivative(self, predicted, actual):
        """Calculate derivative of categorical cross entropy"""
        epsilon = 1e-7  # Smaller epsilon
        predicted = np.clip(predicted, epsilon, 1 - epsilon)
        return predicted - actual  # Simplified form when using softmax

class BinaryCrossEntropy(Loss):
    """Binary Cross Entropy loss for binary classification"""
    def calculate(self, predicted, actual):
        """
        Calculate BCE loss
        
        Args:
            predicted (np.ndarray): Predicted values
            actual (np.ndarray): Actual values
            
        Returns:
            float: BCE loss value
        """
        # Clip values to avoid log(0)
        epsilon = 1e-15
        predicted = np.clip(predicted, epsilon, 1 - epsilon)
        
        # Calculate binary cross entropy
        return -np.mean(
            actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted)
        )
    
    def derivative(self, predicted, actual):
        """
        Calculate BCE derivative
        
        Args:
            predicted (np.ndarray): Predicted values
            actual (np.ndarray): Actual values
            
        Returns:
            np.ndarray: Loss gradient
        """
        # Clip values to avoid division by zero
        epsilon = 1e-15
        predicted = np.clip(predicted, epsilon, 1 - epsilon)
        
        # Calculate gradient
        return -(actual / predicted - (1 - actual) / (1 - predicted)) / len(actual)
