import numpy as np
from src.activations import sigmoid, sigmoid_derivative, relu, relu_derivative, tanh, tanh_derivative
from src.losses import MSE, CategoricalCrossEntropy
from tqdm import tqdm

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    # Subtract max for numerical stability
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

class NeuralNetwork:
    """Neural Network implementation"""
    
    def __init__(self, layer_sizes, activation='relu', loss='mse', use_dropout=False, dropout_rate=0.2, use_batch_norm=False):
        """
        Initialize neural network with additional features
        
        Args:
            layer_sizes (list): List of integers for number of neurons in each layer
            activation (str): Activation function to use ('sigmoid', 'relu', or 'tanh')
            loss (str): Loss function to use ('mse' or 'cce')
            use_dropout (bool): Whether to use dropout
            dropout_rate (float): Dropout rate
            use_batch_norm (bool): Whether to use batch normalization
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # Initialize weights, biases, and batch norm parameters
        self.weights = []
        self.biases = []
        if self.use_batch_norm:
            self.gamma = []  # Scale parameter
            self.beta = []   # Shift parameter
            self.running_mean = []
            self.running_var = []
        
        for i in range(self.num_layers - 1):
            # He initialization
            scale = np.sqrt(2.0 / layer_sizes[i])
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * scale)
            self.biases.append(np.zeros((1, layer_sizes[i+1])))
            
            if self.use_batch_norm and i < self.num_layers - 2:  # No batch norm in last layer
                self.gamma.append(np.ones((1, layer_sizes[i+1])))
                self.beta.append(np.zeros((1, layer_sizes[i+1])))
                self.running_mean.append(np.zeros((1, layer_sizes[i+1])))
                self.running_var.append(np.ones((1, layer_sizes[i+1])))

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
        elif loss == 'cce':
            self.loss = CategoricalCrossEntropy()
        else:
            raise ValueError("Loss must be 'mse' or 'cce'")
            
    def batch_norm_forward(self, x, gamma, beta, running_mean, running_var, training=True):
        """Batch normalization forward pass"""
        eps = 1e-5
        momentum = 0.9
        
        if training:
            mu = np.mean(x, axis=0, keepdims=True)
            var = np.var(x, axis=0, keepdims=True) + eps
            x_norm = (x - mu) / np.sqrt(var)
            out = gamma * x_norm + beta
            
            # Update running statistics - update the specific array, not the entire list
            running_mean[:] = momentum * running_mean + (1 - momentum) * mu
            running_var[:] = momentum * running_var + (1 - momentum) * var
            
            return out, x_norm, mu, var
        else:
            # During inference, use running statistics
            x_norm = (x - running_mean) / np.sqrt(running_var + eps)
            out = gamma * x_norm + beta
            return out, x_norm, running_mean, running_var

    def forward(self, X, training=True):
        """Forward pass with dropout and batch normalization"""
        activations = [X]
        if training and self.use_dropout:
            dropout_masks = []
        
        for i in range(self.num_layers - 1):
            net = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            
            # Apply batch normalization before activation (except last layer)
            if self.use_batch_norm and i < self.num_layers - 2:
                net, _, _, _ = self.batch_norm_forward(
                    net, 
                    self.gamma[i], 
                    self.beta[i],
                    self.running_mean[i],
                    self.running_var[i],
                    training
                )
            
            # Apply activation
            if i == self.num_layers - 2:  # Last layer
                activation = softmax(net)
            else:
                activation = self.activation(net)
                
                # Apply dropout during training
                if training and self.use_dropout:
                    mask = (np.random.rand(*activation.shape) > self.dropout_rate) / (1 - self.dropout_rate)
                    activation *= mask
                    dropout_masks.append(mask)
            
            activations.append(activation)
        
        if training and self.use_dropout:
            return activations, dropout_masks
        return activations[-1]  # Return only the final layer output when not training
        
    def backward(self, X, y, learning_rate=0.01):
        """
        Backward pass through network
        """
        m = X.shape[0]
        
        # Get activations (and dropout masks if using dropout)
        if self.use_dropout:
            activations, dropout_masks = self.forward(X)
        else:
            activations = self.forward(X)
        
        # Calculate initial error
        loss_value = self.loss.calculate(activations[-1], y)
        delta = self.loss.derivative(activations[-1], y)
        
        # Store the weighted sums (pre-activations) during forward pass
        weighted_sums = []
        current_input = activations[0]  # Use stored activation instead of X
        for i in range(self.num_layers - 1):
            weighted_sum = np.dot(current_input, self.weights[i]) + self.biases[i]
            weighted_sums.append(weighted_sum)
            if i == self.num_layers - 2:  # Last layer
                current_input = softmax(weighted_sum)
            else:
                current_input = self.activation(weighted_sum)
                if self.use_dropout:
                    current_input *= dropout_masks[i]
        
        # Backpropagate error
        for i in range(self.num_layers - 2, -1, -1):
            # For the last layer, we already have the correct delta
            if i < self.num_layers - 2:
                # Calculate gradient with respect to pre-activation
                delta = delta * self.activation_derivative(weighted_sums[i])
                
                # Apply dropout mask if using dropout
                if self.use_dropout:
                    delta *= dropout_masks[i]
            
            # Calculate gradients
            dW = np.dot(activations[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            
            # Gradient clipping to prevent explosion
            dW = np.clip(dW, -1.0, 1.0)
            db = np.clip(db, -1.0, 1.0)
            
            if i > 0:
                delta = np.dot(delta, self.weights[i].T)
            
            # Update weights and biases
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * db
        
        return loss_value
        
    def train(self, X, y, epochs=50, learning_rate=0.001, batch_size=32, validation_data=None):
        """
        Train the neural network using mini-batch gradient descent
        """
        history = []
        n_samples = X.shape[0]
        
        # Learning rate schedule
        initial_lr = learning_rate
        
        for epoch in tqdm(range(epochs), desc='Training Progress'):
            epoch_loss = 0
            # Generate random indices for shuffling
            indices = np.random.permutation(n_samples)
            
            # Adjust learning rate (learning rate decay)
            current_lr = initial_lr / (1 + epoch * 0.1)
            
            # Mini-batch training
            for start_idx in range(0, n_samples, batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]
                
                loss = self.backward(X_batch, y_batch, current_lr)
                epoch_loss += loss * len(batch_indices)
            
            # Average loss for the epoch
            epoch_loss /= n_samples
            history.append(epoch_loss)
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")
        
        return history
        
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X (np.ndarray): Input data
            
        Returns:
            np.ndarray: Predictions
        """
        return self.forward(X, training=False)  # Make sure training=False for predictions
