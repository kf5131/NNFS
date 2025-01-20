import numpy as np
from src.activations import sigmoid, sigmoid_derivative, relu, relu_derivative, tanh, tanh_derivative
from src.losses import MSE, CategoricalCrossEntropy
from tqdm import tqdm

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    # Subtract max for numerical stability (per sample)
    shifted_x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(shifted_x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

class NeuralNetwork:
    """Neural Network implementation"""
    
    def __init__(self, layer_sizes, activation='relu', loss='mse', use_dropout=False, 
                 dropout_rate=0.2, use_batch_norm=False, gradient_clip=5.0):
        """
        Initialize neural network with additional features
        
        Args:
            layer_sizes (list): List of integers for number of neurons in each layer
            activation (str): Activation function to use ('sigmoid', 'relu', or 'tanh')
            loss (str): Loss function to use ('mse' or 'cce')
            use_dropout (bool): Whether to use dropout
            dropout_rate (float): Dropout rate
            use_batch_norm (bool): Whether to use batch normalization
            gradient_clip (float): Maximum allowed gradient norm
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.gradient_clip = gradient_clip
        self.current_step = 0  # Add step counter for learning rate schedule
        
        # Initialize weights, biases, and batch norm parameters
        self.weights = []
        self.biases = []
        if self.use_batch_norm:
            self.gamma = []  # Scale parameter
            self.beta = []   # Shift parameter
            self.running_mean = []
            self.running_var = []
        
        for i in range(self.num_layers - 1):
            # More conservative initialization
            if activation == 'relu':
                scale = np.sqrt(1.0 / layer_sizes[i])  # Changed from 2.0 to 1.0
            else:
                scale = np.sqrt(2.0 / (layer_sizes[i] + layer_sizes[i+1]))  # Xavier/Glorot initialization
            
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * scale)
            self.biases.append(np.zeros((1, layer_sizes[i+1])))  # Initialize all biases to zero
            
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
        # Store all intermediate values needed for backprop
        cache = {}
        activations = [X]
        if training and self.use_dropout:
            dropout_masks = []
        
        for i in range(self.num_layers - 1):
            net = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            
            # Store pre-activation values
            cache[f'net_{i}'] = net
            
            # Apply batch normalization before activation (except last layer)
            if self.use_batch_norm and i < self.num_layers - 2:
                net, x_norm, mu, var = self.batch_norm_forward(
                    net, 
                    self.gamma[i], 
                    self.beta[i],
                    self.running_mean[i],
                    self.running_var[i],
                    training
                )
                # Store batch norm intermediate values
                cache[f'bn_{i}'] = (x_norm, mu, var)
            
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
        
        if training:
            return activations, cache, dropout_masks if self.use_dropout else None
        return activations[-1]
        
    def backward(self, X, y, learning_rate=0.01):
        """Backward pass through network"""
        m = X.shape[0]
        
        # Get activations and cached values
        if self.use_dropout:
            activations, cache, dropout_masks = self.forward(X)
        else:
            activations, cache, _ = self.forward(X)
        
        # Calculate initial error
        loss_value = self.loss.calculate(activations[-1], y)
        delta = self.loss.derivative(activations[-1], y)
        
        # Backpropagate error
        for i in range(self.num_layers - 2, -1, -1):
            # For layers before the last layer
            if i < self.num_layers - 2:
                # First calculate the gradient with respect to pre-activation
                delta = np.dot(delta, self.weights[i+1].T)
                delta = delta * self.activation_derivative(activations[i+1])
                
                # Apply dropout mask if using dropout
                if self.use_dropout:
                    delta *= dropout_masks[i]
                
                # Calculate gradient with respect to batch norm output (if used)
                if self.use_batch_norm:
                    x_norm, mu, var = cache[f'bn_{i}']
                    eps = 1e-5
                    
                    # Get the current layer's pre-activation values
                    current_net = cache[f'net_{i}']
                    
                    # Gradients for batch norm parameters
                    dgamma = np.sum(delta * x_norm, axis=0)
                    dbeta = np.sum(delta, axis=0)
                    
                    # Gradient with respect to normalized input
                    dx_norm = delta * self.gamma[i]
                    
                    # Gradient with respect to variance
                    dvar = np.sum(dx_norm * (current_net - mu) * -0.5 * (var + eps)**(-1.5), axis=0)
                    
                    # Gradient with respect to mean
                    dmu = np.sum(dx_norm * -1/np.sqrt(var + eps), axis=0)
                    dmu += dvar * np.mean(-2 * (current_net - mu), axis=0)
                    
                    # Gradient with respect to input
                    delta = dx_norm / np.sqrt(var + eps)
                    delta += 2 * dvar * (current_net - mu) / m
                    delta += dmu / m
                    
                    # Update batch norm parameters
                    self.gamma[i] -= learning_rate * dgamma
                    self.beta[i] -= learning_rate * dbeta
            
            # Calculate weight and bias gradients
            dW = np.dot(activations[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            
            # Gradient clipping
            dW = np.clip(dW, -self.gradient_clip, self.gradient_clip)
            db = np.clip(db, -self.gradient_clip, self.gradient_clip)
            
            # Update weights and biases
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * db
        
        return loss_value
        
    def train(self, X, y, epochs=50, learning_rate=0.001, learning_rate_schedule=None, batch_size=32, validation_data=None):
        """
        Train the neural network using mini-batch gradient descent
        
        Args:
            X: Input data
            y: Target data
            epochs: Number of epochs to train
            learning_rate: Base learning rate (used if no schedule provided)
            learning_rate_schedule: Optional function that takes step count and returns learning rate
            batch_size: Size of mini-batches
            validation_data: Optional tuple of (X_val, y_val) for validation
        """
        history = {'loss': []}  # Changed to dictionary with 'loss' key
        n_samples = X.shape[0]
        
        for epoch in tqdm(range(epochs), desc='Training Progress'):
            epoch_loss = 0
            indices = np.random.permutation(n_samples)
            
            # Mini-batch training
            for start_idx in range(0, n_samples, batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]
                
                # Get current learning rate from schedule or use fixed rate
                if learning_rate_schedule is not None:
                    current_lr = learning_rate_schedule(self.current_step)
                else:
                    current_lr = learning_rate
                
                loss = self.backward(X_batch, y_batch, current_lr)
                epoch_loss += loss * len(batch_indices)
                self.current_step += 1
            
            # Average loss for the epoch
            epoch_loss /= n_samples
            history['loss'].append(epoch_loss)  # Store loss in history dictionary
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}, LR: {current_lr:.6f}")
        
        return history

    def evaluate(self, X, y):
        """
        Evaluate the model on given data
        
        Args:
            X (np.ndarray): Input data
            y (np.ndarray): Target data
            
        Returns:
            float: Loss value
        """
        predictions = self.forward(X, training=False)
        return self.loss.calculate(predictions, y)
        
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X (np.ndarray): Input data
            
        Returns:
            np.ndarray: Predictions
        """
        return self.forward(X, training=False)  # Make sure training=False for predictions
