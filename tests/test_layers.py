import numpy as np
import pytest
from src.layers import Dense, Activation
from src.activations import sigmoid, sigmoid_derivative, relu, relu_derivative, tanh, tanh_derivative

def test_dense_layer_initialization():
    # Test basic initialization
    input_size = 3
    output_size = 2
    layer = Dense(input_size, output_size)
    
    assert layer.weights.shape == (input_size, output_size)
    assert layer.biases.shape == (1, output_size)

def test_dense_layer_forward():
    # Test forward pass
    input_size = 3
    output_size = 2
    layer = Dense(input_size, output_size)
    
    # Set weights and biases manually for predictable output
    layer.weights = np.array([[0.1, 0.2],
                            [0.3, 0.4],
                            [0.5, 0.6]])
    layer.biases = np.array([[0.1, 0.2]])
    
    inputs = np.array([[1.0, 2.0, 3.0]])
    expected = np.array([[2.3, 3.0]])  # Corrected expected value
    
    output = layer.forward(inputs)
    np.testing.assert_array_almost_equal(output, expected)

def test_dense_layer_backward():
    # Test backward pass
    input_size = 3
    output_size = 2
    layer = Dense(input_size, output_size)
    
    # Forward pass first
    inputs = np.array([[1.0, 2.0, 3.0]])
    layer.forward(inputs)
    
    # Test backward pass
    grad = np.array([[1.0, 1.0]])
    grad_input = layer.backward(grad)
    
    assert grad_input.shape == inputs.shape
    assert layer.weight_gradients.shape == layer.weights.shape
    assert layer.bias_gradients.shape == layer.biases.shape

def test_activation_layer():
    # Test activation layer with different activation functions
    inputs = np.array([[-1.0, 0.0, 1.0]])
    
    # Test sigmoid activation
    sigmoid_layer = Activation(sigmoid, sigmoid_derivative)
    sigmoid_output = sigmoid_layer.forward(inputs)
    expected_sigmoid = np.array([[0.26894142, 0.5, 0.73105858]])
    np.testing.assert_array_almost_equal(sigmoid_output, expected_sigmoid)
    
    # Test ReLU activation
    relu_layer = Activation(relu, relu_derivative)
    relu_output = relu_layer.forward(inputs)
    expected_relu = np.array([[0.0, 0.0, 1.0]])
    np.testing.assert_array_almost_equal(relu_output, expected_relu)
    
    # Test tanh activation
    tanh_layer = Activation(tanh, tanh_derivative)
    tanh_output = tanh_layer.forward(inputs)
    expected_tanh = np.array([[-0.76159416, 0.0, 0.76159416]])
    np.testing.assert_array_almost_equal(tanh_output, expected_tanh)

def test_activation_layer_backward():
    # Test backward pass of activation layer
    inputs = np.array([[-1.0, 0.0, 1.0]])
    grad = np.array([[1.0, 1.0, 1.0]])
    
    # Test sigmoid backward
    sigmoid_layer = Activation(sigmoid, sigmoid_derivative)
    sigmoid_layer.forward(inputs)
    grad_sigmoid = sigmoid_layer.backward(grad)
    assert grad_sigmoid.shape == inputs.shape
    
    # Test ReLU backward
    relu_layer = Activation(relu, relu_derivative)
    relu_layer.forward(inputs)
    grad_relu = relu_layer.backward(grad)
    assert grad_relu.shape == inputs.shape
    
    # Test tanh backward
    tanh_layer = Activation(tanh, tanh_derivative)
    tanh_layer.forward(inputs)
    grad_tanh = tanh_layer.backward(grad)
    assert grad_tanh.shape == inputs.shape
