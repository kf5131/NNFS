import numpy as np
import pytest
from src.activations import sigmoid, sigmoid_derivative, relu, relu_derivative, tanh, tanh_derivative

def test_sigmoid():
    # Test basic sigmoid functionality
    assert sigmoid(0) == 0.5
    assert 0 < sigmoid(-10) < 0.5
    assert 0.5 < sigmoid(10) < 1

    # Test array input
    x = np.array([-1, 0, 1])
    expected = np.array([0.26894142, 0.5, 0.73105858])
    np.testing.assert_array_almost_equal(sigmoid(x), expected)

def test_sigmoid_derivative():
    # Test basic derivative functionality
    x = np.array([-1, 0, 1])
    expected = np.array([0.19661193, 0.25, 0.19661193])
    np.testing.assert_array_almost_equal(sigmoid_derivative(x), expected)

def test_relu():
    # Test basic ReLU functionality
    assert relu(-1) == 0
    assert relu(0) == 0
    assert relu(1) == 1

    # Test array input
    x = np.array([-1, 0, 1])
    expected = np.array([0, 0, 1])
    np.testing.assert_array_equal(relu(x), expected)

def test_relu_derivative():
    # Test basic derivative functionality
    x = np.array([-1, 0, 1])
    expected = np.array([0, 0, 1])
    np.testing.assert_array_equal(relu_derivative(x), expected)

def test_tanh():
    # Test basic tanh functionality
    assert -1 < tanh(-10) < 0
    assert tanh(0) == 0
    assert 0 < tanh(10) < 1

    # Test array input
    x = np.array([-1, 0, 1])
    expected = np.array([-0.76159416, 0, 0.76159416])
    np.testing.assert_array_almost_equal(tanh(x), expected)

def test_tanh_derivative():
    # Test basic derivative functionality
    x = np.array([-1, 0, 1])
    expected = np.array([0.41997434, 1, 0.41997434])
    np.testing.assert_array_almost_equal(tanh_derivative(x), expected)
