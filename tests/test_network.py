import unittest
import numpy as np
from src.network import NeuralNetwork

class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        self.nn = NeuralNetwork([2, 3, 1], activation='sigmoid', loss='mse')
        
    def test_initialization(self):
        # Test layer sizes
        self.assertEqual(len(self.nn.weights), 2)
        self.assertEqual(len(self.nn.biases), 2)
        self.assertEqual(self.nn.weights[0].shape, (2, 3))
        self.assertEqual(self.nn.weights[1].shape, (3, 1))
        self.assertEqual(self.nn.biases[0].shape, (1, 3))
        self.assertEqual(self.nn.biases[1].shape, (1, 1))
        
    def test_forward_pass(self):
        X = np.array([[0.5, 0.1]])
        activations = self.nn.forward(X)
        self.assertEqual(len(activations), 3)  # Input + 2 hidden layers
        self.assertEqual(activations[0].shape, (1, 2))  # Input layer
        self.assertEqual(activations[1].shape, (1, 3))  # Hidden layer
        self.assertEqual(activations[2].shape, (1, 1))  # Output layer
        
    def test_training(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [0]])
        history = self.nn.train(X, y, epochs=100, verbose=False)
        self.assertEqual(len(history), 100)
        self.assertIsInstance(history[0], float)
        
    def test_prediction(self):
        X = np.array([[0.5, 0.1]])
        prediction = self.nn.predict(X)
        self.assertEqual(prediction.shape, (1, 1))
        self.assertTrue(0 <= prediction[0,0] <= 1)
        
    def test_invalid_activation(self):
        with self.assertRaises(ValueError):
            NeuralNetwork([2, 3, 1], activation='invalid')
            
    def test_invalid_loss(self):
        with self.assertRaises(ValueError):
            NeuralNetwork([2, 3, 1], loss='invalid')

if __name__ == '__main__':
    unittest.main()
