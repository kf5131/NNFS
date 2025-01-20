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
        output = self.nn.forward(X)
        self.assertEqual(output.shape, (1, 1))
        # Add value range check
        self.assertTrue(np.all((output >= 0) & (output <= 1)))
        # Test with batch of samples
        X_batch = np.random.rand(5, 2)
        output_batch = self.nn.forward(X_batch)
        self.assertEqual(output_batch.shape, (5, 1))
        
    def test_evaluation(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [0]])
        
        # Train the network
        self.nn.train(X, y, epochs=100, learning_rate=0.1)
        
        # Test evaluation
        loss = self.nn.evaluate(X, y)
        self.assertIsInstance(loss, float)
        self.assertTrue(loss >= 0)
        
        # Test predictions
        predictions = self.nn.predict(X)
        self.assertEqual(predictions.shape, (4, 1))
        self.assertTrue(np.all((predictions >= 0) & (predictions <= 1)))
        
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
