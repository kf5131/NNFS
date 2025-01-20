import unittest
import numpy as np
from src.losses import MSE, BinaryCrossEntropy

class TestLosses(unittest.TestCase):
    def setUp(self):
        self.mse = MSE()
        self.bce = BinaryCrossEntropy()
        
    def test_mse_calculation(self):
        predicted = np.array([1.0, 2.0, 3.0])
        actual = np.array([0.8, 2.1, 2.9])
        expected_loss = np.mean(np.square(predicted - actual))
        calculated_loss = self.mse.calculate(predicted, actual)
        self.assertAlmostEqual(expected_loss, calculated_loss)
        
    def test_mse_derivative(self):
        predicted = np.array([1.0, 2.0, 3.0])
        actual = np.array([0.8, 2.1, 2.9])
        expected_derivative = 2 * (predicted - actual) / predicted.size
        calculated_derivative = self.mse.derivative(predicted, actual)
        np.testing.assert_array_almost_equal(expected_derivative, calculated_derivative)
        
    def test_bce_calculation(self):
        predicted = np.array([0.7, 0.3, 0.9])
        actual = np.array([1, 0, 1])
        epsilon = 1e-15
        predicted_clipped = np.clip(predicted, epsilon, 1 - epsilon)
        expected_loss = -np.mean(actual * np.log(predicted_clipped) + 
                               (1 - actual) * np.log(1 - predicted_clipped))
        calculated_loss = self.bce.calculate(predicted, actual)
        self.assertAlmostEqual(expected_loss, calculated_loss)
        
    def test_bce_derivative(self):
        predicted = np.array([0.7, 0.3, 0.9])
        actual = np.array([1, 0, 1])
        epsilon = 1e-15
        predicted_clipped = np.clip(predicted, epsilon, 1 - epsilon)
        expected_derivative = -(actual / predicted_clipped - 
                              (1 - actual) / (1 - predicted_clipped)) / predicted.size
        calculated_derivative = self.bce.derivative(predicted, actual)
        np.testing.assert_array_almost_equal(expected_derivative, calculated_derivative)

if __name__ == '__main__':
    unittest.main()

