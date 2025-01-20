# Simple Neural Network from Scratch

This project implements a basic neural network from scratch in Python, demonstrating the fundamental concepts of deep learning without using any machine learning frameworks.

## Features

- Feedforward neural network implementation
- Backpropagation algorithm
- Gradient descent optimization
- Configurable network architecture
- Training and prediction capabilities

## Requirements

- Python 3.6+
- NumPy
- Matplotlib
- scikit-learn
- TQDM
- Development dependencies:
  - pytest
  - pytest-cov
  - black
  - flake8
  - setuptools


## Installation

1. Clone the repository:

```bash
git clone https://github.com/kf5131/NNFS.git
cd NNFS
```
2. Run setup script:

```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

This will:
- Create a virtual environment
- Activate the virtual environment
- Install required dependencies
- Install the package in development mode

## Testing

To run the tests, use the following command:

```bash
chmod +x scripts/run_tests.sh
./scripts/run_tests.sh
```

This will:
- Run all unit tests
- Generate a coverage report
- Display test results and coverage statistics in the console

For running specific test files or test cases:

```bash
# Run a specific test file
pytest tests/test_specific_file.py

# Run a specific test case
pytest tests/test_file.py::test_specific_case

# Run tests with verbose output
pytest -v
```

## Examples

To run the examples, use the following commands:

### MNIST Example
```bash
python examples/mnist_example.py
```

This will run the MNIST example, print the results to the console, and show a plot of the loss over time . Expect to see the accuracy of ~87% after 200 epochs.

