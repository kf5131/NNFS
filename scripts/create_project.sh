#!/bin/bash

# Create main directories
mkdir -p src
mkdir -p examples
mkdir -p tests
mkdir -p scripts
mkdir -p data

# Create source files
touch src/__init__.py
touch src/activations.py
touch src/layers.py
touch src/losses.py
touch src/network.py
touch src/optimizers.py

# Create example files
touch examples/__init__.py
touch examples/mnist_example.py
touch examples/xor_example.py

# Create test files
touch tests/__init__.py
touch tests/test_activations.py
touch tests/test_layers.py
touch tests/test_losses.py
touch tests/test_network.py

# Create script files
touch scripts/setup.sh
touch scripts/run_tests.sh

# Create root level files
touch setup.py
touch requirements.txt
touch README.md
touch main.py

# Create .gitkeep in empty data directory
touch data/.gitkeep

# Make scripts executable
chmod +x scripts/setup.sh
chmod +x scripts/run_tests.sh
chmod +x scripts/create_project.sh

echo "Project structure created successfully!" 