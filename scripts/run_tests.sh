#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Run tests with coverage
python -m pytest tests/ -v --cov=src
