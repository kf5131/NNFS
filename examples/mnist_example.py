import numpy as np
import urllib.request
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from src.network import NeuralNetwork
import os

def download_url(url, filename, max_retries=3):
    """Download a file with progress bar and retry mechanism"""
    for attempt in range(max_retries):
        try:
            response = urllib.request.urlopen(url, timeout=30)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filename, 'wb') as f, tqdm(
                desc=f'{filename} (Attempt {attempt + 1}/{max_retries})',
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in iter(lambda: response.read(1024), b''):
                    size = f.write(data)
                    pbar.update(size)
            return  # Success, exit function
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"\nDownload failed. Retrying in {wait_time} seconds... ({str(e)})")
                time.sleep(wait_time)
            else:
                raise Exception(f"Failed to download {filename} after {max_retries} attempts: {str(e)}")

def _mnist_files_exist():
    """Check if MNIST CSV files are already downloaded"""
    files = ['mnist_train.csv', 'mnist_test.csv']
    return all(os.path.exists(f) for f in files)

def _validate_mnist_files():
    """Validate MNIST CSV files have correct format"""
    try:
        # Check training file
        with open('mnist_train.csv', 'r') as f:
            first_line = f.readline()
            if len(first_line.split(',')) != 785:  # 784 pixels + 1 label
                return False
        
        # Check test file
        with open('mnist_test.csv', 'r') as f:
            first_line = f.readline()
            if len(first_line.split(',')) != 785:
                return False
        return True
    except Exception:
        return False

def fetch_mnist(force_download=False):
    """Fetch MNIST dataset in CSV format
    
    Args:
        force_download (bool): If True, force redownload even if files exist
    """
    if not force_download and _mnist_files_exist() and _validate_mnist_files():
        print("MNIST dataset files already exist and are valid, skipping download...")
    else:
        print("Downloading MNIST dataset...")
        base_url = 'https://pjreddie.com/media/files/'
        files = [
            'mnist_train.csv',
            'mnist_test.csv'
        ]
        
        # Remove existing files if they exist
        for filename in files:
            if os.path.exists(filename):
                os.remove(filename)
        
        for filename in files:
            download_url(base_url + filename, filename)
    
    # Read training data
    print("Loading training data...")
    data_train = np.loadtxt('mnist_train.csv', delimiter=',')
    X_train = data_train[:, 1:]  # All columns except first
    y_train = data_train[:, 0]   # First column is label
        
    # Read test data
    print("Loading test data...")
    data_test = np.loadtxt('mnist_test.csv', delimiter=',')
    X_test = data_test[:, 1:]    # All columns except first
    y_test = data_test[:, 0]     # First column is label
        
    return (X_train, y_train), (X_test, y_test)

def main():
    # Fetch MNIST dataset
    (X_train, y_train), (X_test, y_test) = fetch_mnist(force_download=False)

    # Normalize pixel values to range [0, 1]
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Convert labels to one-hot encoding
    num_classes = 10
    y_train_onehot = np.zeros((y_train.size, num_classes))
    y_test_onehot = np.zeros((y_test.size, num_classes))
    y_train_onehot[np.arange(y_train.size), y_train.astype(int)] = 1
    y_test_onehot[np.arange(y_test.size), y_test.astype(int)] = 1

    # Create and train neural network with improved parameters
    print("Training neural network...")
    input_size = X_train.shape[1]
    nn = NeuralNetwork(
        layer_sizes=[input_size, 256, 128, 64, num_classes],  # Slightly smaller network
        activation='relu',
        loss='cce',
        use_dropout=True,
        dropout_rate=0.15,  # Reduced dropout rate
        use_batch_norm=True
    )
    
    history = nn.train(
        X_train, 
        y_train_onehot,
        epochs=150,
        learning_rate=0.01,  # Reduced learning rate
        batch_size=128,  # Increased batch size
        validation_data=(X_test, y_test_onehot)
    )

    # Evaluate on test set
    predictions = nn.predict(X_test)
    accuracy = np.mean(np.argmax(predictions, axis=1) == y_test)
    print(f"Test accuracy: {accuracy:.4f}")

    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(history)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
