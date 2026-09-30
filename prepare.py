# import os
# import numpy as np
# from tensorflow import keras

# os.makedirs("data/raw", exist_ok=True)
# (xtrain, ytrain), (xtest, ytest) = keras.datasets.fashion_mnist.load_data()
# np.savez_compressed(
#     "data/raw/fashion_mnist.npz", xtrain=xtrain, ytrain=ytrain, xtest=xtest, ytest=ytest
# )

import os
import numpy as np
from tensorflow.keras import datasets

def main():
    # Define paths
    output_dir = "data/raw"
    output_file = os.path.join(output_dir, "fashion_mnist.npz")
    
    # 1. Create directory structure securely
    print(f"Creating directories at: '{output_dir}'...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Fetch dataset
    print("Downloading/Loading Fashion MNIST dataset from Keras...")
    (xtrain, ytrain), (xtest, ytest) = datasets.fashion_mnist.load_data()
    
    # 3. Compress and save local copies
    print(f"Saving compressed raw data to '{output_file}'...")
    np.savez_compressed(
        output_file, 
        xtrain=xtrain, 
        ytrain=ytrain, 
        xtest=xtest, 
        ytest=ytest
    )
    
    print("Data ingestion complete successfully!")

if __name__ == "__main__":
    main()
