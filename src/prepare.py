import os
import numpy as np
from tensorflow import keras

os.makedirs("data/raw", exist_ok=True)
(xtrain, ytrain), (xtest, ytest) = keras.datasets.fashion_mnist.load_data()
np.savez_compressed(
    "data/raw/fashion_mnist.npz", xtrain=xtrain, ytrain=ytrain, xtest=xtest, ytest=ytest
)
