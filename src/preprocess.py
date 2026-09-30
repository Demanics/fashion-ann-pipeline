import os
import numpy as np
import yaml
from sklearn.model_selection import train_test_split

p = yaml.safe_load(open("params.yaml"))["preprocess"]


# def normalize(x):
#     return x.astype("float32") / 255.0

def normalize(x):
    return np.sqrt(x.astype("float32") / 255.0)


d = np.load("data/raw/fashion_mnist.npz")
xtrain, xvalid, ytrain, yvalid = train_test_split(
    d["xtrain"],
    d["ytrain"],
    test_size=p["test_size"],
    random_state=p["seed"],
    stratify=d["ytrain"],
)

os.makedirs("data/preprocessed", exist_ok=True)
np.savez_compressed(
    "data/preprocessed/data.npz",
    xtrain=normalize(xtrain),
    ytrain=ytrain,
    xvalid=normalize(xvalid),
    yvalid=yvalid,
    xtest=normalize(d["xtest"]),
    ytest=d["ytest"],
)
