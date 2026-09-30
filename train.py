import os
import yaml
import numpy as np
import pandas as pd
from tensorflow import keras

p = yaml.safe_load(open("params.yaml"))["train"]
keras.utils.set_random_seed(p["seed"])
d = np.load("data/preprocessed/data.npz")
model = keras.Sequential(
    [
        keras.Input(shape=(28, 28)),
        keras.layers.Flatten(),
        keras.layers.Dense(p["dense_units"], activation="relu"),
        keras.layers.Dropout(p["dropout_rate"]),
        keras.layers.Dense(10, activation="softmax"),
    ]
)

model.compile(
    optimizer=keras.optimizers.Adam(p["learning_rate"]),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

h = model.fit(
    d["xtrain"],
    d["ytrain"],
    validation_data=(d["xvalid"], d["yvalid"]),
    epochs=p["epochs"],
    batch_size=p["batch_size"],
)

os.makedirs("models", exist_ok=True)
model.save("models/model.h5")
pd.DataFrame(h.history).to_csv("models/history.csv", index=False)
