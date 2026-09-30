import os
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

model = keras.models.load_model("models/model.h5")
d = np.load("data/preprocessed/data.npz")
loss, accuracy = model.evaluate(d["xtest"], d["ytest"], verbose=0)
predictions = model.predict(d["xtest"]).argmax(axis=1)
os.makedirs("reports", exist_ok=True)
ConfusionMatrixDisplay(confusion_matrix(d["ytest"], predictions)).plot()
plt.savefig("reports/confusion_matrix.png")
json.dump(
    {"test_loss": float(loss), "test_accuracy": float(accuracy)},
    open("reports/metrics.json", "w"),
    indent=2,
)
