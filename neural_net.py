#####################################################################################################################
#   Assignment 2: Neural Network Analysis
#   This is a starter code in Python 3.6 for a neural network.
#   You need to have numpy and pandas installed before running this code.
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it
#       in the README file.
#
#   Dataset: UCI Car Evaluation (https://archive.ics.uci.edu/dataset/19/car+evaluation)
#   Usage:   python NeuralNet.py                      (loads the data from the public URL below)
#            python NeuralNet.py /path/to/car.data    (optional: local file or other URL)
#
#####################################################################################################################


import itertools
import os
import sys

import matplotlib
matplotlib.use("Agg")  # save figures to files; no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# Public copy of the UCI Car Evaluation data file (no local path needed).
DATA_URL = "https://raw.githubusercontent.com/SGcoder120/ML-assignment-2/refs/heads/master/car.data"
OUT_DIR = "results"
SEED = 42
NEURONS_PER_LAYER = 16  # fixed number of neurons in every hidden layer

# car.data has no header row, so the column names are supplied here.
COLUMNS = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]

# All six attributes are ordinal, so they are mapped to integers that preserve their order.
ORDINAL_MAP = {
    "buying":   {"low": 0, "med": 1, "high": 2, "vhigh": 3},
    "maint":    {"low": 0, "med": 1, "high": 2, "vhigh": 3},
    "doors":    {"2": 2, "3": 3, "4": 4, "5more": 5},
    "persons":  {"2": 2, "4": 4, "more": 6},
    "lug_boot": {"small": 0, "med": 1, "big": 2},
    "safety":   {"low": 0, "med": 1, "high": 2},
}
CLASS_ORDER = ["unacc", "acc", "good", "vgood"]


class NeuralNet:
    def __init__(self, dataFile, header=True):
        # header=False -> the file has no header row (true for the UCI car.data file)
        if header:
            self.raw_input = pd.read_csv(dataFile)
        else:
            self.raw_input = pd.read_csv(dataFile, header=None, names=COLUMNS)
        os.makedirs(os.path.join(OUT_DIR, "per_model"), exist_ok=True)

    # Pre-processing: null handling, integrity checks, categorical -> numerical, standardization.
    def preprocess(self):
        df = self.raw_input.copy()

        # Handle null / missing values and stray whitespace.
        for c in df.columns:
            df[c] = df[c].astype(str).str.strip()
        df = df.replace({"": np.nan, "nan": np.nan, "?": np.nan})
        n_null = int(df.isnull().sum().sum())
        if n_null:
            df = df.dropna()
        print(f"Rows: {len(self.raw_input)}, null cells found: {n_null}, rows after cleaning: {len(df)}")

        # Data integrity: every value must be one of the documented categories.
        for col, mapping in ORDINAL_MAP.items():
            bad = set(df[col]) - set(mapping)
            if bad:
                raise ValueError(f"Unexpected values in column '{col}': {bad}")
        bad = set(df["class"]) - set(CLASS_ORDER)
        if bad:
            raise ValueError(f"Unexpected class labels: {bad}")

        # Categorical -> numerical (ordered integers; class label -> 0..3).
        for col, mapping in ORDINAL_MAP.items():
            df[col] = df[col].map(mapping).astype(float)
        df["class"] = df["class"].map({c: i for i, c in enumerate(CLASS_ORDER)})

        # Standardization of the attributes (zero mean, unit variance); the class column is left as is.
        features = df.columns[:-1]
        df[features] = (df[features] - df[features].mean()) / df[features].std(ddof=0)

        self.processed_data = df
        print("Class distribution:", df["class"].map(dict(enumerate(CLASS_ORDER))).value_counts().to_dict())
        return 0

    # Train and evaluate models for all combinations of parameters.
    # Outputs: train/test accuracy and error per model (table), history curves (plots).
    def train_evaluate(self):
        ncols = len(self.processed_data.columns)
        nrows = len(self.processed_data.index)
        X = self.processed_data.iloc[:, 0:(ncols - 1)]
        y = self.processed_data.iloc[:, (ncols-1)].astype(int)
        # Stratified 75/25 split (classes are imbalanced), fixed seed for reproducibility.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=SEED, stratify=y)
        X_train, X_test = X_train.to_numpy(), X_test.to_numpy()
        y_train, y_test = y_train.to_numpy(), y_test.to_numpy()

        # Below are the hyperparameters that you need to use for model evaluation
        # You can assume any fixed number of neurons for each hidden layer.

        activations = ['logistic', 'tanh', 'relu']
        learning_rate = [0.01, 0.1]
        max_iterations = [100, 200]  # also known as epochs
        num_hidden_layers = [2, 3]

        # Create the neural network and keep track of the performance metrics.
        classes = np.arange(len(CLASS_ORDER))
        rows, histories = [], {}
        for act, lr, epochs, layers in itertools.product(
                activations, learning_rate, max_iterations, num_hidden_layers):
            model = MLPClassifier(hidden_layer_sizes=(NEURONS_PER_LAYER,) * layers,
                                  activation=act, solver="sgd", learning_rate_init=lr,
                                  random_state=SEED)
            hist = {"train_acc": [], "test_acc": [], "train_loss": [], "test_loss": []}
            for _ in range(epochs):
                model.partial_fit(X_train, y_train, classes=classes)  # exactly one epoch
                for split, Xs, ys in (("train", X_train, y_train), ("test", X_test, y_test)):
                    proba = model.predict_proba(Xs)
                    hist[f"{split}_acc"].append(float((proba.argmax(axis=1) == ys).mean()))
                    hist[f"{split}_loss"].append(float(log_loss(ys, proba, labels=classes)))
            histories[(act, lr, epochs, layers)] = hist
            rows.append({"activation": act, "learning_rate": lr, "epochs": epochs,
                         "hidden_layers": layers, "neurons_per_layer": NEURONS_PER_LAYER,
                         "train_accuracy": hist["train_acc"][-1], "test_accuracy": hist["test_acc"][-1],
                         "train_error": hist["train_loss"][-1], "test_error": hist["test_loss"][-1]})
            print(f"{act:9s} lr={lr:<5} epochs={epochs:<4} layers={layers}  "
                  f"train_acc={rows[-1]['train_accuracy']:.4f} test_acc={rows[-1]['test_accuracy']:.4f} "
                  f"train_loss={rows[-1]['train_error']:.4f} test_loss={rows[-1]['test_error']:.4f}")

        # Table of results (error = cross-entropy loss).
        results = pd.DataFrame(rows)
        results.to_csv(os.path.join(OUT_DIR, "results_table.csv"), index=False)
        print("\n=== Results table (error = cross-entropy loss) ===")
        print(results.round(4).to_string(index=False))

        # Plot the model history. One figure per metric, split into three panels (one per
        # activation function) so the 24 curves stay readable. Colour = (learning rate, layers);
        # solid = 200 epochs, dashed = 100 epochs (same seed, so it traces the first 100 epochs).
        combos = list(itertools.product(learning_rate, num_hidden_layers, max_iterations))
        pairs = list(itertools.product(learning_rate, num_hidden_layers))
        cmap = plt.get_cmap("tab10")
        metrics = [("test_acc", "Test accuracy"), ("train_acc", "Training accuracy"),
                   ("test_loss", "Test error (cross-entropy)"), ("train_loss", "Training error (cross-entropy)")]
        for m, title in metrics:
            fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
            for ax, act in zip(axes, activations):
                for lr, layers, epochs in combos:
                    h = histories[(act, lr, epochs, layers)][m]
                    ax.plot(range(1, len(h) + 1), h, color=cmap(pairs.index((lr, layers))),
                            lw=3.2 if epochs == 200 else 1.3, alpha=0.45 if epochs == 200 else 1.0,
                            ls="-" if epochs == 200 else "--", zorder=2 if epochs == 200 else 3,
                            label=f"lr={lr}, layers={layers}, epochs={epochs}")
                ax.set_title(act)
                ax.set_xlabel("Epoch")
                ax.grid(alpha=0.3)
            axes[0].set_ylabel(title)
            axes[-1].legend(fontsize=8, loc="best")
            fig.suptitle(f"{title} vs. epochs for all 24 models")
            fig.tight_layout()
            fig.savefig(os.path.join(OUT_DIR, f"history_{m}.png"), dpi=130)
            plt.close(fig)

        # Individual history plot for every model (accuracy and error vs. epochs, train and test).
        for (act, lr, epochs, layers), h in histories.items():
            fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
            x = range(1, epochs + 1)
            a1.plot(x, h["train_acc"], label="train"); a1.plot(x, h["test_acc"], label="test")
            a1.set_xlabel("Epoch"); a1.set_ylabel("Accuracy"); a1.legend(); a1.grid(alpha=0.3)
            a2.plot(x, h["train_loss"], label="train"); a2.plot(x, h["test_loss"], label="test")
            a2.set_xlabel("Epoch"); a2.set_ylabel("Error (cross-entropy)"); a2.legend(); a2.grid(alpha=0.3)
            fig.suptitle(f"{act}, lr={lr}, epochs={epochs}, hidden layers={layers}")
            fig.tight_layout()
            fig.savefig(os.path.join(OUT_DIR, "per_model", f"{act}_lr{lr}_ep{epochs}_L{layers}.png"), dpi=100)
            plt.close(fig)

        return 0




if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else DATA_URL  # path or URL; nothing hard-coded locally
    neural_network = NeuralNet(data_path, header=False)
    neural_network.preprocess()
    neural_network.train_evaluate()