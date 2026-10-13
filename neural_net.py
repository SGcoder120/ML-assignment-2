#####################################################################################################################
#   Assignment 2: Neural Network Analysis
#   Dataset: UCI Car Evaluation (https://archive.ics.uci.edu/dataset/19/car+evaluation)
#
#   Usage:   python NeuralNet.py                      (loads the data from the public URL below)
#            python NeuralNet.py /path/to/car.data    (optional: local file or other URL)
#####################################################################################################################

import itertools
import os
import sys

import matplotlib
matplotlib.use("Agg")  # save figures to files instead of opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

DATA_URL = "https://raw.githubusercontent.com/SGcoder120/ML-assignment-2/refs/heads/master/car.data"
OUT_DIR = "results"
SEED = 42
NEURONS = 16  # neurons in each hidden layer

COLUMNS = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]

# every attribute is ordinal, so map it to integers in order
CODES = {
    "buying":   {"low": 0, "med": 1, "high": 2, "vhigh": 3},
    "maint":    {"low": 0, "med": 1, "high": 2, "vhigh": 3},
    "doors":    {"2": 2, "3": 3, "4": 4, "5more": 5},
    "persons":  {"2": 2, "4": 4, "more": 6},
    "lug_boot": {"small": 0, "med": 1, "big": 2},
    "safety":   {"low": 0, "med": 1, "high": 2},
    "class":    {"unacc": 0, "acc": 1, "good": 2, "vgood": 3},
}

class NeuralNet:
    def __init__(self, dataFile, header=True):
        if header:
            self.raw_input = pd.read_csv(dataFile, dtype=str, na_values="?")
        else:
            self.raw_input = pd.read_csv(dataFile, header=None, names=COLUMNS, dtype=str, na_values="?")
        os.makedirs(os.path.join(OUT_DIR, "per_model"), exist_ok=True)

    def preprocess(self):
        df = self.raw_input.copy()

        # missing values: drop rows with any
        n_missing = int(df.isnull().any(axis=1).sum())
        df = df.dropna()

        # categorical -> numeric; a value not in CODES becomes NaN, which fails the integrity check
        for col, codes in CODES.items():
            df[col] = df[col].str.strip().map(codes)
        if df.isnull().any().any():
            raise ValueError("data contains values outside the expected categories")

        # standardize the attributes (not the class column)
        features = df.columns[:-1]
        df[features] = (df[features] - df[features].mean()) / df[features].std(ddof=0)

        print(f"{len(self.raw_input)} rows, {n_missing} with missing values dropped")
        self.processed_data = df
        return 0

    def train_evaluate(self):
        ncols = len(self.processed_data.columns)
        X = self.processed_data.iloc[:, 0:(ncols - 1)].to_numpy()
        y = self.processed_data.iloc[:, (ncols - 1)].astype(int).to_numpy()
        # stratified because most cars are "unacc"
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=SEED, stratify=y)

        activations = ['logistic', 'tanh', 'relu']
        learning_rate = [0.01, 0.1]
        max_iterations = [100, 200]  # also known as epochs
        num_hidden_layers = [2, 3]

        results = []
        histories = {}
        for act, lr, epochs, layers in itertools.product(
                activations, learning_rate, max_iterations, num_hidden_layers):
            model = MLPClassifier(hidden_layer_sizes=(NEURONS,) * layers, activation=act,
                                  solver="sgd", learning_rate_init=lr, random_state=SEED)
            hist = {"train_acc": [], "test_acc": [], "train_loss": [], "test_loss": []}

            # train one epoch at a time so we can record the metrics after each one
            for epoch in range(epochs):
                model.partial_fit(X_train, y_train, classes=[0, 1, 2, 3])
                for name, X_s, y_s in [("train", X_train, y_train), ("test", X_test, y_test)]:
                    proba = model.predict_proba(X_s)
                    hist[name + "_acc"].append(np.mean(proba.argmax(axis=1) == y_s))
                    hist[name + "_loss"].append(log_loss(y_s, proba, labels=[0, 1, 2, 3]))

            histories[(act, lr, epochs, layers)] = hist
            results.append({"activation": act, "learning_rate": lr, "epochs": epochs,
                            "hidden_layers": layers, "neurons_per_layer": NEURONS,
                            "train_accuracy": hist["train_acc"][-1], "test_accuracy": hist["test_acc"][-1],
                            "train_error": hist["train_loss"][-1], "test_error": hist["test_loss"][-1]})

        # results table (error = cross-entropy loss)
        table = pd.DataFrame(results)
        table.to_csv(os.path.join(OUT_DIR, "results_table.csv"), index=False)
        print(table.round(4).to_string(index=False))

        # history plots: one figure per metric, one panel per activation function
        metrics = {"test_acc": "Test accuracy", "train_acc": "Training accuracy",
                   "test_loss": "Test error (cross-entropy)", "train_loss": "Training error (cross-entropy)"}
        colors = {(lr, layers): "C%d" % i
                  for i, (lr, layers) in enumerate(itertools.product(learning_rate, num_hidden_layers))}
        for key, label in metrics.items():
            fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
            for ax, act in zip(axes, activations):
                # 200-epoch runs (thick, faded) first so the 100-epoch runs (thin, dashed) show on top
                for epochs in [200, 100]:
                    for lr, layers in colors:
                        ax.plot(range(1, epochs + 1), histories[(act, lr, epochs, layers)][key],
                                color=colors[(lr, layers)],
                                linestyle="-" if epochs == 200 else "--",
                                linewidth=3 if epochs == 200 else 1.3,
                                alpha=0.5 if epochs == 200 else 1,
                                label=f"lr={lr}, layers={layers}, epochs={epochs}")
                ax.set_title(act)
                ax.set_xlabel("Epoch")
                ax.grid(alpha=0.3)
            axes[0].set_ylabel(label)
            axes[-1].legend(fontsize=8)
            fig.suptitle(label + " vs. epochs for all 24 models")
            fig.tight_layout()
            fig.savefig(os.path.join(OUT_DIR, f"history_{key}.png"), dpi=130)
            plt.close(fig)

        # one plot per model: accuracy and error, train and test
        for (act, lr, epochs, layers), hist in histories.items():
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
            ax1.plot(hist["train_acc"], label="train")
            ax1.plot(hist["test_acc"], label="test")
            ax1.set_ylabel("Accuracy")
            ax2.plot(hist["train_loss"], label="train")
            ax2.plot(hist["test_loss"], label="test")
            ax2.set_ylabel("Error (cross-entropy)")
            for ax in (ax1, ax2):
                ax.set_xlabel("Epoch")
                ax.legend()
                ax.grid(alpha=0.3)
            fig.suptitle(f"{act}, lr={lr}, epochs={epochs}, hidden layers={layers}")
            fig.tight_layout()
            fig.savefig(os.path.join(OUT_DIR, "per_model", f"{act}_lr{lr}_ep{epochs}_L{layers}.png"), dpi=100)
            plt.close(fig)

        return 0


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else DATA_URL
    neural_network = NeuralNet(data_path, header=False)
    neural_network.preprocess()
    neural_network.train_evaluate()