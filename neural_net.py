#####################################################################################################################
#   Assignment 2: Neural Network Analysis
#   Dataset: UCI Car Evaluation (https://archive.ics.uci.edu/dataset/19/car+evaluation)
#####################################################################################################################

import itertools
import os
import sys

import matplotlib
matplotlib.use("Agg")  # save plots as image files instead of opening a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Literal
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

DATA_URL = "https://raw.githubusercontent.com/SGcoder120/ML-assignment-2/refs/heads/master/car.data"
OUT_DIR = "results"  # folder where the table and plots are saved
SEED = 42            # fixed seed so the results are the same every run
NEURONS = 16         # neurons in each hidden layer

# car.data has no header row, so we name the columns here
COLUMNS = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]

# every column is an ordered category, so turn each value into a number in the same order, (for example low=0, med=1, high=2, vhigh=3)
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
        # header=False means the file has no column names (like car.data)
        if header:
            self.raw_input = pd.read_csv(dataFile, dtype=str, na_values="?")
        else:
            self.raw_input = pd.read_csv(dataFile, header=None, names=COLUMNS, dtype=str, na_values="?")
        os.makedirs(os.path.join(OUT_DIR, "per_model"), exist_ok=True)

    def preprocess(self):
        df = self.raw_input.copy()

        # remove any rows that have a missing value
        n_missing = int(df.isnull().any(axis=1).sum())
        df = df.dropna()

        # change the text values into numbers, a value we don't recognize becomes NaN, and then we stop with an error
        for col, codes in CODES.items():
            df[col] = df[col].str.strip().map(codes)
        if df.isnull().any().any():
            raise ValueError("data contains values outside the expected categories")

        # standardize the inputs (mean 0, standard deviation 1), but leave the class column alone
        features = df.columns[:-1]
        df[features] = (df[features] - df[features].mean()) / df[features].std(ddof=0)

        print(f"{len(self.raw_input)} rows, {n_missing} with missing values dropped")
        self.processed_data = df
        return 0

    def train_evaluate(self):
        # X = the six input columns, y = the class column
        ncols = len(self.processed_data.columns)
        X = self.processed_data.iloc[:, 0:(ncols - 1)].to_numpy()
        y = self.processed_data.iloc[:, (ncols - 1)].astype(int).to_numpy()

        # 75% of the rows for training, 25% for testing, stratify keeps the same share of each class in both parts (most cars are "unacc")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=SEED, stratify=y)

        # the hyperparameter values to try (every combination gets its own model)
        activations: list[Literal['logistic', 'tanh', 'relu']] = ['logistic', 'tanh', 'relu']
        learning_rate = [0.01, 0.1]
        max_iterations = [100, 200]  # epochs
        num_hidden_layers = [2, 3]

        results = []    # one row per model for the results table
        histories = {}  # accuracy and loss after every epoch, for the plots
        for act, lr, epochs, layers in itertools.product(
                activations, learning_rate, max_iterations, num_hidden_layers):
            model = MLPClassifier(hidden_layer_sizes=(NEURONS,) * layers, activation=act,
                                  solver="sgd", learning_rate_init=lr, random_state=SEED)
            hist = {"train_acc": [], "test_acc": [], "train_loss": [], "test_loss": []}

            # train for one epoch at a time, and measure after each epoch
            for epoch in range(epochs):
                model.partial_fit(X_train, y_train, classes=[0, 1, 2, 3])
                for name, X_s, y_s in [("train", X_train, y_train), ("test", X_test, y_test)]:
                    proba = model.predict_proba(X_s)
                    hist[name + "_acc"].append(np.mean(proba.argmax(axis=1) == y_s))
                    hist[name + "_loss"].append(log_loss(y_s, proba, labels=[0, 1, 2, 3]))

            # save the final numbers after the last epoch
            histories[(act, lr, epochs, layers)] = hist
            results.append({"activation": act, "learning_rate": lr, "epochs": epochs,
                            "hidden_layers": layers, "neurons_per_layer": NEURONS,
                            "train_accuracy": hist["train_acc"][-1], "test_accuracy": hist["test_acc"][-1],
                            "train_error": hist["train_loss"][-1], "test_error": hist["test_loss"][-1]})

        # save and print the results table (error = cross-entropy loss)
        table = pd.DataFrame(results)
        table.to_csv(os.path.join(OUT_DIR, "results_table.csv"), index=False)
        print(table.round(4).to_string(index=False))

        # 4 plots: test accuracy, train accuracy, test error, train error; each plot has 3 panels, one for each activation function
        metrics = {"test_acc": "Test accuracy", "train_acc": "Training accuracy",
                   "test_loss": "Test error (cross-entropy)", "train_loss": "Training error (cross-entropy)"}
        # one color for each learning rate / number of layers combination
        colors = {(lr, layers): "C%d" % i
                  for i, (lr, layers) in enumerate(itertools.product(learning_rate, num_hidden_layers))}
        for key, label in metrics.items():
            fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
            for ax, act in zip(axes, activations):
                # draw the 200 epoch lines first (thick), then the 100 epoch lines (dashed) on top
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

        # one plot for every model: accuracy and error, for train and test
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
    data_path = DATA_URL
    neural_network = NeuralNet(data_path, header=False)
    neural_network.preprocess()
    neural_network.train_evaluate()