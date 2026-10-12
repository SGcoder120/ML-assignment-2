#####################################################################################################################
#   Assignment 2: Neural Network Analysis
#   Dataset: UCI Car Evaluation  (https://archive.ics.uci.edu/dataset/19/car+evaluation)
#   Task:    multi-class classification (unacc / acc / good / vgood)
#
#   Usage:   python NeuralNet.py --data car.data --out results --seed 42
#   See README.md for details and for the list of changes made to the starter code.
#####################################################################################################################

import argparse
import itertools
import os

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

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
    def __init__(self, dataFile, header=False, out_dir="results", seed=42,
                 neurons_per_layer=16, test_size=0.25):
        # The UCI car.data file has no header row, so column names are supplied here.
        self.raw_input = pd.read_csv(dataFile, header=0 if header else None,
                                     names=None if header else COLUMNS)
        self.out_dir = out_dir
        self.seed = seed
        self.neurons_per_layer = neurons_per_layer
        self.test_size = test_size
        os.makedirs(os.path.join(out_dir, "per_model"), exist_ok=True)

    # ------------------------------------------------------------------
    # Pre-processing: nulls, integrity checks, categorical -> numeric,
    # train/test split, standardization (scaler fitted on training data only).
    # ------------------------------------------------------------------
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

        # Categorical -> numerical.
        for col, mapping in ORDINAL_MAP.items():
            df[col] = df[col].map(mapping).astype(float)
        df["class"] = df["class"].map({c: i for i, c in enumerate(CLASS_ORDER)})

        X = df.drop(columns="class")
        y = df["class"].astype(int)

        # Stratified split because the classes are highly imbalanced (~70% 'unacc').
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.seed, stratify=y)

        # Standardization: fit on training data only to avoid leakage into the test set.
        scaler = StandardScaler().fit(X_train)
        self.X_train = scaler.transform(X_train)
        self.X_test = scaler.transform(X_test)
        self.y_train = y_train.to_numpy()
        self.y_test = y_test.to_numpy()

        self.processed_data = df
        print(f"Train size: {len(self.y_train)}, test size: {len(self.y_test)}")
        print("Class distribution:", df["class"].map(dict(enumerate(CLASS_ORDER))).value_counts().to_dict())
        return 0

    # ------------------------------------------------------------------
    # Train and evaluate every hyperparameter combination.
    # ------------------------------------------------------------------
    def train_evaluate(self):
        activations = ['logistic', 'tanh', 'relu']
        learning_rate = [0.01, 0.1]
        max_iterations = [100, 200]  # epochs
        num_hidden_layers = [2, 3]

        classes = np.arange(len(CLASS_ORDER))
        rows, histories = [], {}

        for act, lr, epochs, layers in itertools.product(
                activations, learning_rate, max_iterations, num_hidden_layers):
            key = (act, lr, epochs, layers)
            model = MLPClassifier(
                hidden_layer_sizes=(self.neurons_per_layer,) * layers,
                activation=act, solver="sgd", learning_rate_init=lr,
                random_state=self.seed)

            hist = {"train_acc": [], "test_acc": [], "train_loss": [], "test_loss": []}
            for _ in range(epochs):
                model.partial_fit(self.X_train, self.y_train, classes=classes)  # exactly one epoch
                for split, X, y in (("train", self.X_train, self.y_train),
                                    ("test", self.X_test, self.y_test)):
                    proba = model.predict_proba(X)
                    hist[f"{split}_acc"].append(float((proba.argmax(axis=1) == y).mean()))
                    hist[f"{split}_loss"].append(float(log_loss(y, proba, labels=classes)))
            histories[key] = hist

            rows.append({
                "activation": act, "learning_rate": lr, "epochs": epochs,
                "hidden_layers": layers, "neurons_per_layer": self.neurons_per_layer,
                "train_accuracy": hist["train_acc"][-1], "test_accuracy": hist["test_acc"][-1],
                "train_error": hist["train_loss"][-1], "test_error": hist["test_loss"][-1],
            })
            print(f"{act:9s} lr={lr:<5} epochs={epochs:<4} layers={layers}  "
                  f"train_acc={rows[-1]['train_accuracy']:.4f} test_acc={rows[-1]['test_accuracy']:.4f} "
                  f"train_loss={rows[-1]['train_error']:.4f} test_loss={rows[-1]['test_error']:.4f}")

        self.results = pd.DataFrame(rows)
        self.histories = histories
        self.results.to_csv(os.path.join(self.out_dir, "results_table.csv"), index=False)
        print("\n=== Results table (error = cross-entropy loss) ===")
        print(self.results.round(4).to_string(index=False))

        self._plot_all(activations)
        self._plot_each()
        return 0

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def _plot_all(self, activations):
        """One figure per metric, split into one panel per activation function (the
        assignment allows splitting a congested plot into 2-3 parts). Colour = (lr, layers)
        combination, line style = epochs (solid 200, dashed 100); the same colour scheme is used in every panel."""
        combos = list(itertools.product([0.01, 0.1], [2, 3], [100, 200]))
        cmap = plt.get_cmap("tab10")
        pairs = list(itertools.product([0.01, 0.1], [2, 3]))
        color = {(lr, l, e): cmap(pairs.index((lr, l))) for (lr, l, e) in combos}
        metrics = [("test_acc", "Test accuracy"), ("train_acc", "Training accuracy"),
                   ("test_loss", "Test error (cross-entropy)"), ("train_loss", "Training error (cross-entropy)")]
        for m, title in metrics:
            fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
            for ax, act in zip(axes, activations):
                for (lr, layers, epochs) in combos:
                    h = self.histories[(act, lr, epochs, layers)][m]
                    # The 100-epoch run (dashed) uses the same seed as the 200-epoch run (solid),
                    # so it traces the first 100 epochs of it.
                    ax.plot(range(1, len(h) + 1), h, color=color[(lr, layers, epochs)],
                            lw=3.2 if epochs == 200 else 1.3,
                            alpha=0.45 if epochs == 200 else 1.0,
                            zorder=2 if epochs == 200 else 3,
                            ls="-" if epochs == 200 else "--",
                            label=f"lr={lr}, layers={layers}, epochs={epochs}")
                ax.set_title(f"{act}")
                ax.set_xlabel("Epoch")
                ax.grid(alpha=0.3)
            axes[0].set_ylabel(title)
            axes[-1].legend(fontsize=8, loc="best")
            fig.suptitle(f"{title} vs. epochs for all 24 models")
            fig.tight_layout()
            fig.savefig(os.path.join(self.out_dir, f"history_{m}.png"), dpi=130)
            plt.close(fig)

    def _plot_each(self):
        for (act, lr, epochs, layers), h in self.histories.items():
            fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
            x = range(1, epochs + 1)
            a1.plot(x, h["train_acc"], label="train"); a1.plot(x, h["test_acc"], label="test")
            a1.set_xlabel("Epoch"); a1.set_ylabel("Accuracy"); a1.legend(); a1.grid(alpha=0.3)
            a2.plot(x, h["train_loss"], label="train"); a2.plot(x, h["test_loss"], label="test")
            a2.set_xlabel("Epoch"); a2.set_ylabel("Error (cross-entropy)"); a2.legend(); a2.grid(alpha=0.3)
            fig.suptitle(f"{act}, lr={lr}, epochs={epochs}, hidden layers={layers}")
            fig.tight_layout()
            fig.savefig(os.path.join(self.out_dir, "per_model", f"{act}_lr{lr}_ep{epochs}_L{layers}.png"), dpi=100)
            plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural network hyperparameter study on UCI Car Evaluation")
    parser.add_argument("--data", default="car.data", help="path or URL to car.data")
    parser.add_argument("--out", default="results", help="output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--neurons", type=int, default=16, help="neurons per hidden layer")
    args = parser.parse_args()

    neural_network = NeuralNet(args.data, out_dir=args.out, seed=args.seed,
                               neurons_per_layer=args.neurons)
    neural_network.preprocess()
    neural_network.train_evaluate()