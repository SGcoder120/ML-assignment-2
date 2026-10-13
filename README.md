# CS 4375 HW2 – Programming Part (Neural Networks)

**Dataset:** UCI Car Evaluation – https://archive.ics.uci.edu/dataset/19/car+evaluation
(1728 rows, 6 ordinal attributes, 4-class target: unacc / acc / good / vgood). The data file is not included in the submission.

## Requirements
Python 3, `numpy`, `pandas`, `scikit-learn`, `matplotlib` (see `requirements.txt`)
Install with: `pip install -r requirements.txt`

## How to run
The program loads the data from a public URL by default (a copy of the UCI `car.data` file):
https://raw.githubusercontent.com/SGcoder120/ML-assignment-2/refs/heads/master/car.data

    pip install -r requirements.txt
    python NeuralNet.py

To use a local copy of `car.data` (or another URL) instead, pass it as the only argument:

    python NeuralNet.py /path/to/car.data

No local paths are hard-coded. An internet connection is needed when using the default URL. Output goes to a `results/` folder
created in the current directory. Constants at the top of `NeuralNet.py` set the output folder, random seed (42) and neurons per hidden layer (16).

## Outputs (in `results/`)
- `results_table.csv` (also printed): hyperparameters, training/test accuracy, training/test error for all 24 models.
- `history_test_acc.png`, `history_train_acc.png`, `history_test_loss.png`, `history_train_loss.png`: history of all 24 models,
  split into three panels (one per activation function). Colour = (learning rate, #layers); solid = 200 epochs, dashed = 100 epochs.
- `per_model/*.png`: accuracy and error vs. epochs (train and test) for each individual model.

## Hyperparameters (all 3 x 2 x 2 x 2 = 24 combinations)
activation ∈ {logistic, tanh, relu}; learning rate ∈ {0.01, 0.1}; epochs ∈ {100, 200}; hidden layers ∈ {2, 3}.

## Changes to the starter code
The starter's structure (`NeuralNet` class with `preprocess` and `train_evaluate`, the hyperparameter lists, and `train_test_split`) is kept. Modifications:
- `car.data` has no header row, so the constructor is called with `header=False` and column names are supplied in code.
- `preprocess` is filled in (nulls, integrity checks, ordinal encoding, standardization).
- `train_evaluate` uses a stratified, seeded split (`test_size=0.25`, `random_state=42`), builds scikit-learn `MLPClassifier` models (SGD solver with default momentum, batch size 200, constant learning rate), and trains one epoch at a time with `partial_fit` so per-epoch train/test accuracy and loss can be recorded. It also writes the results table and plots.
- The data path is read from an optional command-line argument and defaults to the public URL above.

## Assumptions
- Every hidden layer has 16 neurons (the assignment allows a fixed number).
- All six attributes are ordinal and are encoded as ordered integers (e.g. low<med<high<vhigh; doors "5more"=5; persons "more"=6), then standardized (zero mean, unit variance) in `preprocess`, i.e. using statistics of the full dataset before the train/test split. Because the attributes have fixed small ranges, the effect on the test set is negligible.
- 75/25 stratified train/test split (stratified because ~70% of cars are "unacc"), fixed seed 42. One run per configuration, so small differences between models are within seed noise.
- "Error" is the multi-class cross-entropy (log loss) on the predicted probabilities; accuracy is the fraction correctly classified.
- No missing values exist in this dataset; the code still drops null rows and verifies that all values are valid categories.
- The 100-epoch and 200-epoch runs use the same seed, so a 100-epoch curve equals the first 100 epochs of the matching 200-epoch curve.

## Brief report
See `report.md`.