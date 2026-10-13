# CS 4375 HW2: Neural Networks (Programming Part)

Shivesh Gupta

**Dataset:** UCI Car Evaluation, https://archive.ics.uci.edu/dataset/19/car+evaluation
(1728 cars, 6 attributes, 4 classes: unacc, acc, good, vgood). The data is not included in the submission. The code downloads it from a copy on GitHub:
https://raw.githubusercontent.com/SGcoder120/ML-assignment-2/refs/heads/master/car.data

## How to run
Needs Python 3 and the packages in `requirements.txt` (numpy, pandas, scikit-learn, matplotlib).

    pip install -r requirements.txt
    python NeuralNet.py

You need an internet connection to download the data. To use a different file or URL, change `DATA_URL` at the top of `NeuralNet.py`.
Nothing is hard-coded to a local path. Everything is saved to a `results/` folder in the current directory.
`OUT_DIR`, `SEED` (42) and `NEURONS` (16) at the top of the file can also be changed.

## What the program does
It trains 24 neural networks, one for every combination of:
- activation: logistic, tanh, relu
- learning rate: 0.01, 0.1
- epochs: 100, 200
- hidden layers: 2, 3

After every epoch it records accuracy and error on the training and test data.

## Output (in `results/`)
- `results_table.csv` (also printed): hyperparameters, train and test accuracy, and train and test error for each model.
- `history_test_acc.png`, `history_train_acc.png`, `history_test_loss.png`, `history_train_loss.png`: all 24 models, split into three panels (one per activation). The color is the learning rate and number of layers, solid lines are 200 epochs and dashed lines are 100 epochs.
- `per_model/`: one plot of accuracy and error against epochs for each model.

## Changes to the starter code
I kept the `NeuralNet` class, `preprocess`, `train_evaluate`, the hyperparameter lists and `train_test_split`. What I changed:
- `car.data` has no header row, so the constructor is called with `header=False` and the column names are in the code.
- Filled in `preprocess` (missing values, checking the categories, converting to numbers, standardizing).
- `train_evaluate` uses a stratified split with a fixed seed and scikit-learn's `MLPClassifier` (SGD with the default momentum, batch size 200, constant learning rate). It trains one epoch at a time with `partial_fit` so the results after each epoch can be saved. It also writes the table and plots.
- The data is loaded from the URL in `DATA_URL` instead of a local `train.csv`.

## Assumptions
- Every hidden layer has 16 neurons.
- All six attributes are ordered categories, so they are converted to numbers in order (for example low=0, med=1, high=2, vhigh=3; "5more" doors = 5; "more" persons = 6) and then standardized. The standardizing uses the whole dataset before the train/test split, so a little test information leaks into the scaling. The values only take a few fixed numbers, so the effect should be tiny.
- The split is 75% train and 25% test, stratified because about 70% of the cars are "unacc", with seed 42. Each configuration is run once, so small differences between models could just be randomness.
- "Error" is cross-entropy loss. Accuracy is the share of cars classified correctly.
- The dataset has no missing values, but the code still drops rows with missing values and checks that every value is a valid category.
- The 100-epoch and 200-epoch runs use the same seed, so a 100-epoch curve is the same as the first 100 epochs of the matching 200-epoch curve.
