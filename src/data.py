import os

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "R8")
SEED = 42


def read_file(path):
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                label, text = line.split("\t", 1)
                labels.append(label)
                texts.append(text)
    return texts, np.array(labels)


def load_r8():
    train_x, train_y = read_file(os.path.join(DATA, "train.txt"))
    test_x, test_y = read_file(os.path.join(DATA, "test.txt"))
    return train_x, train_y, test_x, test_y


def train_val_split(train_x, train_y):
    # 80/20 stratified split of the training set, used only for choosing hyperparameters.
    # The test set is not used until the final evaluation.
    idx = np.arange(len(train_y))
    fit_idx, val_idx = train_test_split(idx, test_size=0.2, stratify=train_y, random_state=SEED)
    fit_x = [train_x[i] for i in fit_idx]
    val_x = [train_x[i] for i in val_idx]
    return fit_x, train_y[fit_idx], val_x, train_y[val_idx]
