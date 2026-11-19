import numpy as np


def nearest(D, k):
    # indices of the k closest training documents for each row of D
    return np.argsort(D, axis=1, kind="stable")[:, :k]


def paper_top2_accuracy(D, train_y, true_y):
    """The scoring used in the paper's code (k=2): if the two neighbours disagree,
    the prediction counts as correct when EITHER label is right. This uses the true
    label, so it is not something a real classifier could do."""
    nb = nearest(D, 2)
    correct = 0
    for i in range(len(true_y)):
        if true_y[i] in train_y[nb[i]]:
            correct += 1
    return correct / len(true_y)


def predict(D, train_y, k, weighted=False):
    """kNN vote. Returns the predicted labels and the winner's share of the vote.
    Ties are broken by the closest neighbour, so k=2 gives the same answer as k=1."""
    nb = nearest(D, k)
    preds, shares = [], []
    for i in range(len(D)):
        labels = train_y[nb[i]]
        if weighted:
            weights = 1.0 / (D[i, nb[i]] + 1e-3)
        else:
            weights = np.ones(k)
        votes = {}
        for lab, w in zip(labels, weights):
            votes[lab] = votes.get(lab, 0) + w
        best = max(votes.values())
        winner = [lab for lab in labels if votes[lab] == best][0]  # closest of the tied labels
        preds.append(winner)
        shares.append(best / sum(votes.values()))
    return np.array(preds, dtype=train_y.dtype), np.array(shares)
