# Runs every experiment in the report and saves tables/figures to results/.
#   python src/main.py
# The first run takes roughly 10 minutes because of the gzip distances (cached after that).
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.svm import LinearSVC

import knn
import rules
from data import ROOT, SEED, load_r8, train_val_split
from ncd import ncd_matrix

RESULTS = os.path.join(ROOT, "results")
K_VALUES = [1, 2, 3, 5, 7, 9, 15]
TAU_VALUES = [0.0, 0.6, 0.7, 0.8, 0.9, 1.0]


def macro_f1(y, p):
    return f1_score(y, p, average="macro")


def hybrid_predict(D, train_y, texts, k, tau):
    """My extension: weighted gzip-kNN, but if the winning label has less than tau of
    the vote (the neighbours disagree) and a keyword rule fires, use the rule's label."""
    preds, shares = knn.predict(D, train_y, k, weighted=True)
    preds = preds.copy()
    n_backoff = 0
    for i in range(len(preds)):
        if shares[i] < tau:
            label, fired = rules.predict_one(texts[i])
            if fired:
                preds[i] = label
                n_backoff += 1
    return preds, n_backoff / len(preds)


def tfidf_model(kind, C):
    vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2)
    clf = LogisticRegression(C=C, max_iter=2000) if kind == "lr" else LinearSVC(C=C)
    return vec, clf


def tune_tfidf(kind, Cs, fit_x, fit_y, val_x, val_y):
    best_C, best_f1 = None, -1
    for C in Cs:
        vec, clf = tfidf_model(kind, C)
        clf.fit(vec.fit_transform(fit_x), fit_y)
        f1 = macro_f1(val_y, clf.predict(vec.transform(val_x)))
        if f1 > best_f1:
            best_C, best_f1 = C, f1
    return best_C


def main():
    os.makedirs(os.path.join(RESULTS, "figures"), exist_ok=True)
    train_x, train_y, test_x, test_y = load_r8()
    fit_x, fit_y, val_x, val_y = train_val_split(train_x, train_y)
    classes = sorted(set(train_y))
    print(f"train {len(train_x)}, test {len(test_x)}, fit {len(fit_x)}, val {len(val_x)}")

    D_val = ncd_matrix(val_x, fit_x, "val_x_fit")      # for tuning
    D_test = ncd_matrix(test_x, train_x, "test_x_train")  # for the final results

    # ---------- 1. reproduce the paper (k=2, full training set)
    top2 = knn.paper_top2_accuracy(D_test, train_y, test_y)
    fair = accuracy_score(test_y, knn.predict(D_test, train_y, 2)[0])
    nb = knn.nearest(D_test, 2)
    tie_rate = float(np.mean(train_y[nb[:, 0]] != train_y[nb[:, 1]]))
    repro = pd.DataFrame([
        {"setting": "gzip kNN k=2, paper's scoring", "ours": top2, "published": 0.954},
        {"setting": "gzip kNN k=2, fair tie-break", "ours": fair, "published": 0.915},
    ])
    repro.to_csv(os.path.join(RESULTS, "reproduction.csv"), index=False)
    print(repro.round(3).to_string(index=False))
    print(f"neighbours disagree on {tie_rate:.1%} of test documents")

    # ---------- 2. tune on the validation set
    # TF-IDF baselines: choose C
    C_lr = tune_tfidf("lr", [1, 10, 100], fit_x, fit_y, val_x, val_y)
    C_svm = tune_tfidf("svm", [0.1, 1, 10], fit_x, fit_y, val_x, val_y)

    # gzip kNN: choose k (tau = 0 means no back-off), then k and tau together for the extension
    grid = []
    for k in K_VALUES:
        for tau in TAU_VALUES:
            p, rate = hybrid_predict(D_val, fit_y, val_x, k, tau)
            grid.append({"k": k, "tau": tau, "val_macro_f1": macro_f1(val_y, p), "backoff_rate": rate})
    grid = pd.DataFrame(grid)
    grid.to_csv(os.path.join(RESULTS, "tuning.csv"), index=False)
    no_backoff = grid[grid.tau == 0]
    k_A = int(no_backoff.loc[no_backoff.val_macro_f1.idxmax(), "k"])
    best = grid.loc[grid.val_macro_f1.idxmax()]
    k_AB, tau_AB = int(best.k), float(best.tau)
    print(f"chosen: C_lr={C_lr}, C_svm={C_svm}, k (weighted only)={k_A}, extension k={k_AB} tau={tau_AB}")
    with open(os.path.join(RESULTS, "best_params.json"), "w") as f:
        json.dump({"k": k_AB, "tau": tau_AB, "C_lr": C_lr, "C_svm": C_svm, "k_weighted_only": k_A}, f, indent=2)

    # ---------- 3. final results on the test set (models use the full training set)
    preds = {}
    majority = pd.Series(train_y).mode()[0]
    preds["Majority class"] = np.full(len(test_y), majority)
    rng = np.random.default_rng(SEED)
    preds["Random (class frequencies)"] = rng.choice(train_y, len(test_y))
    preds["Keyword rules"] = np.array(rules.predict(test_x))
    for kind, name, C in [("lr", "TF-IDF + LogReg", C_lr), ("svm", "TF-IDF + LinearSVM", C_svm)]:
        vec, clf = tfidf_model(kind, C)
        clf.fit(vec.fit_transform(train_x), train_y)
        preds[name] = clf.predict(vec.transform(test_x))
    preds["gzip kNN k=2 (paper, fair)"] = knn.predict(D_test, train_y, 2)[0]
    preds[f"gzip kNN weighted, k={k_A}"] = knn.predict(D_test, train_y, k_A, weighted=True)[0]
    preds["gzip kNN k=2 + rule back-off"] = hybrid_predict(D_test, train_y, test_x, 2, tau_AB)[0]
    ext_pred, ext_rate = hybrid_predict(D_test, train_y, test_x, k_AB, tau_AB)
    preds[f"Extension: weighted k={k_AB} + back-off"] = ext_pred

    table = pd.DataFrame([{"method": m, "test_accuracy": accuracy_score(test_y, p),
                           "test_macro_f1": macro_f1(test_y, p)} for m, p in preds.items()])
    table.to_csv(os.path.join(RESULTS, "main_results.csv"), index=False)
    print(table.round(3).to_string(index=False))
    print(f"extension hands {ext_rate:.1%} of test documents to the rules")

    per_class = pd.DataFrame({m: f1_score(test_y, preds[m], labels=classes, average=None)
                              for m in ["Keyword rules", "gzip kNN k=2 (paper, fair)",
                                        f"Extension: weighted k={k_AB} + back-off", "TF-IDF + LinearSVM"]},
                             index=classes)
    per_class["n_test"] = [int((test_y == c).sum()) for c in classes]
    per_class.to_csv(os.path.join(RESULTS, "per_class_f1.csv"))
    print(per_class.round(3).to_string())

    # ---------- 4. figures
    plt.figure(figsize=(4, 3))
    for tau, style in [(0.0, "o-"), (tau_AB, "s-")]:
        d = grid[grid.tau == tau]
        label = "weighted kNN only" if tau == 0 else rf"with rule back-off ($\tau$={tau})"
        plt.plot(d.k, d.val_macro_f1, style, label=label)
    plt.xlabel("k")
    plt.ylabel("validation macro-F1")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "figures", "tuning_k.png"), dpi=200)
    plt.close()

    cm = confusion_matrix(test_y, ext_pred, labels=classes)
    plt.figure(figsize=(4, 3.5))
    plt.imshow(cm / cm.sum(axis=1, keepdims=True), cmap="Blues")
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                         color="white" if cm[i, j] > cm[i].sum() / 2 else "black")
    plt.xticks(range(len(classes)), classes, rotation=45, ha="right", fontsize=7)
    plt.yticks(range(len(classes)), classes, fontsize=7)
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "figures", "confusion.png"), dpi=200)
    plt.close()

    errors = [f"true={t} pred={p}: {x[:200]}" for x, t, p in zip(test_x, test_y, ext_pred) if t != p]
    with open(os.path.join(RESULTS, "errors.txt"), "w") as f:
        f.write(f"{len(errors)} errors\n\n" + "\n".join(errors))
    print("done, see results/")


if __name__ == "__main__":
    main()
