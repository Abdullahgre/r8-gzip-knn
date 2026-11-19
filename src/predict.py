# Classify new text with the final model (weighted gzip kNN + rule back-off).
# A kNN "model" is just the training data plus the chosen k and tau, which main.py
# saves in results/best_params.json. Run main.py first.
#   python src/predict.py "opec ministers agreed to cut crude output"
import json
import os
import re
import sys

from data import ROOT, load_r8
from main import hybrid_predict
from ncd import ncd_matrix

if __name__ == "__main__":
    texts = sys.argv[1:]
    if not texts:
        print('usage: python src/predict.py "some news text"')
        sys.exit()
    with open(os.path.join(ROOT, "results", "best_params.json")) as f:
        params = json.load(f)
    train_x, train_y, _, _ = load_r8()
    # same preprocessing as R8: lower case, letters only
    clean = [" ".join(re.findall(r"[a-z]+", t.lower())) for t in texts]
    D = ncd_matrix(clean, train_x)
    preds, _ = hybrid_predict(D, train_y, clean, params["k"], params["tau"])
    for t, p in zip(texts, preds):
        print(f"{p:10s} {t[:70]}")
