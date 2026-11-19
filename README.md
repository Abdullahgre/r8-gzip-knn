# gzip + kNN text classification on R8

Coursework project. I reproduce the "gzip classifier" from Jiang et al. (2023) on the
R8 Reuters dataset, compare it with some baselines, and extend it by falling back to
hand-written keyword rules when the nearest neighbours disagree.

## How to run

Python 3.10+.

```
pip install -r requirements.txt
python data/download.py
python src/main.py
```

`main.py` takes about 5-10 minutes the first time, because it has to compute gzip
distances between every pair of documents. These get saved in `results/cache/`, so it
is much faster after that.

Outputs go into `results/`:

- `reproduction.csv` - my numbers vs the paper's
- `main_results.csv` - all methods on the test set (accuracy and macro-F1)
- `tuning.csv` - validation scores for every k and tau
- `per_class_f1.csv`, `errors.txt`, `figures/` - for the error analysis

To classify your own text (after running `main.py`):

```
python src/predict.py "opec ministers agreed to cut crude output"
```

## Files

- `data/download.py` - downloads R8 from the TextGCN repo
- `src/data.py` - loading the data and the train/validation split
- `src/ncd.py` - gzip Normalised Compression Distance
- `src/knn.py` - kNN voting, plus the paper's scoring method
- `src/rules.py` - keyword rule classifier
- `src/main.py` - runs all the experiments
- `src/predict.py` - use the final model on new text
- `writeup/` - the report (LaTeX)

## References

- Jiang et al. (2023), original code: https://github.com/bazingagin/npc_gzip (I wrote my own version, no code copied)
- Data: R8 from the TextGCN repo, https://github.com/yao8839836/text_gcn
