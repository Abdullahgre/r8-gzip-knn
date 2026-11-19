# Downloads the R8 dataset (Reuters-21578, 8 classes) from the TextGCN repo (Yao et al., 2019).
# Each line of the files is "<label>\t<text>". The text is already lower-cased and
# has punctuation and numbers removed.
import os
import urllib.request

URL = "https://raw.githubusercontent.com/yao8839836/text_gcn/master/data/R8/"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "R8")

os.makedirs(OUT, exist_ok=True)
for name in ["train.txt", "test.txt"]:
    path = os.path.join(OUT, name)
    if os.path.exists(path):
        print(name, "already downloaded")
        continue
    print("downloading", name)
    urllib.request.urlretrieve(URL + name, path)
print("done, data is in", OUT)
