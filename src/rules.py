# Hand-written keyword classifier for R8.
# I picked the keywords by looking at the most common words in each class of the
# training set and keeping the ones that clearly belong to the topic. Strong words
# count 3 points, supporting words 1 point. The class with the most points wins.
import re
from collections import Counter

STRONG = {
    "earn": ["revs", "qtr", "shr", "shrs", "div", "qtly", "payout", "mths", "dividend", "quarterly", "oper"],
    "acq": ["acquire", "acquisition", "acquired", "merger", "takeover", "tender", "stake", "undisclosed", "buyout"],
    "crude": ["oil", "crude", "opec", "barrel", "barrels", "bpd", "bbl", "refinery", "refineries", "petroleum"],
    "grain": ["grain", "wheat", "corn", "crop", "crops", "harvest", "bushels", "soybeans", "maize", "barley"],
    "interest": ["prime", "lending", "discount", "rate"],
    "money-fx": ["currency", "currencies", "dollar", "yen", "intervention", "monetary", "liquidity", "shortage"],
    "ship": ["ship", "ships", "shipping", "vessel", "vessels", "port", "ports", "tanker", "seamen", "cargo", "freight"],
    "trade": ["trade", "tariffs", "tariff", "gatt", "dumping", "imports", "exports", "protectionism", "retaliation", "deficit", "surplus"],
}
SUPPORT = {
    "earn": ["net", "profit", "loss", "cts", "note", "prior", "split", "record"],
    "acq": ["offer", "bid", "buy", "buys", "purchase", "sell", "sells", "sale", "terms", "transaction", "owns", "completes"],
    "crude": ["energy", "gas", "drilling", "saudi", "postings", "wti"],
    "grain": ["agriculture", "usda", "farm", "farmers", "tonnes", "export"],
    "interest": ["interest", "rates", "fed", "bank", "points", "cut", "cuts"],
    "money-fx": ["money", "england", "bundesbank", "bills", "market"],
    "ship": ["strike", "canal", "gulf", "crew", "maritime"],
    "trade": ["bilateral", "sanctions", "retaliate", "unfair", "japan", "semiconductors"],
}
# earnings reports follow a fixed template ("shr cts vs ... net ... revs")
EARN_TEMPLATE = {"vs", "shr", "cts", "revs", "qtr", "net", "mths", "avg"}
# used to break ties: bigger classes first
ORDER = ["earn", "acq", "crude", "trade", "money-fx", "interest", "ship", "grain"]


def score(text):
    counts = Counter(re.findall(r"[a-z]+", text.lower()))
    s = {}
    for c in ORDER:
        s[c] = sum(3 * min(counts[w], 3) for w in STRONG[c]) + sum(min(counts[w], 3) for w in SUPPORT[c])
    if sum(1 for w in EARN_TEMPLATE if counts[w] > 0) >= 2:
        s["earn"] += 6
    return s


def predict_one(text):
    """Returns (label, fired). fired is False when no keyword matched at all."""
    s = score(text)
    best = max(s.values())
    if best == 0:
        return "earn", False  # nothing matched: guess the majority class
    for c in ORDER:
        if s[c] == best:
            return c, True


def predict(texts):
    return [predict_one(t)[0] for t in texts]
