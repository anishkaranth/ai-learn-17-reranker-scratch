"""Tokenizer with a tiny stopword list and a crude suffix stemmer (shared by sparse and dense)."""
import re

STOP = set("""a an the and or of to in on for with is are be by at as it its this that from your you my me i we
can do does how what when my our after before over into about not no over each per up so if then than""".split())
_TOK = re.compile(r"[a-z0-9]+(?:[-+][a-z0-9]+)*")


def stem(w: str) -> str:
    for suf in ("ings", "ing", "edly", "ed", "es", "s", "ly"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def tokenize(text: str):
    return [stem(t) for t in _TOK.findall(text.lower().replace("--", "")) if t not in STOP]
