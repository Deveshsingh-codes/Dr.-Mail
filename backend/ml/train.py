"""Download SpamAssassin corpus, train TF-IDF + MultinomialNB, save pipeline + real metrics.
Run from backend/:  python -m ml.train"""
import email, json, tarfile
from email import policy
from datetime import datetime, timezone
import joblib, requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import config
from ml.preprocess import clean_text
from ml.evaluate import evaluate

def download():
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in config.CORPUS_FILES:
        dest = config.RAW_DIR / name
        if not dest.exists():
            print("Downloading", name)
            r = requests.get(config.CORPUS_URL + name, timeout=120); r.raise_for_status()
            dest.write_bytes(r.content)

def extract_text(raw: bytes) -> str:
    msg = email.message_from_bytes(raw, policy=policy.default)
    parts = [str(msg.get("subject", ""))]
    for part in msg.walk():
        if part.get_content_maintype() == "text":
            try:
                parts.append(part.get_content())
            except Exception:
                pass
    return " ".join(parts)

def load_corpus():
    texts, labels = [], []
    for name, label in config.CORPUS_FILES.items():
        with tarfile.open(config.RAW_DIR / name) as tf:
            for m in tf.getmembers():
                if m.isfile() and not m.name.endswith("cmds"):
                    texts.append(clean_text(extract_text(tf.extractfile(m).read()))); labels.append(label)
    seen, X, y = set(), [], []          # drop empty + duplicate messages
    for t, l in zip(texts, labels):
        if t and t not in seen:
            seen.add(t); X.append(t); y.append(l)
    return X, y, len(texts)

def main():
    download()
    X, y, raw_n = load_corpus()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE)
    # Vectoriser lives inside the Pipeline -> fitted on training data only (no leakage).
    # fit_prior=False: uniform class prior to counter class imbalance.
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, stop_words="english", ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True)),
        ("nb", MultinomialNB(alpha=0.1, fit_prior=False)),
    ]).fit(Xtr, ytr)
    m = evaluate(yte, pipe.predict(Xte))
    m["dataset"] = {"name": "Apache SpamAssassin Public Corpus", "source": config.CORPUS_URL, "files": list(config.CORPUS_FILES),
                    "raw_messages": raw_n, "after_cleaning": len(X), "spam": sum(y), "ham": len(y) - sum(y),
                    "train_size": len(Xtr), "test_size": len(Xte)}
    m["model"] = "TfidfVectorizer + MultinomialNB"
    m["trained_at"] = datetime.now(timezone.utc).isoformat()
    config.MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(pipe, config.MODEL_PATH)
    config.METRICS_PATH.write_text(json.dumps(m, indent=2))
    print(m["report"], m["confusion_matrix"], sep="\n")

if __name__ == "__main__":
    main()
