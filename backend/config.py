from pathlib import Path
BASE = Path(__file__).parent
MODEL_PATH = BASE / "models" / "spam_classifier.joblib"
METRICS_PATH = BASE / "models" / "metrics.json"
RAW_DIR = BASE / "data" / "raw"
FRONTEND_DIR = BASE.parent / "frontend"
MAX_CHARS = 50_000
RANDOM_STATE = 42
TEST_SIZE = 0.2
# Apache SpamAssassin Public Corpus (label: 0=ham, 1=spam)
CORPUS_URL = "https://spamassassin.apache.org/old/publiccorpus/"
CORPUS_FILES = {
    "20030228_easy_ham_2.tar.bz2": 0,
    "20030228_hard_ham.tar.bz2": 0,
    "20030228_spam_2.tar.bz2": 1,
    "20050311_spam_2.tar.bz2": 1,
}
