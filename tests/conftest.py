import sys, json, pathlib, pytest, joblib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "backend"))
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Tiny fixture corpus: only to test the plumbing, NOT a real model.
SPAM = ["win free money now click http://x.biz", "cheap viagra pills $$$ order now", "you won a free prize claim now"] * 3
HAM = ["meeting agenda for tomorrow attached", "please review the project report", "lunch on friday with the team"] * 3

@pytest.fixture(scope="session")
def model_files(tmp_path_factory):
    d = tmp_path_factory.mktemp("m")
    pipe = Pipeline([("tfidf", TfidfVectorizer()), ("nb", MultinomialNB())]).fit(SPAM + HAM, [1]*9 + [0]*9)
    joblib.dump(pipe, d / "m.joblib"); (d / "metrics.json").write_text(json.dumps({"accuracy": 1.0}))
    return d / "m.joblib", d / "metrics.json"

@pytest.fixture()
def client(model_files):
    from app import create_app
    return create_app(*model_files).test_client()
