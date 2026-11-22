import pytest
from ml.predict import Predictor, ValidationError
from ml.preprocess import clean_text

def test_clean_keeps_urls_strips_html():
    assert clean_text("<p>Hi  <b>there</b> http://a.com $5</p>") == "Hi there http://a.com $5"
    assert clean_text(None) == ""

def test_predictions_and_probabilities(model_files):
    p = Predictor(*model_files)
    s, h = p.predict("free money prize click now"), p.predict("project meeting agenda")
    assert s["label"] == "SPAM" and h["label"] == "HAM"
    assert abs(sum(s["probabilities"].values()) - 1) < 1e-6

@pytest.mark.parametrize("bad", ["", "   ", None, "x" * 60000])
def test_rejects_bad(model_files, bad):
    with pytest.raises(ValidationError):
        Predictor(*model_files).predict(bad)
