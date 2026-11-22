import json, joblib
from ml.preprocess import clean_text

class ValidationError(ValueError):
    pass

class Predictor:
    def __init__(self, model_path, metrics_path=None, max_chars=50_000):
        self.pipe = joblib.load(model_path)
        self.max_chars = max_chars
        try:
            self.metrics = json.loads(open(metrics_path).read()) if metrics_path else None
        except (OSError, ValueError):
            self.metrics = None

    def predict(self, text) -> dict:
        if not isinstance(text, str) or not text.strip():
            raise ValidationError("Email message cannot be empty.")
        if len(text) > self.max_chars:
            raise ValidationError(f"Message too long (max {self.max_chars} characters).")
        cleaned = clean_text(text)
        if not cleaned:
            raise ValidationError("Message has no readable text.")
        proba = list(self.pipe.predict_proba([cleaned])[0]); cls = list(self.pipe.classes_)
        ham, spam = float(proba[cls.index(0)]), float(proba[cls.index(1)])
        return {"label": "SPAM" if spam > ham else "HAM", "probabilities": {"spam": spam, "ham": ham},
                "note": "Model-estimated probabilities, not guarantees of safety or accuracy."}
