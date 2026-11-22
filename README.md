# MailGuard – Intelligent Email Spam Classifier
A Flask web app that classifies emails as SPAM or HAM using scikit-learn `TfidfVectorizer` + `MultinomialNB`, with a Neo-Brutalist frontend.

**Problem:** spam wastes time and carries phishing risk. **Objective:** a real trained model behind a real API and UI; no keyword rules, no external AI APIs.

## Features
Analyzer with validation, probabilities, loading/error states · real metrics + confusion matrix from the backend · local-only history (60-char previews in localStorage).

## Stack
Python, Flask, scikit-learn, Joblib, HTML/CSS/JS (no frameworks).

## ML workflow
1. Download Apache SpamAssassin Public Corpus (`easy_ham_2`, `hard_ham`, `spam_2` (2003), `spam_2` (2005)) from https://spamassassin.apache.org/old/publiccorpus/ – check the site for terms.
2. Parse emails (subject + text parts), strip HTML, normalise whitespace (URLs/numbers/symbols kept), drop empty and duplicate messages.
3. Stratified 80/20 split (`random_state=42`).
4. Pipeline: `TfidfVectorizer(max_features=20000, stop_words="english", ngram_range=(1,2), min_df=2, max_df=0.9, sublinear_tf=True)` → `MultinomialNB(alpha=0.1, fit_prior=False)`. The vectorizer is fitted on training data only. `fit_prior=False` (uniform prior) mitigates class imbalance.
5. Evaluate on the test set; save `models/spam_classifier.joblib` and `models/metrics.json`. All numbers shown in the UI come from that file.

## Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd backend
python -m ml.train      # downloads data, trains, writes real metrics
python app.py           # http://127.0.0.1:5000
```
## Test
`pytest tests` (model tests use a tiny fixture model to check the plumbing; they say nothing about real-world accuracy).

## API
`POST /api/predict` `{"message": "..."}` · `GET /api/health` · `GET /api/model-info`.
No `/api/history`: history is browser-local only.

## Structure
`backend/{app.py,config.py,routes/,ml/,models/,data/}` · `frontend/{index.html,css/,js/}` · `tests/`

## Limitations
Trained on 2002–2005 mail; modern spam differs. Naive Bayes assumes term independence, and probabilities tend to be overconfident. Not a security product.
