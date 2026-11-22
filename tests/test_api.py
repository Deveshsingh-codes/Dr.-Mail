from app import create_app

def test_health(client):
    assert client.get("/api/health").get_json()["model_loaded"] is True

def test_predict_ok(client):
    r = client.post("/api/predict", json={"message": "win free money now"})
    assert r.status_code == 200 and r.get_json()["label"] in ("SPAM", "HAM")

def test_predict_errors(client):
    assert client.post("/api/predict", json={"message": ""}).status_code == 400
    assert client.post("/api/predict", data="{bad", content_type="application/json").status_code == 400
    assert client.post("/api/predict", json={"x": 1}).status_code == 400

def test_model_info(client):
    assert client.get("/api/model-info").get_json()["accuracy"] == 1.0

def test_missing_model_handled(tmp_path):
    c = create_app(tmp_path / "none.joblib").test_client()
    assert c.get("/api/health").status_code == 503
    assert c.post("/api/predict", json={"message": "hi"}).status_code == 503

def test_frontend_served(client):
    assert client.get("/").status_code == 200
