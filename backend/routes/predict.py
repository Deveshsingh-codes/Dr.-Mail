from flask import Blueprint, jsonify, request, current_app
from ml.predict import ValidationError

bp = Blueprint("predict", __name__)

@bp.get("/api/health")
def health():
    ok = current_app.config.get("PREDICTOR") is not None
    return jsonify(status="ok" if ok else "degraded", model_loaded=ok), (200 if ok else 503)

@bp.post("/api/predict")
def predict():
    pred = current_app.config.get("PREDICTOR")
    if pred is None:
        return jsonify(error="Model is not available. Train it first."), 503
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or "message" not in data:
        return jsonify(error="Send JSON like {\"message\": \"...\"}."), 400
    try:
        return jsonify(pred.predict(data["message"]))
    except ValidationError as e:
        return jsonify(error=str(e)), 400
    except Exception:
        current_app.logger.exception("prediction failed")
        return jsonify(error="Prediction failed. Please try again."), 500
