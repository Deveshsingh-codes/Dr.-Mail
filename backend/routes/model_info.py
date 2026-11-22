from flask import Blueprint, jsonify, current_app

bp = Blueprint("model_info", __name__)

@bp.get("/api/model-info")
def model_info():
    pred = current_app.config.get("PREDICTOR")
    if pred is None or not pred.metrics:
        return jsonify(error="Model metrics unavailable. Run training."), 503
    return jsonify(pred.metrics)
