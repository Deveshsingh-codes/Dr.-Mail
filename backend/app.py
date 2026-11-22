import logging
from flask import Flask, jsonify, send_from_directory
from werkzeug.exceptions import HTTPException
import config
from ml.predict import Predictor
from routes import predict, model_info

def create_app(model_path=config.MODEL_PATH, metrics_path=config.METRICS_PATH):
    app = Flask(__name__, static_folder=None)
    try:
        app.config["PREDICTOR"] = Predictor(model_path, metrics_path, config.MAX_CHARS)
    except Exception:
        logging.exception("Could not load model"); app.config["PREDICTOR"] = None
    app.register_blueprint(predict.bp); app.register_blueprint(model_info.bp)

    @app.get("/")
    def index(): return send_from_directory(config.FRONTEND_DIR, "index.html")
    @app.get("/<path:p>")
    def assets(p): return send_from_directory(config.FRONTEND_DIR, p)
    @app.errorhandler(Exception)
    def err(e):
        if isinstance(e, HTTPException):
            return jsonify(error=e.name), e.code
        app.logger.exception(e); return jsonify(error="Internal server error"), 500
    return app

if __name__ == "__main__":
    create_app().run(debug=False, port=5000)  # frontend served same-origin: no CORS needed
