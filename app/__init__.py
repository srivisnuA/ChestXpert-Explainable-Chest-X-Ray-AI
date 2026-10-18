"""Flask application factory; independent configuration for each app instance."""
import os
import threading

from flask import Flask, jsonify


def create_app(config=None):
    application = Flask(__name__)
    application.config.update(
        MAX_CONTENT_LENGTH=16 * 1024 * 1024,
        CHESTXPERT_CHECKPOINT=os.getenv("CHESTXPERT_CHECKPOINT", "models/checkpoints/resnet18_latest.pt"),
    )
    if config:
        application.config.update(config)
    application.extensions["chestxpert"] = {"predictor": None, "lock": threading.Lock()}
    from app.api import api
    application.register_blueprint(api, url_prefix="/api")

    @application.get("/")
    def index():
        return jsonify(service="ChestXpert", description="Chest X-ray research prototype",
                       endpoints=["/api/health", "/api/ready", "/api/predict"])

    @application.errorhandler(413)
    def too_large(error):
        return jsonify(error="Upload exceeds the configured request size limit."), 413

    @application.after_request
    def private_response(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    return application
