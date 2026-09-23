from flask import Flask, jsonify

from app.api import api


def create_app() -> Flask:
    application = Flask(__name__)
    application.register_blueprint(api, url_prefix="/api")

    @application.get("/")
    def index():
        return jsonify(
            {
                "service": "ChestXpert",
                "description": "Explainable chest X-ray AI research prototype",
                "endpoints": ["/api/health", "/api/predict"],
            }
        )

    return application


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
