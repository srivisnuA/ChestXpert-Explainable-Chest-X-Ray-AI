from flask import Flask, jsonify

app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "chestxpert",
        "model_loaded": False,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
