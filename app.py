import os
import json
import numpy as np
from flask import Flask, request, jsonify, render_template

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# ---- CONFIG (must match training) ----
MAX_LEN = 200

MODEL_PATH = os.path.join("model", "fake_jobs_cnn.keras")
TOKENIZER_PATH = os.path.join("model", "tokenizer.json")

# confidence thresholds (tweak if you want)
HIGH_CONF_REAL = float(os.getenv("HIGH_CONF_REAL", "0.75"))  # Real + >= this -> GREEN


app = Flask(__name__, template_folder="templates", static_folder="static")


def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(f"Tokenizer file not found at: {TOKENIZER_PATH}")

    model = load_model(MODEL_PATH)

    with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
        tok_data = f.read()
    tokenizer = tokenizer_from_json(tok_data)

    return model, tokenizer


model, tokenizer = load_artifacts()


def preprocess_text(text: str):
    text = (text or "").strip()
    if not text:
        return None, ""

    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    return padded, text


def build_signal(label_id: int, conf_real: float) -> str:
    """Return traffic-light signal string: red | yellow | green"""
    if label_id == 1:   # Fake
        return "red"
    # Real
    return "green" if conf_real >= HIGH_CONF_REAL else "yellow"


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Accept JSON or form-data
    payload = request.get_json(silent=True) if request.is_json else request.form

    text = (payload.get("text") or payload.get("job_text") or "").strip()
    x, clean_text = preprocess_text(text)

    if x is None:
        return jsonify({"ok": False, "error": "Text is required"}), 400

    # Model output expected shape (1,1) with probability of class=1 (Fake)
    proba_fake = float(model.predict(x, verbose=0).ravel()[0])
    proba_fake = max(0.0, min(1.0, proba_fake))
    proba_real = 1.0 - proba_fake

    # label_id mapping: 0 = Real, 1 = Fake
    label_id = 1 if proba_fake >= 0.5 else 0
    label_text = "Fake" if label_id == 1 else "Real"

    # confidence of predicted label
    confidence = proba_fake if label_id == 1 else proba_real

    signal = build_signal(label_id, proba_real)

    resp = {
        "ok": True,
        "label": label_text,          # "Fake" / "Real"
        "label_id": label_id,         # 0 or 1
        "score_fake": proba_fake,     # P(Fake)
        "score_real": proba_real,     # P(Real)
        "confidence": confidence,     # confidence of predicted label
        "threshold": 0.5,
        "high_conf_real": HIGH_CONF_REAL,
        "signal": signal,             # red | yellow | green
        "input_text": clean_text,
    }
    return jsonify(resp), 200


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
