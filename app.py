# app.py
import os
import json
import numpy as np
from flask import Flask, request, jsonify, render_template

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# ---- CONFIG (must match notebook) ----
MAX_WORDS = 50000   # tumhare notebook ka MAX_WORDS
MAX_LEN = 200       # tumhare notebook ka MAX_LEN

MODEL_PATH = os.path.join("model", "fake_jobs_cnn.keras")
TOKENIZER_PATH = os.path.join("model", "tokenizer.json")

app = Flask(__name__, template_folder="templates", static_folder="static")

# ---- Load model + tokenizer once ----
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(f"Tokenizer file not found at {TOKENIZER_PATH}")

    model = load_model(MODEL_PATH)

    with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
        tok_data = f.read()
    tokenizer = tokenizer_from_json(tok_data)

    return model, tokenizer

model, tokenizer = load_artifacts()
# --------------------------------------


def preprocess_text(text: str):
    text = (text or "").strip()
    if not text:
        return None

    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(
        seq,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post",
    )
    return padded


@app.route("/", methods=["GET"])
def home():
    # Simple page with single textarea
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Accept JSON or form-data
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        text = payload.get("text", "")
    else:
        text = request.form.get("text", "")

    X = preprocess_text(text)
    if X is None:
        return jsonify({"ok": False, "error": "Text field is required"}), 400

    # 🔮 Model prediction
    # Assume sigmoid output: probability of class '1' (Fake)
    proba_fake = float(model.predict(X)[0][0])

    # Client mapping: 0 = Real, 1 = Fake
    label_id = 1 if proba_fake >= 0.5 else 0
    label_text = "Fake" if label_id == 1 else "Real"

    resp = {
        "ok": True,
        "label": label_text,          # "Fake" / "Real"
        "label_id": label_id,         # 0 or 1 (client mapping)
        "score_fake": proba_fake,     # probability between 0–1
        "threshold": 0.5,
        "input_text": text,
    }
    return jsonify(resp), 200


if __name__ == "__main__":
    # Local dev ke liye
    app.run(debug=True)
