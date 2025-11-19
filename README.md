## Web App

This repository includes a Flask web app:

- Single text input for job description
- Uses `fake_jobs_cnn.keras` + `tokenizer.json`
- Output: **Fake Job** or **Real Job** with confidence (0–100%)

Run locally:

```bash
pip install -r requirements.txt
python app.py
# then open http://127.0.0.1:5000
