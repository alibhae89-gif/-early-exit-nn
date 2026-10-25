import json
import os
import time

import joblib
import psycopg
import torch
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from psycopg.rows import dict_row
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

torch.set_num_threads(min(2, os.cpu_count() or 1))

DB_URL = os.environ.get("DATABASE_URL")
DEFAULT_T = 0.9

if not os.path.isdir("stage2"):
    raise SystemExit("The trained model folder 'stage2' is missing. Run train_stage2.py first.")

RESULTS = json.load(open("results_sms.json"))
s1 = joblib.load("stage1.joblib")
vec, clf = s1["vec"], s1["clf"]
tok = AutoTokenizer.from_pretrained("stage2")
model = AutoModelForSequenceClassification.from_pretrained("stage2").eval()

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS screenings (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    threshold REAL NOT NULL,
    stage TEXT NOT NULL,
    label TEXT NOT NULL,
    cheap_confidence REAL NOT NULL,
    ms REAL NOT NULL,
    msg_len INT NOT NULL
)
"""


def db(sql, params=(), fetch=False):
    """Run one SQL command. Returns rows (fetch), True (ok) or None (failed)."""
    if not DB_URL:
        return None
    try:
        with psycopg.connect(DB_URL, connect_timeout=3, row_factory=dict_row) as conn:
            cur = conn.execute(sql, params)
            return cur.fetchall() if fetch else True
    except Exception as e:
        print("db error:", e, flush=True)
        return None


def init_db():
    for _ in range(15):
        if db(CREATE_SQL):
            return
        time.sleep(2)


def screen_text(text, t):
    t0 = time.perf_counter()
    p = clf.predict_proba(vec.transform([text]))[0]
    conf, pred, spam_prob = float(p.max()), int(p.argmax()), float(p[1])
    stage = "cheap"
    if conf < t:
        enc = tok(text, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            probs = torch.softmax(model(**enc).logits, dim=1)[0]
        pred, spam_prob, stage = int(probs.argmax()), float(probs[1]), "heavy"
    ms = (time.perf_counter() - t0) * 1000
    return {
        "label": "spam" if pred == 1 else "not spam",
        "spam_probability": round(spam_prob, 4),
        "stage": stage,
        "cheap_confidence": round(conf, 4),
        "ms": round(ms, 2),
        "threshold": t,
    }


init_db()
screen_text("warm up message", 1.01)  # load the heavy path once at startup

app = FastAPI(title="Early-Exit SMS Spam Screener")


class Message(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    threshold: float | None = Field(None, ge=0.0, le=1.0)


@app.get("/", response_class=HTMLResponse)
def home():
    return open("sms.html").read()


@app.get("/health")
def health():
    return {"status": "ok", "default_threshold": DEFAULT_T,
            "database": bool(db("SELECT 1", fetch=True))}


@app.post("/screen")
def screen(m: Message):
    t = DEFAULT_T if m.threshold is None else m.threshold
    out = screen_text(m.text, t)
    # The message text is NOT stored, only metadata.
    db("INSERT INTO screenings (threshold, stage, label, cheap_confidence, ms, msg_len)"
       " VALUES (%s,%s,%s,%s,%s,%s)",
       (t, out["stage"], out["label"], out["cheap_confidence"], out["ms"], len(m.text)))
    return out


@app.get("/stats")
def stats():
    rows = db(
        "SELECT count(*)::int AS total,"
        " avg((stage = 'cheap')::int)::float AS answered_by_cheap,"
        " avg((label = 'spam')::int)::float AS spam_share,"
        " avg(ms)::float AS avg_ms FROM screenings", fetch=True)
    if not rows:
        return {"error": "database unavailable"}
    out = rows[0]
    out["heavy_only_ms"] = RESULTS["heavy"]["ms"]
    return out


@app.get("/history")
def history(limit: int = Query(15, ge=1, le=200)):
    rows = db("SELECT id, created_at, threshold, stage, label, cheap_confidence, ms, msg_len"
              " FROM screenings ORDER BY id DESC LIMIT %s", (limit,), fetch=True)
    return rows if rows is not None else {"error": "database unavailable"}


@app.get("/results")
def results():
    return RESULTS
