import json
import os
import random
import time

import numpy as np
import psycopg
import torch
from fastapi import FastAPI, Query
from psycopg.rows import dict_row
from pydantic import BaseModel, Field

from model_xl import BaselineNetXL, EarlyExitNetXL

torch.set_num_threads(1)

DB_URL = os.environ.get("DATABASE_URL")
DEFAULT_T = json.load(open("config.json"))["threshold"]
data = np.load("data.npz")
X_test = torch.tensor(data["X_test"])
y_test = data["y_test"]

ee = EarlyExitNetXL()
ee.load_state_dict(torch.load("ee_xl.pt"))
ee.eval()
bl = BaselineNetXL()
bl.load_state_dict(torch.load("bl_xl.pt"))
bl.eval()

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    source TEXT NOT NULL,
    threshold REAL NOT NULL,
    exited_early BOOLEAN NOT NULL,
    exit1_confidence REAL NOT NULL,
    ee_prediction INT NOT NULL,
    baseline_prediction INT NOT NULL,
    true_label INT,
    ee_ms REAL NOT NULL,
    baseline_ms REAL NOT NULL
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


init_db()

app = FastAPI(title="Early-Exit Neural Network API")


class Features(BaseModel):
    features: list[float] = Field(..., min_length=20, max_length=20)
    threshold: float | None = Field(None, ge=0.0, le=1.0)


def run(x, t):
    with torch.no_grad():
        t0 = time.perf_counter()
        h = ee.stem(x)
        conf, pred = torch.softmax(ee.exit1(h), 1).max(1)
        exited = bool(conf.item() >= t)
        if not exited:
            pred = ee.exit2(ee.deep(h)).argmax(1)
        ee_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        base_pred = bl(x).argmax(1)
        bl_ms = (time.perf_counter() - t0) * 1000
    return {
        "threshold": t,
        "early_exit": {
            "prediction": int(pred.item()),
            "exited_early": exited,
            "exit1_confidence": round(conf.item(), 3),
            "ms": round(ee_ms, 3),
        },
        "baseline": {"prediction": int(base_pred.item()), "ms": round(bl_ms, 3)},
    }


def log(out, source, true_label=None):
    e, b = out["early_exit"], out["baseline"]
    db(
        "INSERT INTO predictions (source, threshold, exited_early, exit1_confidence,"
        " ee_prediction, baseline_prediction, true_label, ee_ms, baseline_ms)"
        " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
        (source, out["threshold"], e["exited_early"], e["exit1_confidence"],
         e["prediction"], b["prediction"], true_label, e["ms"], b["ms"]),
    )


@app.get("/health")
def health():
    return {"status": "ok", "default_threshold": DEFAULT_T,
            "database": bool(db("SELECT 1", fetch=True))}


@app.get("/predict/random")
def predict_random(threshold: float = Query(DEFAULT_T, ge=0.0, le=1.0)):
    i = random.randrange(len(X_test))
    out = run(X_test[i:i + 1], threshold)
    out["true_label"] = int(y_test[i])
    log(out, "random", out["true_label"])
    return out


@app.post("/predict")
def predict(body: Features):
    t = DEFAULT_T if body.threshold is None else body.threshold
    x = torch.tensor([body.features], dtype=torch.float32)
    out = run(x, t)
    log(out, "api")
    return out


@app.get("/stats")
def stats():
    rows = db(
        "SELECT count(*)::int AS total,"
        " avg(exited_early::int)::float AS exit_rate,"
        " avg(ee_ms)::float AS avg_early_exit_ms,"
        " avg(baseline_ms)::float AS avg_baseline_ms,"
        " avg((ee_prediction = true_label)::int)::float AS early_exit_accuracy,"
        " avg((baseline_prediction = true_label)::int)::float AS baseline_accuracy"
        " FROM predictions", fetch=True)
    return rows[0] if rows else {"error": "database unavailable"}


@app.get("/history")
def history(limit: int = Query(20, ge=1, le=200)):
    rows = db("SELECT * FROM predictions ORDER BY id DESC LIMIT %s", (limit,), fetch=True)
    return rows if rows is not None else {"error": "database unavailable"}


@app.get("/results")
def results():
    return json.load(open("results_xl.json"))

from fastapi.responses import HTMLResponse


@app.get("/", response_class=HTMLResponse)
def home():
    return open("index.html").read()
