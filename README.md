# Early-Exit SMS Spam Screener

A two-stage spam filter built to cut latency and compute cost. A tiny fast model (TF-IDF +
logistic regression) answers messages it is confident about. Only unsure messages go to a
heavier language model (DistilBERT). Includes a REST API, a PostgreSQL log, and a web page
with a cost calculator.

Message text is never stored. The database keeps only the verdict, timing and message length.

## Results (836 held-out test messages, about 112 spam)

Timings are single-message CPU runs on a shared 2-vCPU cloud server.

| Setup | Accuracy | ms/message | Server-hours per million messages | Spam caught | Real messages flagged |
|---|---|---|---|---|---|
| Cheap model only | 98.4% | 0.37 | 0.10 | 92.9% | 5 |
| Heavy model only | 99.2% | 43.8 | 12.2 | 97.3% | 4 |
| Early-exit, threshold 0.7 | 99.0% | 1.13 | 0.31 | 94.6% | 2 |
| Early-exit, threshold 0.8 | 99.3% | 2.09 | 0.58 | 94.6% | 0 |
| Early-exit, threshold 0.9 | 99.5% | 4.04 | 1.12 | 96.4% | 0 |
| Early-exit, threshold 0.95 | 99.5% | 7.89 | 2.19 | 97.3% | 1 |
| Early-exit, threshold 0.99 | 99.2% | 34.80 | 9.67 | 97.3% | 4 |

At threshold 0.9 the average message is about 11 times faster than heavy-only, with similar
accuracy. The threshold trades speed for care: lower is faster, higher sends more messages
to the heavy model.

## Limits

- The test set is small (836 messages), and each result comes from one training run.
  Differences of a few messages are noise, so this does not show early-exit beats heavy-only.
- The data is an old English SMS collection. This is a demo, not production protection.
- Timings come from one cloud server. This version has not yet been tested on a local PC.
- With an always-on server, the saving appears as capacity (messages one server can handle),
  not a smaller bill, unless you pay per compute second.

## Run it

The trained model folder (stage2/, about 268 MB) is not in this repo. Rebuild it:

    mkdir -p sms
    curl -sSL -o sms/sms.tsv https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv
    pip install torch transformers scikit-learn pandas joblib
    python train_stage1.py
    python train_stage2.py      # about 18 minutes on 2 CPUs
    python benchmark_sms.py
    echo "DB_PASSWORD=$(openssl rand -hex 16)" > .env
    docker compose up -d --build

Then open http://localhost:8000/ (web app) or http://localhost:8000/docs (API).

## API

- POST /screen : {"text": "...", "threshold": 0.9} returns the verdict and which stage answered
- GET /stats, GET /history : read from the database
- GET /results : benchmark table
- GET /health

## Files

sms_app.py + sms.html (service), train_stage1.py, train_stage2.py, benchmark_sms.py,
sms_data.py (shared data split), Dockerfile.sms, docker-compose.yml.

Version 1 (an early-exit network on synthetic data) is also kept here: model.py, model_xl.py,
app.py, index.html, Dockerfile, docker-compose.v1.yml.
