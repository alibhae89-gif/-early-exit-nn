# Early-Exit Neural Network

A small neural network with two exits. Easy inputs are answered at an early exit,
and hard ones continue through the deeper layers. Includes a REST API, a PostgreSQL
log of every prediction, and a web page with live statistics.

## Results (4,000 test samples, synthetic 4-class data, deep path width 1024)

| Threshold | Accuracy | Exit rate | ms/sample | Speedup |
|---|---|---|---|---|
| Baseline (full depth) | 88.5% | n/a | 0.539 | 1.00x |
| 0.5 | 82.2% | 76.7% | 0.216 | 2.49x |
| 0.6 | 85.9% | 58.0% | 0.384 | 1.40x |
| 0.7 | 87.7% | 41.2% | 0.440 | 1.22x |
| 0.8 | 88.5% | 25.5% | 0.516 | 1.04x |
| 0.9 | 88.8% | 10.7% | 0.607 | 0.89x |

## Findings

- Early-exit saves time only when the skipped layers are expensive. With a tiny network
  (width 128) the confidence check cost as much as it saved.
- Speed is traded for accuracy through the threshold. Matching baseline accuracy gave
  almost no speedup on this synthetic data.
- Timings come from CPU runs on a shared cloud server and vary between runs.

## Run it (Docker)

    echo "DB_PASSWORD=$(openssl rand -hex 16)" > .env
    docker compose up -d --build

Then open http://localhost:8000/ (web app) or http://localhost:8000/docs (API).

## API

- GET /predict/random?threshold=0.8 : classify a random test sample
- POST /predict : classify 20 feature values
- GET /stats, GET /history : read from the database
- GET /results : benchmark table
- GET /health

## Files

data.py, model.py / model_xl.py, train_*.py, final_train.py, benchmark.py,
scale_test.py (experiments), app.py + index.html (service), Dockerfile,
docker-compose.yml.

## Limitations

Synthetic data only. This version is a technique demonstration, not a production system.
