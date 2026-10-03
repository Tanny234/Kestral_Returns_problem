# Kestrel Returns Risk

Scores an order's chance of being returned using only what is known at dispatch, and recommends a
pre-dispatch confirmation call (not a hold). No API key or internet is needed after `pip install`.

## Data (not included)
Kestrel's data is confidential (ops-policy section 10), so it is NOT in this repo. Before running,
create a `data/` folder in the project root and copy in the pack files:
`train.csv`, `test_unlabelled.csv`, `products.csv`, `customers.csv`, `sample_submission.csv`.

## Run (Python 3.10+)

**Windows (Command Prompt)**
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python make_predictions.py
cd app
uvicorn main:app --port 8000
```

**macOS / Linux**
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python make_predictions.py
cd app
uvicorn main:app --port 8000
```

`make_predictions.py` refits the model on all labelled data, then writes `predictions.csv`,
`app/model.joblib` and a local copy of the reference tables in `app/ref/`.

Open http://localhost:8000 for the screen (form pre-filled with an example; press the button to score).
API: `POST /score` (JSON order, see `app/main.py`), `GET /health`.

## Behaviour to know about
- Unknown customer ID: scored as a new, non-Shield customer, with a note on the result.
- Bad input (unknown SKU, channel, payment mode, or date): clear error message, not a crash.
- If `model.joblib` is missing, the service starts and returns a 503 saying to run `make_predictions.py`.

## Reproducing the evidence
`train_model.py`, `experiments.py`, `cost_analysis.py` and `evidence.py` (run from the project root)
reproduce every number in EVIDENCE.md. Design decisions are in DECISIONS.md and the plain-language
recommendation for Ritu is in MEMO_to_Ritu.md.

Do not publish or share this folder or the data beyond the engagement team.
