# Kestrel Returns Risk

Scores an order for return risk using only what is known at dispatch, and recommends a pre-dispatch
confirmation call (not a hold). No API key or internet is needed after `pip install`.

## Run (clean machine, Python 3.10+)
```
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python make_predictions.py          # refits the model from data/, writes app/model.joblib and predictions.csv
cd app && uvicorn main:app --port 8000
```
Open http://localhost:8000 for the screen. Endpoint: `POST /score` (JSON order record, see `app/main.py`), `GET /health`.
If `model.joblib` is missing the service still starts and answers 503 with a clear message.

Example:
```
curl -s localhost:8000/score -H 'content-type: application/json' -d '{"order_placed_at":"2026-10-02 14:30","customer_id":"KC105196","sku":"KH-RV-02","sales_channel":"marketplace","payment_mode":"cod","discount_pct":22,"qty":1,"promised_delivery_days":8,"customer_prior_orders":4,"customer_prior_returns":2}'
```
Files: `train_model.py`, `experiments.py`, `cost_analysis.py`, `evidence.py` reproduce every number in EVIDENCE.md.
Data is confidential (policy section 10): do not publish this folder.
