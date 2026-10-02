"""Kestrel Returns Risk service. Run:  uvicorn main:app --port 8000   (from the app/ folder)
No external API or key is used anywhere - the model runs locally."""
import os, sys, pathlib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
HERE = pathlib.Path(__file__).parent; sys.path.insert(0, str(HERE))
import joblib
from model import ReturnsModel  # noqa: needed for unpickling
from reasons import explain

CALL_THRESHOLD = 0.15   # call-before-dispatch cut-off (break-even is ~0.11; 0.15 leaves a safety margin)
RET, CALL, PREV = 1150, 45, 0.35

app = FastAPI(title="Kestrel Returns Risk")
products = pd.read_csv(HERE/"ref"/"products.csv"); customers = pd.read_csv(HERE/"ref"/"customers.csv")
try:
    MODEL = joblib.load(HERE/"model.joblib"); LOAD_ERR = None
except Exception as e:
    MODEL, LOAD_ERR = None, str(e)

class Order(BaseModel):
    order_placed_at: str
    customer_id: str
    sku: str
    sales_channel: str
    payment_mode: str
    discount_pct: float = 0
    qty: int = 1
    order_value_inr: float | None = None
    promised_delivery_days: int
    delivery_pincode: int = 0
    is_gift: str = "N"
    customer_prior_orders: int = 0
    customer_prior_returns: int = 0
    delivery_note: str | None = None
    order_id: str | None = None

def bad(msg, code=422): return JSONResponse({"ok": False, "error": msg}, status_code=code)

@app.get("/health")
def health(): return {"ok": MODEL is not None, "error": LOAD_ERR}

@app.post("/score")
def score(o: Order):
    if MODEL is None: return bad(f"Model file could not be loaded ({LOAD_ERR}). Run make_predictions.py first.", 503)
    if o.sku not in set(products.sku): return bad(f"Unknown product code '{o.sku}'.")
    if o.sales_channel not in ("app","web","marketplace","partner_outlet"): return bad("sales_channel must be app, web, marketplace or partner_outlet.")
    if o.payment_mode not in ("prepaid_upi","prepaid_card","cod","emi"): return bad("payment_mode must be prepaid_upi, prepaid_card, cod or emi.")
    try: pd.to_datetime(o.order_placed_at)
    except Exception: return bad("order_placed_at must look like 2026-10-01 14:30.")
    cust_known = o.customer_id in set(customers.customer_id)
    row = o.model_dump(); p = products.set_index("sku").loc[o.sku]
    if row["order_value_inr"] is None: row["order_value_inr"] = float(p.list_price_inr*o.qty*(1-o.discount_pct/100))
    cu = customers
    if not cust_known:  # new customer: treat as non-member with unknown location
        cu = pd.concat([customers, pd.DataFrame([{"customer_id":o.customer_id,"city":"Unknown","state":customers.state.mode()[0],"signup_date":o.order_placed_at[:10],"shield_member":"N"}])])
    df = pd.DataFrame([row])
    try:
        s, X, Z = MODEL.score(df, products, cu)
    except Exception as e:
        return bad(f"Could not score this record: {e}", 400)
    s = float(s[0]); ups, downs = explain(MODEL.contributions(Z.iloc[0]), X)
    flag = s >= CALL_THRESHOLD
    value = PREV*RET*s - CALL
    return {"ok": True, "order_id": o.order_id, "return_probability": round(s,4),
            "risk_band": "high" if s>=0.30 else ("elevated" if flag else "normal"),
            "recommended_action": "Confirmation call before dispatch (do not hold)" if flag else "Dispatch as normal",
            "expected_net_value_of_call_inr": round(value),
            "reasons_up": ups, "reasons_down": downs,
            "notes": (["Customer not found in records - treated as a new non-Shield customer."] if not cust_known else []) +
                     ["Probability is the model's estimate from past orders; about 1 in 3 of the flagged orders is actually returned."]}

@app.get("/", response_class=HTMLResponse)
def index(): return (HERE/"index.html").read_text()
