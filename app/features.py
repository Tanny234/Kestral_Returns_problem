"""Feature building shared by training and the service. Only fields known at dispatch."""
import numpy as np, pandas as pd

NUM = ["discount_pct","qty","value_ratio","promised_delivery_days","customer_prior_orders",
       "customer_prior_returns","prior_return_rate","list_price_inr","warranty_months",
       "tenure_days","product_age_days","hour","dow","month"]
CAT = ["sales_channel","payment_mode","family","state","is_gift","shield_member","has_address","note_type"]
FEATURES = NUM + CAT

def note_type(s):
    if s is None or (isinstance(s,float) and np.isnan(s)) or str(s).strip()=="" : return "none"
    s=str(s)
    return "landmark" if s.startswith("Landmark") else ("gate" if s.startswith("Gate code") else "other")

def fix_value(df):
    """Oct-2025 orders were stored x100 by the new gateway; recover from list price."""
    return df

def build(df, products, customers):
    d = df.copy()
    d["t"] = pd.to_datetime(d["order_placed_at"])
    d = d.merge(products, on="sku", how="left").merge(customers, on="customer_id", how="left")
    exp = d["list_price_inr"]*d["qty"]*(1-d["discount_pct"]/100)
    ratio = d["order_value_inr"]/exp
    # value as stored may be x100 (paise) in Oct 2025; ratio near 100 => rescale
    ratio = np.where(ratio>50, ratio/100, ratio)
    d["value_ratio"] = ratio
    d["prior_return_rate"] = np.where(d.customer_prior_orders>0, d.customer_prior_returns/d.customer_prior_orders.clip(lower=1), -1)
    d["tenure_days"] = (d["t"]-pd.to_datetime(d["signup_date"])).dt.days
    d["product_age_days"] = (d["t"]-pd.to_datetime(d["launch_date"])).dt.days
    d["hour"]=d.t.dt.hour; d["dow"]=d.t.dt.dayofweek; d["month"]=d.t.dt.month
    d["has_address"] = np.where(d.delivery_pincode==0,"N","Y")
    d["note_type"] = d["delivery_note"].map(note_type)
    for c in CAT: d[c]=d[c].astype("category")
    return d
