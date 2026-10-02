import pandas as pd
def _fmt(group, X):
    r = X.iloc[0]
    up = {
      "delivery": f"Long delivery promise ({int(r.promised_delivery_days)} days) - returns rise with longer promises",
      "discount": f"Heavy discount ({int(r.discount_pct)}%) - deal-driven buyers return more",
      "history": f"Customer history: {int(r.customer_prior_returns)} return(s) in {int(r.customer_prior_orders)} past order(s)",
      "payment": f"Payment mode is {r.payment_mode} (cash on delivery returns most)",
      "shield": "Kestrel Shield member (free returns, so return more often)",
      "family": f"Product family: {r.family} (higher-return category)",
      "product": f"Higher-priced item (list Rs {int(r.list_price_inr):,})",
      "is": "Marked as a gift",
      "sales": f"Sales channel: {r.sales_channel}",
      "qty": f"Quantity {int(r.qty)}",
      "address": "No delivery address captured",
      "value": "Order value does not match list price less discount",
    }
    down = {
      "delivery": f"Short delivery promise ({int(r.promised_delivery_days)} days)",
      "discount": f"Low discount ({int(r.discount_pct)}%)",
      "history": "Clean history with Kestrel (no past returns)" if r.customer_prior_returns==0 else f"Few past returns ({int(r.customer_prior_returns)})",
      "payment": f"Payment mode is {r.payment_mode} (lower-return)",
      "shield": "Not a Shield member",
      "family": f"Product family: {r.family} (lower-return category)",
      "product": f"Lower-priced item (list Rs {int(r.list_price_inr):,})",
      "is": "Not a gift",
      "sales": f"Sales channel: {r.sales_channel}",
    }
    return up, down
def explain(contrib: pd.Series, X, n_up=3, n_down=2):
    up, down = _fmt(None, X)
    c = contrib.sort_values()
    ups = [up[g] for g, v in c[::-1].items() if v > 0.12 and g in up][:n_up]
    downs = [down[g] for g, v in c.items() if v < -0.12 and g in down][:n_down]
    return ups, downs
