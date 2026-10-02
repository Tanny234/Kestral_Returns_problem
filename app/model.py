"""Returns-risk model: 50/50 blend of a small gradient-boosted model and a logistic regression.
Uses only fields known at dispatch. The logistic part also supplies the plain-language reasons."""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from features import build

SMALL = ["discount_pct","promised_delivery_days","customer_prior_orders","customer_prior_returns","prior_return_rate",
         "list_price_inr","sales_channel","payment_mode","family","is_gift","shield_member"]
CATS = ["sales_channel","payment_mode","family","is_gift","shield_member","has_address"]
NUMS = ["discount_pct","qty","value_ratio","promised_delivery_days","customer_prior_orders","customer_prior_returns","prior_return_rate","list_price_inr"]
P1 = dict(n_estimators=250,learning_rate=0.03,num_leaves=7,min_child_samples=60,subsample=0.8,subsample_freq=1,
          colsample_bytree=0.8,reg_lambda=10,verbose=-1,random_state=0)
GROUPS = {"promised_delivery_days":"delivery","pd2":"delivery","discount_pct":"discount",
          "customer_prior_returns":"history","prior_return_rate":"history","prr_pos":"history","customer_prior_orders":"history",
          "list_price_inr":"product","qty":"qty","value_ratio":"value","has_address_N":"address","has_address_Y":"address"}

def lr_matrix(X, cols=None):
    Z = pd.get_dummies(X[NUMS+CATS], columns=CATS).astype(float)
    Z["pd2"] = Z.promised_delivery_days**2
    Z["prr_pos"] = (X.customer_prior_returns>0).astype(float)
    if cols is not None:
        Z = Z.reindex(columns=cols, fill_value=0.0)
    return Z

class ReturnsModel:
    def fit(self, raw, products, customers):
        X = build(raw, products, customers); y = X.returned.values
        self.cats = {c: list(X[c].cat.categories) for c in SMALL if str(X[c].dtype)=="category"}
        self.gbm = lgb.LGBMClassifier(**P1).fit(X[SMALL], y)
        Z = lr_matrix(X); self.cols = list(Z.columns); self.mu = Z.mean(); self.sd = Z.std().replace(0,1)
        self.lr = LogisticRegression(C=0.3, max_iter=3000).fit((Z-self.mu)/self.sd, y)
        self.base = float(y.mean()); return self

    def _prep(self, raw, products, customers):
        X = build(raw, products, customers)
        for c, cats in self.cats.items(): X[c] = pd.Categorical(X[c].astype(object), categories=cats)
        return X

    def score(self, raw, products, customers):
        X = self._prep(raw, products, customers)
        p1 = self.gbm.predict_proba(X[SMALL])[:,1]
        Z = (lr_matrix(X, self.cols)-self.mu)/self.sd
        p2 = self.lr.predict_proba(Z)[:,1]
        return 0.5*p1+0.5*p2, X, Z

    def contributions(self, Z_row):
        c = pd.Series(self.lr.coef_[0]*Z_row.values, index=self.cols)
        return c.groupby(lambda k: GROUPS.get(k, k.split("_")[0] if k.startswith(("sales","payment","family","is_gift","shield")) else k)).sum()
