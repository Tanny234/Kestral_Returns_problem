import pandas as pd, numpy as np, sys, lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'app')
from features import build, FEATURES, CAT, NUM
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
D='data/'; pr=pd.read_csv(D+'products.csv'); cu=pd.read_csv(D+'customers.csv')
tr=pd.read_csv(D+'train.csv').drop_duplicates('order_id').reset_index(drop=True)
X=build(tr,pr,cu); y=X.returned.values
LRN=["discount_pct","qty","value_ratio","promised_delivery_days","customer_prior_orders","customer_prior_returns","prior_return_rate","list_price_inr"]
def lr_mat(X):
    Z=pd.get_dummies(X[LRN+["sales_channel","payment_mode","family","is_gift","shield_member","has_address"]],columns=["sales_channel","payment_mode","family","is_gift","shield_member","has_address"]).astype(float)
    Z["pd2"]=Z.promised_delivery_days**2; Z["prr_pos"]=(X.customer_prior_returns>0).astype(float); return Z
Z=lr_mat(X)
small=["discount_pct","promised_delivery_days","customer_prior_orders","customer_prior_returns","prior_return_rate","list_price_inr","sales_channel","payment_mode","family","is_gift","shield_member"]
P1=dict(n_estimators=250,learning_rate=0.03,num_leaves=7,min_child_samples=60,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,reg_lambda=10,verbose=-1,random_state=0)
