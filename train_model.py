import pandas as pd, numpy as np, json, joblib, lightgbm as lgb, sys
sys.path.insert(0,'app')
from features import build, FEATURES, CAT
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
import os; os.makedirs('out',exist_ok=True)
D='data/'
pr=pd.read_csv(D+'products.csv'); cu=pd.read_csv(D+'customers.csv')
tr=pd.read_csv(D+'train.csv').drop_duplicates('order_id').reset_index(drop=True)   # de-dup partner feed
X=build(tr,pr,cu); y=X.returned.values
cut=pd.Timestamp('2026-04-01'); a=(X.t<cut).values; b=~a
params=dict(n_estimators=300,learning_rate=0.03,num_leaves=15,min_child_samples=40,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,reg_lambda=5,verbose=-1,random_state=0)
m=lgb.LGBMClassifier(**params).fit(X.loc[a,FEATURES],y[a])
p=m.predict_proba(X.loc[b,FEATURES])[:,1]; yb=y[b]
print('val n',b.sum(),'base',round(yb.mean(),4))
print('AUC',round(roc_auc_score(yb,p),4),'PR-AUC',round(average_precision_score(yb,p),4),'Brier',round(brier_score_loss(yb,p),4),'Brier base',round(brier_score_loss(yb,np.full_like(p,y[a].mean())),4))
print('acc @0.5',round(((p>0.5)==yb).mean(),4),' all-zero',round(1-yb.mean(),4))
imp=pd.Series(m.booster_.feature_importance('gain'),FEATURES).sort_values(ascending=False); print((imp/imp.sum()).round(3).head(12))
# precision at top-k
o=np.argsort(-p)
for k in [0.05,0.1,0.2,0.3]:
    n=int(len(p)*k); print(f'top {k:.0%}: precision {yb[o[:n]].mean():.3f} recall {yb[o[:n]].sum()/yb.sum():.3f}')
# calibration
q=pd.qcut(p,5,duplicates='drop'); print(pd.DataFrame({'p':p,'y':yb}).groupby(q).mean().round(3))
# simple LR baseline
Xd=pd.get_dummies(X[FEATURES],columns=CAT).astype(float); Xd=(Xd-Xd[a].mean())/Xd[a].std().replace(0,1)
lr=LogisticRegression(C=0.3,max_iter=2000).fit(Xd[a].fillna(0),y[a]); pl=lr.predict_proba(Xd[b].fillna(0))[:,1]
print('LR AUC',round(roc_auc_score(yb,pl),4),'PR',round(average_precision_score(yb,pl),4))
# leaky comparison
Xl=X.copy(); Xl['svc']=tr.last_service_event_type.astype('category').values; Xl['pk']=tr.pickup_scheduled_at.notna().astype(int).values
ml=lgb.LGBMClassifier(**params).fit(Xl.loc[a,FEATURES+['svc','pk']],y[a]); pll=ml.predict_proba(Xl.loc[b,FEATURES+['svc','pk']])[:,1]
print('LEAKY AUC',round(roc_auc_score(yb,pll),4),'acc',round(((pll>0.5)==yb).mean(),4))
pd.DataFrame({'p':p,'y':yb,'shield':X.loc[b,'shield_member'].astype(str).values,'value':tr.loc[b,'order_value_inr'].values,'pay':X.loc[b,'payment_mode'].astype(str).values}).to_pickle('out/val.pkl')
