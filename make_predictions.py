import pandas as pd, numpy as np, sys, joblib
sys.path.insert(0,'app')
from model import ReturnsModel
D='data/'; pr=pd.read_csv(D+'products.csv'); cu=pd.read_csv(D+'customers.csv')
tr=pd.read_csv(D+'train.csv').drop_duplicates('order_id'); te=pd.read_csv(D+'test_unlabelled.csv')
m=ReturnsModel().fit(tr,pr,cu); joblib.dump(m,'app/model.joblib')
p,X,Z=m.score(te,pr,cu)
sub=pd.read_csv(D+'sample_submission.csv'); out=pd.DataFrame({'order_id':te.order_id,'score':np.round(p,6)})
assert set(out.order_id)==set(sub.order_id) and len(out)==len(sub) and out.order_id.is_unique
out=sub[['order_id']].merge(out,on='order_id'); out.to_csv('predictions.csv',index=False)
print(out.score.describe().round(3)); print('mean score',out.score.mean().round(4),'train base',round(m.base,4))
print('share >=0.15',(out.score>=0.15).mean().round(3))
# drift check train-tail vs test
Xt=m._prep(tr[tr.order_placed_at>='2026-04-01'],pr,cu)
for c in ['promised_delivery_days','discount_pct','customer_prior_returns']: print(c,'val',round(Xt[c].mean(),3),'test',round(X[c].mean(),3))
for c in ['payment_mode','shield_member']: print(pd.concat([Xt[c].value_counts(normalize=True).round(3),X[c].value_counts(normalize=True).round(3)],axis=1))
print('test value ratio range',X.value_ratio.min().round(3),X.value_ratio.max().round(3))
