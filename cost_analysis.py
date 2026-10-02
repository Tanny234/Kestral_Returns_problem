import os; os.makedirs('out',exist_ok=True)
import pandas as pd, numpy as np, sys, lightgbm as lgb, warnings, json; warnings.filterwarnings('ignore')
sys.path.insert(0,'app'); sys.path.insert(0,'.')
from experiments_lib import *
cut=pd.Timestamp('2026-04-01'); a=(X.t<cut).values; b=~a; yb=y[b]
m1=lgb.LGBMClassifier(**P1).fit(X.loc[a,small],y[a]); p1=m1.predict_proba(X.loc[b,small])[:,1]
mu,sd=Z[a].mean(),Z[a].std().replace(0,1); lr=LogisticRegression(C=0.3,max_iter=3000).fit((Z[a]-mu)/sd,y[a]); p2=lr.predict_proba((Z[b]-mu)/sd)[:,1]
p=0.5*p1+0.5*p2
val=tr.loc[b,'order_value_inr'].values; val=np.where(val>1e6,val/100,val)
shield=(X.loc[b,'shield_member']=='Y').values
RET=1150; CALL=45; PREV=0.35; CANC=0.12
print('validation orders',b.sum(),'returns',yb.sum(),'return cost if nothing done Rs',int(yb.sum()*RET))
print('mean order value',val.mean().round(0))
o=np.argsort(-p); res=[]
for k in [0.05,0.1,0.15,0.2,0.3,0.4,0.5,1.0]:
    n=int(len(p)*k); idx=o[:n]; r=yb[idx].sum()
    call_net=PREV*RET*r-CALL*n
    out={'top':k,'flagged':n,'returns_caught':int(r),'precision':round(r/n,3),'recall':round(r/yb.sum(),3),'CALL_net_Rs':int(call_net)}
    for mg in (0.15,0.25,0.35):
        # hold: 12% cancel regardless; cancelled returners save RET; cancelled good orders lose margin
        cancelled_ret=CANC*r; cancelled_good=CANC*(n-r)
        lost=(val[idx]*(1-yb[idx])).sum()*CANC*mg
        out[f'HOLD_net_m{int(mg*100)}']=int(cancelled_ret*RET-lost)
    res.append(out)
R=pd.DataFrame(res); print(R.to_string(index=False))
# best call threshold by probability
for th in [0.08,0.11,0.15,0.2,0.25]:
    idx=p>=th; n=idx.sum(); r=yb[idx].sum(); print(f'call if p>={th}: n={n} caught={r} net Rs {int(PREV*RET*r-CALL*n)}')
# call everyone
print('call everyone net Rs',int(PREV*RET*yb.sum()-CALL*len(yb)))
# shield effect
for nm,msk in [('shield',shield),('non-shield',~shield)]:
    idx=(p>=0.15)&msk; print(nm,'called',idx.sum(),'rate in group',yb[msk].mean().round(3),'return rate among called',yb[idx].mean().round(3) if idx.sum() else None)
print('share of called that are shield', shield[p>=0.15].mean().round(3))
# accuracy at best threshold
for th in [0.3,0.4,0.5]: print('acc @',th, round(((p>=th)==yb).mean(),4))
# calibration of blend
q=pd.qcut(p,5); print(pd.DataFrame({'p':p,'y':yb}).groupby(q).mean().round(3))
R.to_csv('out/cost_table.csv',index=False)
