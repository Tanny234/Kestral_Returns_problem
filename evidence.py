import sys; sys.path.insert(0,'.'); sys.path.insert(0,'app')
from experiments_lib import *
from sklearn.metrics import roc_auc_score, average_precision_score
cut=pd.Timestamp('2026-04-01'); a=(X.t<cut).values; b=~a; yb=y[b]
m1=lgb.LGBMClassifier(**P1).fit(X.loc[a,small],y[a]); p1=m1.predict_proba(X.loc[b,small])[:,1]
mu,sd=Z[a].mean(),Z[a].std().replace(0,1); lr=LogisticRegression(C=0.3,max_iter=3000).fit((Z[a]-mu)/sd,y[a]); p=0.5*p1+0.5*lr.predict_proba((Z[b]-mu)/sd)[:,1]
rng=np.random.default_rng(0); n=len(p); aucs=[];nets=[];prs=[]
for _ in range(1000):
    i=rng.integers(0,n,n); 
    if yb[i].sum()==0: continue
    aucs.append(roc_auc_score(yb[i],p[i])); prs.append(average_precision_score(yb[i],p[i]))
    c=p[i]>=0.15; nets.append(0.35*1150*yb[i][c].sum()-45*c.sum())
q=lambda v:(round(np.percentile(v,2.5),3),round(np.percentile(v,97.5),3))
print('AUC',round(roc_auc_score(yb,p),3),q(aucs)); print('PR-AUC',round(average_precision_score(yb,p),3),q(prs)); print('call net Rs',q([int(x) for x in nets]))
c=p>=0.15; print('called',c.sum(),'of',n,'; returns among called',yb[c].sum(),'of',yb.sum(),'; not-called returns',yb[~c].sum())
print('false alarms (called, not returned)',(c&(yb==0)).sum(),'=> wasted calls Rs',int((c&(yb==0)).sum()*45))
print('by shield:',pd.DataFrame({'c':c,'y':yb,'s':(X.loc[b,'shield_member']=='Y').values}).groupby('s').agg(called=('c','sum'),n=('c','size'),ret=('y','sum')))
# sensitivity to call effectiveness
for eff in (0.15,0.2,0.25,0.35):
    print('effect',eff,'net Rs',int(eff*1150*yb[c].sum()-45*c.sum()))
# what a leaky model gives in accuracy, and the 95% bar
print('best achievable accuracy honest model:',max(((p>=t)==yb).mean() for t in np.linspace(0.1,0.9,81)).round(4),' all-no-return:',round(1-yb.mean(),4))
