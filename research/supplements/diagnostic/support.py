"""Train-only full-feature neighborhoods and calibrated diagnostic domain models.
No NPE is fitted. Domain classifiers use frozen train/validation/test partitions.
"""
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import repository_root, research_root, archive_root, diagnostic_artifacts, output_dir, require_opt_in
import os,sys,json,pickle
sys.dont_write_bytecode=True
os.environ['OMP_NUM_THREADS']='2'
import numpy as np,pandas as pd
from scipy.spatial import cKDTree
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
D=diagnostic_artifacts();R=archive_root()
O=None
S=dict(np.load(R/'data/derived/final_experiment_v1/splits.npz'))
FEATURES=['separation','vr','vt','log_stellar_sum'];LG=np.array([765.,-109.3,82.4,np.log10(1.638e11)])
def physical(frame,domain,n):
    x=frame[FEATURES[:n]].to_numpy(float).copy();x[:,1]+=(67.74 if domain=='tng' else 68.1)*x[:,0]/1000
    return x
def equal_weights(y):
    return np.where(y==0,.5/np.sum(y==0),.5/np.sum(y==1))
def calibr_metrics(y,p):
    w=equal_weights(y);p=np.clip(p,1e-8,1-1e-8)
    return dict(auc=roc_auc_score(y,p),brier=float(np.sum(w*(p-y)**2)),log_loss=float(-np.sum(w*(y*np.log(p)+(1-y)*np.log(1-p)))))
def reliability(y,p):
    w=equal_weights(y);rows=[]
    for lo in np.arange(0,1,.1):
        use=(p>=lo)&(p<(lo+.1) if lo<.9 else p<=1)
        if use.any():rows.append(dict(low=lo,high=lo+.1,n=int(use.sum()),n_tng=int(y[use].sum()),weight=float(w[use].sum()),pred=float(np.average(p[use],weights=w[use])),observed=float(np.average(y[use],weights=w[use]))))
    return rows
class Support:
    def __init__(self,family,fit=False):
        require_opt_in("load diagnostic catalogues/classifiers or refit a domain classifier")
        self.family=family;self.n=3 if family=='dynamics' else 4
        self.frames={d:pd.read_pickle(D/f'verified_{d}_frame.pkl') for d in ['flamingo','tng']}
        self.train={d:f.iloc[S[d+'_train']] for d,f in self.frames.items()}
        self.arr={d:physical(f,d,self.n) for d,f in self.train.items()}
        self.mean=np.mean([a.mean(0) for a in self.arr.values()],0)
        self.scale=np.sqrt(np.mean([((a-self.mean)**2).mean(0) for a in self.arr.values()],0))
        self.std={d:(a-self.mean)/self.scale for d,a in self.arr.items()}
        self.tree={d:cKDTree(a) for d,a in self.std.items()}
        self.threshold={}
        for d,f in self.frames.items():
            val=(physical(f.iloc[S[d+'_validation']],d,self.n)-self.mean)/self.scale
            dist=self.tree[d].query(val,k=20,workers=2)[0][:,-1]
            self.threshold[d]=float(np.quantile(dist,.95))
        if fit:self.fit()
        else:
            saved=pickle.load(open(D/f'domain_classifier_{family}.pkl','rb'));self.clf=saved['classifier'];self.cal=saved['calibrator']
    def dataset(self,part):
        xx=[physical(f.iloc[S[d+'_'+part]],d,self.n) for d,f in self.frames.items()]
        return np.concatenate(xx),np.r_[np.zeros(len(xx[0]),int),np.ones(len(xx[1]),int)]
    def fit(self):
        global O
        O=output_dir('classifier')
        x,y=self.dataset('train');vx,vy=self.dataset('validation');tx,ty=self.dataset('test')
        self.clf=HistGradientBoostingClassifier(learning_rate=.06,max_iter=160,max_leaf_nodes=15,min_samples_leaf=30,l2_regularization=1,early_stopping=False,class_weight='balanced',random_state=261102)
        with threadpool_limits(limits=2):
            self.clf.fit(x,y)
            self.cal=LogisticRegression(C=1e6,class_weight='balanced',solver='lbfgs',max_iter=500,random_state=261102)
            self.cal.fit(logit(np.clip(self.clf.predict_proba(vx)[:,1],1e-6,1-1e-6))[:,None],vy)
            p=1-self.wF(tx);raw=self.clf.predict_proba(tx)[:,1]
        self.metrics={'family':self.family,**calibr_metrics(ty,p),'wF_LG':float(self.wF(LG[None,:self.n])[0]),'raw_wF_LG':float(1-self.clf.predict_proba(LG[None,:self.n])[0,1]),'support_threshold95':self.threshold,'n_validation_tng':int(vy.sum()),'n_test_tng':int(ty.sum()),'raw_metrics':calibr_metrics(ty,raw)}
        rel=reliability(ty,p);self.metrics['ece10']=float(sum(r['weight']*abs(r['pred']-r['observed']) for r in rel))
        pd.DataFrame(rel).to_csv(O/f'classifier_reliability_{self.family}.csv',index=False)
        pd.DataFrame(dict(domain_label=ty,pT=p,pT_raw=raw)).to_csv(O/f'classifier_test_{self.family}.csv',index=False)
        pickle.dump({'classifier':self.clf,'calibrator':self.cal,'features':FEATURES[:self.n],'coordinate':'physical vr','metrics':self.metrics},open(O/f'domain_classifier_{self.family}.pkl','wb'))
    def wF(self,x):
        raw=self.clf.predict_proba(np.asarray(x)[:,:self.n])[:,1]
        return 1-self.cal.predict_proba(logit(np.clip(raw,1e-6,1-1e-6))[:,None])[:,1]
    def evaluate(self,x):
        z=(np.asarray(x)[:,:self.n]-self.mean)/self.scale;out={'wF_classifier':self.wF(x)}
        for d in self.tree:
            radius=self.tree[d].query(z,k=20,workers=2)[0][:,-1]
            out['r20_'+d]=radius;out['supported_'+d]=radius<=self.threshold[d]
        out['common_support']=out['supported_flamingo'] & out['supported_tng']
        return out
    def neighborhoods(self):
        rows=[];neighbor_rows=[];q=(LG[:self.n]-self.mean)/self.scale
        for radius in [.25,.35,.5,.75,1.,1.5]:
            ids={d:tree.query_ball_point(q,radius) for d,tree in self.tree.items()}
            probs={d:len(ix)/len(self.train[d]) for d,ix in ids.items()}
            w=probs['flamingo']/(sum(probs.values()) or np.nan)
            for d,ix in ids.items():
                f=self.train[d].iloc[ix];dist=np.linalg.norm(self.std[d][ix]-q,axis=1)
                for host in ['pooled','same_host','different_host']:
                    mask=np.ones(len(f),bool) if host=='pooled' else (f.same_host.to_numpy() if host=='same_host' else ~f.same_host.to_numpy())
                    ff=f.loc[mask];mass=ff.log_m200c.to_numpy();groups=S[d+'_groups'][S[d+'_train'][np.asarray(ix,dtype=int)]][mask]
                    rows.append(dict(family=self.family,radius=radius,domain=d,host=host,n=len(ff),independent_components=len(np.unique(groups)),n_training=len(self.train[d]),empirical_ball_probability=probs[d],wF_equal_domain_ball=w,same_host_fraction=float(ff.same_host.mean()) if len(ff) else np.nan,mean_distance=float(np.mean(dist[mask])) if len(ff) else np.nan,mean_log_mass=float(np.mean(mass)) if len(ff) else np.nan,q16=float(np.quantile(mass,.16)) if len(ff) else np.nan,median_log_mass=float(np.median(mass)) if len(ff) else np.nan,q84=float(np.quantile(mass,.84)) if len(ff) else np.nan))
        for d,tree in self.tree.items():
            dist,ix=tree.query(q,k=50)
            for rank,(dd,ii) in enumerate(zip(dist,ix),1):
                f=self.train[d].iloc[ii];neighbor_rows.append(dict(family=self.family,domain=d,rank=rank,distance=dd,pair_index=int(f.pair_index),same_host=bool(f.same_host),log_mass=f.log_m200c))
        return rows,neighbor_rows
if __name__=='__main__':
    O=output_dir('classifier')
    assert json.loads((D/'audit_gate.json').read_text())['scientific_pilot_allowed']
    allrows=[];neighbors=[];metrics=[]
    for family in ['dynamics','stellar_sum']:
        s=Support(family,fit=True);a,b=s.neighborhoods();allrows+=a;neighbors+=b;metrics.append(s.metrics)
        print(s.metrics,flush=True)
    pd.DataFrame(allrows).to_csv(O/'lg_neighborhoods.csv',index=False)
    pd.DataFrame(neighbors).to_csv(O/'lg_nearest_neighbors.csv',index=False)
    (O/'classifier_validation.json').write_text(json.dumps(metrics,indent=2))
    # Reuse archived classifiers for the calibration evidence they actually contain.
    old_path=research_root()/'out/common_support/domain_common_support/20260925T103101Z/domain_scores_oof.csv.gz'
    if not old_path.is_file():
        print('Historical group-OOF scores are unavailable; the new classifier outputs are complete.')
        raise SystemExit(0)
    old=pd.read_csv(old_path)
    oldrows=[]
    for family in ['dynamics','stellar_sum']:
        f=old.loc[old.feature_space==family];rel=reliability(f.domain_label.to_numpy(),f.domain_score.to_numpy())
        oldrows.append({'family':family,**calibr_metrics(f.domain_label.to_numpy(),f.domain_score.to_numpy()),'ece10':sum(r['weight']*abs(r['pred']-r['observed']) for r in rel)})
        pd.DataFrame(rel).to_csv(O/f'archived_classifier_reliability_{family}.csv',index=False)
    pd.DataFrame(oldrows).to_csv(O/'archived_classifier_validation.csv',index=False)
