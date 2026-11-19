"""Separate CPU diagnostic adapter; frozen checkpoints and loader remain untouched.

For the inspected scalar affine MAF, invert base z=0,+/-1 to recover its
conditional Gaussian, then restrict it to the frozen BoxUniform support.
Validation against archived NLLs/draws is mandatory before maps are generated.
"""
import os,sys,json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import repository_root, research_root, archive_root, diagnostic_artifacts, output_dir, require_opt_in
sys.dont_write_bytecode=True
os.environ['OMP_NUM_THREADS']='2'
import numpy as np
import torch
from scipy.stats import truncnorm
from inspect_checkpoint import load_cpu
torch.set_num_threads(2)
O=diagnostic_artifacts();R=archive_root()
REG=json.loads((R/'out/final_six/model_registry_v1.json').read_text())
class Posterior:
    def __init__(self,key):
        require_opt_in("load a saved specialist/combined NPE checkpoint")
        self.key=key;self.model=load_cpu(R/REG[key]/'model.pkl')
        self.d=self.model['density_estimator'].cpu().eval();self.net=self.d.net
        self.sc=self.model['scaler'];self.names=self.model['feature_names']
        self.h0=67.74 if key.startswith('tng') else 68.1
        transforms=self.net._transform._transforms
        assert len(transforms)==11
        assert type(self.net._distribution).__name__=='StandardNormal'
        assert type(self.net._embedding_net).__name__=='Identity'
        assert type(transforms[0]).__name__=='PointwiseAffineTransform'
        for t in transforms[1:]:
            assert type(t).__name__ in ('MaskedAffineAutoregressiveTransform','RandomPermutation')
            if hasattr(t,'features'):assert t.features==1
        base=self.model['posterior'].prior.base_dist
        assert np.array_equal(base.low.detach().numpy(),[11.5]) and np.array_equal(base.high.detach().numpy(),[14.5])
    def scale(self,raw,lg_float32=False):
        # LG saved helper returns float32 and sklearn 1.9 casts its stats;
        # saved heldout path transforms float64 before torch casts to float32.
        if lg_float32:
            x=np.asarray(raw,np.float32).copy();x-=self.sc.mean_.astype(np.float32);x/=self.sc.scale_.astype(np.float32)
        else:x=((np.asarray(raw,float)-self.sc.mean_)/self.sc.scale_).astype(np.float32)
        return x
    def raw_physical(self,physical):
        raw=np.asarray(physical,float)[:,:len(self.names)].copy()
        raw[:,1]-=self.h0*raw[:,0]/1000
        return raw
    def normal(self,raw,lg_float32=False,check=False):
        x=self.scale(raw,lg_float32);mus=[];sds=[];errors=[]
        with torch.no_grad():
            for start in range(0,len(x),1024):
                cx=torch.tensor(x[start:start+1024]);n=len(cx)
                def inv(z):return self.net._transform.inverse(torch.full((n,1),float(z)),context=cx)[0][:,0]
                mu=inv(0);sd=(inv(1)-inv(-1))/2
                assert torch.all(sd>0)
                if check:
                    errors.append(float(torch.max(torch.abs(inv(2.3)-(mu+2.3*sd)))))
                mus.append(mu.numpy());sds.append(sd.numpy())
        if check:assert max(errors)<2e-5,errors
        return np.concatenate(mus).astype(float),np.concatenate(sds).astype(float)
    def quantiles(self,physical,probs=(.025,.16,.5,.84,.975)):
        raw=self.raw_physical(physical);mu,sd=self.normal(raw,lg_float32=True)
        a=(11.5-mu)/sd;b=(14.5-mu)/sd
        return truncnorm.ppf(np.asarray(probs)[:,None],a,b,loc=mu,scale=sd).T,mu,sd
    def validation_nll(self,raw,theta):
        x=self.scale(raw);loss=[]
        with torch.no_grad():
            for start in range(0,len(x),1024):
                y=torch.tensor(np.asarray(theta[start:start+1024,None]),dtype=torch.float32)
                cx=torch.tensor(x[start:start+1024]);loss.extend((-self.net.log_prob(y,context=cx)).numpy().tolist())
        return float(np.mean(loss))
