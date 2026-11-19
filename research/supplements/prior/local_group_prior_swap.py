"""Apply the training-prior density ratios to saved Local Group draws.
For input mixtures normalize within each observation before pooling, preserving
the original equal observation weights. Only20 proposals per observation make
those mixture results a finite-sample sensitivity, not a replacement inference.
"""
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import repository_root, research_root, archive_root, diagnostic_artifacts, output_dir, require_opt_in
import json,hashlib
import numpy as np
import pandas as pd
from scipy.special import logsumexp
O=output_dir('prior')
L=archive_root()/'out/local_group_final/20261004T180753744204Z'
with np.load(O/'training_prior_density_grids.npz') as d:
    grid=d['grid'];density={b:{s:d[f'{s}_{b}'] for s in ['flamingo','tng','combined']} for b in [.5,1.,2.]}
rows=[];hashes={}
def wq(x,w):
    order=np.argsort(x);a=x[order];v=w[order];v=v/v.sum()
    return np.interp([.16,.5,.84],np.cumsum(v)-.5*v,a)
for model in ['flamingo_dynamics','flamingo_stellar_sum','tng_dynamics','tng_stellar_sum','combined_dynamics','combined_stellar_sum']:
    src=model.split('_')[0];targets=['flamingo','tng'] if src=='combined' else ['tng' if src=='flamingo' else 'flamingo']
    for scenario in ['nominal','uncertainty']:
        path=L/f'{model}_lg_{scenario}_draws.npz'
        hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as f:
            x=f['theta_log10_msun'].astype(float)
            if scenario=='uncertainty':
                conditional=f['theta_by_observation'].astype(float)
                ow=f['observation_weights'].astype(float)
                np.testing.assert_allclose(ow,np.full(5000,1/5000))
                np.testing.assert_array_equal(conditional.ravel(),x)
        base=np.quantile(x,[.16,.5,.84])
        for target in targets:
            for b in [.5,1.,2.]:
                lr=density[b][target]-density[b][src]
                if scenario=='nominal':
                    lw=np.interp(x,grid,lr);w=np.exp(lw-logsumexp(lw))
                    ess=np.array([1/np.sum(w*w)])
                else:
                    lw=np.interp(conditional,grid,lr)
                    cw=np.exp(lw-logsumexp(lw,axis=0)[None,:]);ess=1/np.sum(cw*cw,axis=0)
                    w=(cw*ow[None,:]).ravel()
                q=wq(x,w)
                rows.append(dict(model=model,source_prior=src,target_prior=target,scenario=scenario,bandwidth_factor=b,
                    original_log_median=base[1],swapped_log_median=q[1],median_shift_dex=q[1]-base[1],original_width68=base[2]-base[0],swapped_width68=q[2]-q[0],
                    swapped_mass_q16=q[0],swapped_mass_median=10**q[1]/1e12,swapped_mass_q84=10**q[2]/1e12,
                    proposals_per_component=len(x) if scenario=='nominal' else 20,ess_component_median=np.median(ess),ess_component_p05=np.quantile(ess,.05),ess_component_min=ess.min()))
                rows[-1]['swapped_mass_q16']=10**q[0]/1e12
pd.DataFrame(rows).to_csv(O/'local_group_prior_swap.csv',index=False)
(O/'local_group_prior_swap_provenance.json').write_text(json.dumps({'sources':hashes,'mixture_rule':'Normalize within each20-draw observation column, retain each original1/5000 observation weight, then pool. Do not reweight flattened mixtures globally.','limitation':'Mixture updates have only20 proposals per observation; nominal100000-draw sensitivities are better resolved. Neither substitutes for a physical prior or validates either simulation.'},indent=2))
print(pd.DataFrame(rows).query('bandwidth_factor==1 and scenario=="nominal"')[['model','target_prior','median_shift_dex','swapped_mass_median','ess_component_median']].to_string(index=False))
