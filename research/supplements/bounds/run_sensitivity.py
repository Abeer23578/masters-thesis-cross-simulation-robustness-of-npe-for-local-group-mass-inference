"""CPU-only support sensitivity of frozen final-six conditional Gaussian MAFs.

No retraining, GPU use, random posterior sampling, empirical-prior replacement,
or original checkpoint modification. Analytic truncated normals at fixed inputs.
"""
import os,sys,json,hashlib,platform
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import repository_root, research_root, archive_root, diagnostic_artifacts, output_dir, require_opt_in
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['OMP_NUM_THREADS']='2'
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import scipy
from scipy.stats import truncnorm,norm,kstest
from scipy.special import ndtr
from scipy.optimize import brentq

O=output_dir('bounds')
P=diagnostic_artifacts()
HELPERS=Path(__file__).resolve().parents[1]/'diagnostic'
R=archive_root()
E=R/'out/final_six/evaluate/20261003T133254130755Z'
L=R/'out/local_group_final/20261004T180753744204Z'
sys.path.insert(0,str(HELPERS))
from posterior_cpu import Posterior,REG
import torch
torch.set_num_threads(2)
assert not torch.cuda.is_available()
sys.path.insert(0,str(research_root()/'src'))
from lg_npe.final_design import frame_digest

SCENARIOS=[
    ('original',11.5,14.5,'baseline'),
    ('expand_lower',11.0,14.5,'wider'),
    ('expand_upper',11.5,15.0,'wider'),
    ('wide',11.0,15.0,'wider'),
    ('very_wide',10.0,16.0,'wider'),
    ('upper_14',11.5,14.0,'modest_tightening'),
    ('near_population',11.54,13.87,'near_catalogue_range'),
    ('lower_12',12.0,14.5,'restrictive_stress'),
    ('upper_13',11.5,13.0,'restrictive_stress'),
    ('tight_12_13',12.0,13.0,'restrictive_stress'),
]
(O/'scenarios.json').write_text(json.dumps([dict(scenario=n,lower=a,upper=b,kind=k) for n,a,b,k in SCENARIOS],indent=2))
PROBS=np.array([.025,.16,.5,.84,.975])
sources={}
def sha(p):
    p=Path(p);h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def record(p):sources[str(p)]=sha(p)
checks=[]
def check(name,passed,**details):
    checks.append(dict(check=name,passed=bool(passed),**details))
    assert passed,(name,details)
record(R/'out/final_six/model_registry_v1.json')
record(HELPERS/'posterior_cpu.py');record(HELPERS/'inspect_checkpoint.py')
record(R/'data/derived/final_experiment_v1/splits.npz')
record(E/'evaluation_manifest.json')
splits=dict(np.load(R/'data/derived/final_experiment_v1/splits.npz'))
manifest=json.loads((E/'evaluation_manifest.json').read_text())
frames={d:pd.read_pickle(P/f'verified_{d}_frame.pkl') for d in ['flamingo','tng']}
for d,f in frames.items():
    record(P/f'verified_{d}_frame.pkl')
    check('frame_identity_'+d,frame_digest(f)==manifest['data_identity'][d])
ids=pd.read_csv(repository_root()/'results/classifier/checkpoint_identities.csv').set_index('model')
held=[];lg=[];baseline_comparison=[];ranges=[];params={}

def dist_quantiles(mu,sd,lo,hi):
    return truncnorm.ppf(PROBS[:,None],(lo-mu)/sd,(hi-mu)/sd,loc=mu,scale=sd)
def rejected(mu,sd,lo,hi):
    return ndtr((lo-mu)/sd)+ndtr((mu-hi)/sd)
def mixture_quantiles(mu,sd,w,lo,hi):
    aa=(lo-mu)/sd;bb=(hi-mu)/sd
    def cdf(y):return float(np.dot(w,truncnorm.cdf(y,aa,bb,loc=mu,scale=sd)))
    return np.array([brentq(lambda y:cdf(y)-q,lo,hi,xtol=2e-12) for q in PROBS])
def metrics(q,y,m):
    e=q[2,m]-y[m]
    return dict(n=int(m.sum()),bias=float(e.mean()),rmse=float(np.sqrt(np.mean(e*e))),
        coverage68=float(np.mean((y[m]>=q[1,m])&(y[m]<=q[3,m]))),
        coverage95=float(np.mean((y[m]>=q[0,m])&(y[m]<=q[4,m]))),
        mean_width68=float(np.mean(q[3,m]-q[1,m])),mean_width95=float(np.mean(q[4,m]-q[0,m])))

for key,rel in REG.items():
    modelpath=R/rel/'model.pkl';record(modelpath)
    check(key+'_checkpoint_sha',sources[str(modelpath)]==ids.loc[key,'sha256'])
    p=Posterior(key)
    check(key+'_cpu_device',all(t.device.type=='cpu' for t in p.d.parameters()))
    runpath=R/rel/'run_manifest.json';record(runpath)
    run=json.loads(runpath.read_text());best=run['training_summary']['best_epoch']
    for domain in run['config']['training_domains']:
        f=frames[domain].iloc[splits[domain+'_validation']]
        actual=p.validation_nll(f[p.names].to_numpy(float),f.log_m200c.to_numpy(float))
        expect=run['training_summary']['history'][best-1]['validation_by_domain'][domain]
        check(key+'_validation_nll_'+domain,abs(actual-expect)<2e-5,actual=actual,archived=expect,absolute_error=abs(actual-expect))
    train_y=np.concatenate([frames[d].iloc[splits[d+'_train']].log_m200c.to_numpy() for d in run['config']['training_domains']])
    for name,lo,hi,kind in SCENARIOS:
        ranges.append(dict(model=key,scenario=name,training_n=len(train_y),training_min=float(train_y.min()),training_max=float(train_y.max()),training_outside_box=int(np.sum((train_y<lo)|(train_y>hi)))))
    for domain in ['flamingo','tng']:
        f=frames[domain].iloc[splits[domain+'_test']]
        y=f.log_m200c.to_numpy(float);same=f.same_host.to_numpy(bool)
        mu,sd=p.normal(f[p.names].to_numpy(float),check=True)
        params[key+'__'+domain+'_mu']=mu;params[key+'__'+domain+'_sigma']=sd
        base=dist_quantiles(mu,sd,11.5,14.5)
        summarypath=E/f'{key}_{domain}_summary.csv.gz';record(summarypath)
        archived=pd.read_csv(summarypath)
        check(key+'_'+domain+'_row_identity',np.array_equal(archived.pair_index,f.pair_index))
        check(key+'_'+domain+'_truth',np.allclose(archived.truth,y,atol=1e-6))
        # Saved draws verify the recovered conditional distribution, not exact
        # finite-Monte-Carlo quantile agreement with analytic quantiles.
        path=E/f'{key}_{domain}_draws.npz';record(path)
        with np.load(path) as z:
            ix=np.unique(np.linspace(0,len(f)-1,min(64,len(f))).astype(int))
            draws=z['samples'][:,ix].astype(float)
        pit=truncnorm.cdf(draws,(11.5-mu[ix])/sd[ix],(14.5-mu[ix])/sd[ix],loc=mu[ix],scale=sd[ix])
        ks=float(kstest(pit.ravel(),'uniform').statistic)
        check(key+'_'+domain+'_archived_draw_cdf',ks<.025,ks_distance=ks,n_draws=pit.size,tolerance=.025)
        masks={'pooled':np.ones(len(f),bool),'same_host':same,'different_host':~same}
        for subset,m in masks.items():
            analytic=metrics(base,y,m);emp=np.array([archived[c] for c in ['q025','q16','median','q84','q975']])
            sampled=metrics(emp,y,m)
            baseline_comparison.append(dict(model=key,domain=domain,subset=subset,**{'analytic_'+k:v for k,v in analytic.items()},**{'saved500_'+k:v for k,v in sampled.items()}))
        per=pd.DataFrame(dict(pair_index=f.pair_index.to_numpy(),truth=y,same_host=same,mu=mu,sigma=sd,original_median=base[2],original_q16=base[1],original_q84=base[3]))
        for name,lo,hi,kind in SCENARIOS:
            q=base if name=='original' else dist_quantiles(mu,sd,lo,hi)
            reject=rejected(mu,sd,lo,hi);change=q[2]-base[2]
            check(key+'_'+domain+'_'+name+'_valid_quantiles',np.isfinite(q).all() and (np.diff(q,axis=0)>=0).all() and (q>=lo).all() and (q<=hi).all())
            if kind=='wider':check(key+'_'+domain+'_'+name+'_less_rejection',np.all(reject<=rejected(mu,sd,11.5,14.5)+1e-12))
            for subset,m in masks.items():
                bm=metrics(base,y,m);mm=metrics(q,y,m)
                held.append(dict(model=key,domain=domain,subset=subset,scenario=name,kind=kind,lower=lo,upper=hi,
                    **mm,**{'change_'+k:mm[k]-bm[k] for k in ['bias','rmse','coverage68','coverage95','mean_width68','mean_width95']},
                    median_abs_individual_median_change=float(np.median(abs(change[m]))),
                    p95_abs_individual_median_change=float(np.quantile(abs(change[m]),.95)),
                    max_abs_individual_median_change=float(np.max(abs(change[m]))),
                    fraction_individual_change_over_001=float(np.mean(abs(change[m])>.01)),
                    mean_rejected_probability=float(np.mean(reject[m])),p95_rejected_probability=float(np.quantile(reject[m],.95)),max_rejected_probability=float(np.max(reject[m])),
                    truth_outside_box=int(np.sum((y[m]<lo)|(y[m]>hi)))))
            if name in ['wide','near_population','tight_12_13']:
                per[name+'_median']=q[2];per[name+'_q16']=q[1];per[name+'_q84']=q[3]
        per.to_csv(O/f'{key}__{domain}_per_pair.csv.gz',index=False)
    for scenario in ['nominal','uncertainty']:
        path=L/f'{key}_lg_{scenario}_draws.npz';record(path)
        with np.load(path) as z:
            raw=z['features_raw'].copy();saved=z['theta_log10_msun'].astype(float)
            w=np.ones(1) if scenario=='nominal' else z['observation_weights'].copy()
            check(key+'_'+scenario+'_saved_checkpoint',str(z['model_sha256'])==sources[str(modelpath)])
        check(key+'_'+scenario+'_observation_weights',np.isclose(w.sum(),1.) and np.all(w>0))
        mu,sd=p.normal(raw,lg_float32=True,check=True)
        params[key+'__lg_'+scenario+'_mu']=mu;params[key+'__lg_'+scenario+'_sigma']=sd
        params[key+'__lg_'+scenario+'_weights']=w
        base=mixture_quantiles(mu,sd,w,11.5,14.5)
        err=float(np.max(abs(base-np.quantile(saved,PROBS))))
        check(key+'_'+scenario+'_archive_quantiles',err<.01,max_abs_quantile_error_dex=err,tolerance=.01)
        for name,lo,hi,kind in SCENARIOS:
            q=base if name=='original' else mixture_quantiles(mu,sd,w,lo,hi)
            medshift=q[2]-base[2]
            lg.append(dict(model=key,input_scenario=scenario,scenario=name,kind=kind,lower=lo,upper=hi,n_observation_inputs=len(mu),
                q025=q[0],q16=q[1],median=q[2],q84=q[3],q975=q[4],
                median_mass_1e12=10**q[2]/1e12,lower68_mass_1e12=10**q[1]/1e12,upper68_mass_1e12=10**q[3]/1e12,
                median_shift_dex=medshift,median_mass_change_percent=100*np.expm1(np.log(10)*medshift),
                width68=q[3]-q[1],width95=q[4]-q[0],width68_change_percent=100*((q[3]-q[1])/(base[3]-base[1])-1),
                width95_change_percent=100*((q[4]-q[0])/(base[4]-base[0])-1),
                mean_rejected_probability=float(np.dot(w,rejected(mu,sd,lo,hi))),max_component_rejected_probability=float(np.max(rejected(mu,sd,lo,hi)))))
    print(key,'validated; both test populations and both LG input scenarios completed',flush=True)

pd.DataFrame(held).to_csv(O/'heldout_bound_sensitivity.csv',index=False)
pd.DataFrame(lg).to_csv(O/'local_group_bound_sensitivity.csv',index=False)
pd.DataFrame(ranges).to_csv(O/'training_support_checks.csv',index=False)
pd.DataFrame(baseline_comparison).to_csv(O/'analytic_vs_saved500_baseline.csv',index=False)
pd.DataFrame(checks).to_csv(O/'validation_checks.csv',index=False)
np.savez_compressed(O/'conditional_gaussian_parameters.npz',**params)
record(Path(__file__))
(O/'provenance.json').write_text(json.dumps(dict(
    sources=sources,settings={'device':'cpu','cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],'cuda_available':torch.cuda.is_available(),'threads':2,'no_retraining':True,'no_random_sampling':True,'quantile_probs':PROBS.tolist(),'scenarios':[dict(scenario=n,lower=a,upper=b,kind=k) for n,a,b,k in SCENARIOS]},
    environment={'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'scipy':scipy.__version__},
    validation={'passed':all(c['passed'] for c in checks),'n_checks':len(checks)},
    method='Recover mu and sigma by affine inversion of frozen scalar MAF; analytically truncate and renormalize at each box. Average normalized component CDFs with saved observation weights for LG uncertainty, then invert mixture CDF.',
    caveats=['This varies support only; it does not replace the empirical training mass prior.','Widening exposes learned tails outside original support; those tails are not validated new physical information.','CPU software versions differ from the archived environment; checkpoint hashes, original validation NLLs, affine checks and archived distribution comparisons are checked.','Analytic baseline removes finite 500-draw quantile noise; changes are measured relative to this same analytic baseline.','Restrictive stress boxes may exclude training and true test masses and are not proposed priors.','The observational mixture keeps the same 5000 approximate inputs and equal weights; it is not exact observational marginalization.']),indent=2))
print('ALL CHECKS PASSED',len(checks),flush=True)
