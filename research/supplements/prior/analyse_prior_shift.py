"""Training-only empirical-prior swap of saved final-six evaluation draws.
No model loading, fitting, retraining, or posterior generation.
"""
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import repository_root, research_root, archive_root, diagnostic_artifacts, output_dir, require_opt_in
import sys, json, hashlib
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, pearsonr, spearmanr
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding='utf8')
R=research_root(); A=archive_root(); O=output_dir('prior')
E=A/'out/final_six/evaluate/20261003T133254130755Z'
sys.path.insert(0,str(R/'src'))
from lg_npe.final_design import load_final_frames,frame_digest
frames=load_final_frames(R)
manifest=json.loads((E/'evaluation_manifest.json').read_text())
for domain,frame in frames.items():
    assert frame_digest(frame)==manifest['data_identity'][domain]
splitfile=A/'data/derived/final_experiment_v1/splits.npz'
with np.load(splitfile) as f: splits={k:f[k] for k in f.files}
training={d:f.iloc[splits[d+'_train']].log_m200c.to_numpy(float) for d,f in frames.items()}
pop=[];corr=[]
for domain,frame in frames.items():
    tr=frame.iloc[splits[domain+'_train']]
    pop.append(dict(domain=domain,n=len(tr),mean=tr.log_m200c.mean(),median=tr.log_m200c.median(),std=tr.log_m200c.std(),minimum=tr.log_m200c.min(),maximum=tr.log_m200c.max()))
    for feature in ['met_diff','sigma_mean','sigma_diff','kappa_mean','kappa_diff']:
        corr.append(dict(domain=domain,population='frozen_training',feature=feature,n=len(tr),pearson=pearsonr(tr[feature],tr.log_m200c).statistic,spearman=spearmanr(tr[feature],tr.log_m200c).statistic))
pd.DataFrame(pop).to_csv(O/'training_prior_summary.csv',index=False)
pd.DataFrame(corr).to_csv(O/'training_feature_correlations.csv',index=False)
print('TRAINING PRIORS',pop,flush=True)
print('CORRELATIONS',corr,flush=True)

grid=np.linspace(11.5,14.5,6001)
bandwidths=[.5,1.,2.]
densities={}; density_records=[]
with threadpool_limits(limits=2):
    for factor in bandwidths:
        densities[factor]={}
        for domain,y in training.items():
            kde=gaussian_kde(y,bw_method=lambda k: k.scotts_factor()*factor)
            norm=kde.integrate_box_1d(11.5,14.5)
            lp=kde.logpdf(grid)-np.log(norm)
            assert np.isfinite(lp).all()
            densities[factor][domain]=lp
            test=np.linspace(11.500125,14.499875,31)
            direct=kde.logpdf(test)-np.log(norm)
            err=float(np.max(np.abs(np.interp(test,grid,lp)-direct)))
            assert err<.001,err
            density_records.append(dict(domain=domain,bandwidth_factor=factor,kernel_std=float(np.sqrt(kde.covariance[0,0])),normalizer=norm,max_log_interpolation_error=err))
            print('DENSITY',density_records[-1],flush=True)
        densities[factor]['combined']=np.logaddexp(densities[factor]['flamingo'],densities[factor]['tng'])-np.log(2.)
np.savez_compressed(O/'training_prior_density_grids.npz',grid=grid,**{f'{d}_{b}':v for b,ds in densities.items() for d,v in ds.items()})
pd.DataFrame(density_records).to_csv(O/'density_estimation_details.csv',index=False)

def weighted_quantiles(sorted_samples,weights,levels):
    # Midpoint weighted CDF interpolation; median equals numpy median for uniform weights.
    cdf=np.cumsum(weights,axis=1)-.5*weights
    rows=np.arange(len(sorted_samples)); out=[]
    for q in levels:
        upper=np.argmax(cdf>=q,axis=1)
        upper=np.where(cdf[:,-1]<q,sorted_samples.shape[1]-1,upper)
        lower=np.maximum(upper-1,0)
        c0,c1=cdf[rows,lower],cdf[rows,upper]
        t=np.divide(q-c0,c1-c0,out=np.zeros_like(c0),where=c1>c0)
        t=np.clip(t,0,1)
        out.append(sorted_samples[rows,lower]+t*(sorted_samples[rows,upper]-sorted_samples[rows,lower]))
    return np.array(out)

levels=[.025,.16,.5,.84,.975]
def summarize(q,y,ess,maxweight,subsetmask):
    m=subsetmask; residual=q[2,m]-y[m]
    return dict(n=int(m.sum()),rmse=float(np.sqrt(np.mean(residual**2))),bias=float(residual.mean()),
                coverage68=float(np.mean((y[m]>=q[1,m])&(y[m]<=q[3,m]))),coverage95=float(np.mean((y[m]>=q[0,m])&(y[m]<=q[4,m]))),
                mean_width68=float(np.mean(q[3,m]-q[1,m])),ess_median=float(np.median(ess[m])),ess_p05=float(np.quantile(ess[m],.05)),
                ess_min=float(np.min(ess[m])),fraction_ess_below50=float(np.mean(ess[m]<50)),fraction_ess_below100=float(np.mean(ess[m]<100)),max_normalized_weight=float(np.max(maxweight[m])))

archived=pd.read_csv(E/'six_model_metrics.csv')
rows=[];per_case={};sources=[splitfile,E/'evaluation_manifest.json',E/'six_model_metrics.csv']
for model in ['flamingo_dynamics','flamingo_stellar_sum','tng_dynamics','tng_stellar_sum','combined_dynamics','combined_stellar_sum']:
    source=model.split('_')[0]
    destinations=['flamingo','tng'] if source=='combined' else ['tng' if source=='flamingo' else 'flamingo']
    for dest in destinations:
        path=E/f'{model}_{dest}_draws.npz';sources.append(path)
        with np.load(path) as f:
            samples=f['samples'].astype(float).T
            ix=f['pair_index'].copy()
        np.testing.assert_array_equal(ix,frames[dest].iloc[splits[dest+'_test']].pair_index)
        frame=frames[dest].set_index('pair_index').loc[ix]
        y=frame.log_m200c.to_numpy(float);same=frame.same_host.to_numpy(bool)
        mask=dict(pooled=np.ones(len(y),bool),same_host=same,different_host=~same)
        base=np.quantile(samples,levels,axis=1)
        ref=archived[(archived.model==model)&(archived.domain==dest)&(archived.subset=='pooled')].iloc[0]
        assert np.isclose(np.mean(base[2]-y),ref.bias,atol=1e-6)
        for subset,m in mask.items():
            rows.append(dict(model=model,source_prior=source,target_prior=dest,evaluation=dest,bandwidth_factor=0.,subset=subset,treatment='original',**summarize(base,y,np.full(len(y),500.),np.full(len(y),.002),m)))
        s=np.sort(samples,axis=1)
        uniform=weighted_quantiles(s,np.full_like(s,1/500),levels)
        assert np.allclose(uniform[2],base[2],atol=1e-10)
        for bandwidth in bandwidths:
            logratio=np.interp(s,grid,densities[bandwidth][dest]-densities[bandwidth][source])
            weights=np.exp(logratio-logsumexp(logratio,axis=1)[:,None])
            assert np.allclose(weights.sum(axis=1),1.)
            ess=1/np.sum(weights**2,axis=1);maxweight=weights.max(axis=1)
            q=weighted_quantiles(s,weights,levels)
            for subset,m in mask.items():
                row=dict(model=model,source_prior=source,target_prior=dest,evaluation=dest,bandwidth_factor=bandwidth,subset=subset,treatment='prior_swap',**summarize(q,y,ess,maxweight,m))
                rows.append(row)
                if subset=='pooled':print('RESULT',json.dumps(row),flush=True)
            if bandwidth==1.:
                per_case[model+'__'+dest]=pd.DataFrame(dict(pair_index=ix,truth=y,same_host=same,component=splits[dest+'_groups'][ix],original_median=base[2],swapped_median=q[2],original_q16=base[1],original_q84=base[3],swapped_q16=q[1],swapped_q84=q[3],ess=ess,max_normalized_weight=maxweight))
                per_case[model+'__'+dest].to_csv(O/f'{model}__{dest}_per_pair.csv.gz',index=False)
results=pd.DataFrame(rows);results.to_csv(O/'prior_swap_test_metrics.csv',index=False)
main=results[(results.subset=='pooled')&results.bandwidth_factor.isin([0.,1.])]
main.to_csv(O/'prior_swap_main_table.csv',index=False)

delta=.244;gauss=[]
for model in ['flamingo_dynamics','tng_dynamics','flamingo_stellar_sum','tng_stellar_sum']:
    source=model.split('_')[0];dest='tng' if source=='flamingo' else 'flamingo'
    own=archived[(archived.model==model)&(archived.domain==source)&(archived.subset=='pooled')].iloc[0]
    cross=archived[(archived.model==model)&(archived.domain==dest)&(archived.subset=='pooled')].iloc[0]
    gauss.append(dict(model=model,within_r2=own.r2,heuristic_delta_median=.244,expected_bias=(1-own.r2)*delta*(-1 if source=='flamingo' else 1),observed_bias=cross.bias))
pd.DataFrame(gauss).to_csv(O/'gaussian_heuristic.csv',index=False)

plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':300})
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for domain,color in [('flamingo','#0072B2'),('tng','#D55E00'),('combined','#009E73')]:
    axes[0].plot(grid,np.exp(densities[1.][domain]),color=color,label=domain.upper() if domain!='combined' else 'Equal-domain mixture')
axes[0].set(xlim=(11.5,14),xlabel=r'$\theta=\log_{10}(M/M_\odot)$',ylabel='Training mass density');axes[0].legend(frameon=False,fontsize=8)
special=main[~main.model.str.startswith('combined')]
names=['flamingo_dynamics','tng_dynamics','flamingo_stellar_sum','tng_stellar_sum']
for i,name in enumerate(names):
    d=special[special.model==name]
    a=d[d.treatment=='original'].bias.iloc[0];b=d[d.treatment=='prior_swap'].bias.iloc[0]
    axes[1].plot([a,b],[i,i],color='.65',zorder=1)
    axes[1].scatter(a,i,color='#D55E00',marker='s',label='Original' if i==0 else None)
    axes[1].scatter(b,i,color='#0072B2',label='Prior swapped' if i==0 else None)
axes[1].set_yticks(range(4),['F to T: dynamics','T to F: dynamics','F to T: stellar sum','T to F: stellar sum']);axes[1].invert_yaxis();axes[1].axvline(0,color='.5',ls=':');axes[1].set_xlabel('Mean posterior-median error (dex)');axes[1].legend(frameon=False,fontsize=8)
for ext in ['pdf','png']:fig.savefig(O/f'prior_swap_bias.{ext}')
plt.close(fig)

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
record={'sources':{str(p):sha(p) for p in sources},'verified_frame_digests':{d:frame_digest(f) for d,f in frames.items()},'density_estimation':density_records,'settings':{'grid_points':len(grid),'support':[11.5,14.5],'bandwidths_relative_to_Scott':bandwidths,'training_prior_only':True,'combined_prior':'0.5 p_F(theta) + 0.5 p_T(theta)','weights':'p_destination(theta) / p_source(theta), normalized separately for each evaluation pair','weight_clipping':False,'posterior_draws_per_case':500,'weighted_quantiles':'linear interpolation of midpoint weighted empirical CDF'},'limitations':['This is a sensitivity of approximate frozen posterior estimators under a KDE representation of the training mass priors, not an exact decomposition of physical causes.','A domain-prior swap keeps the source effective conditional likelihood; it does not change source feature definitions, selected population, or neural approximation error.','Combined prior replacement cannot remove simulation identity dependence of the combined effective likelihood at fixed theta.','Low ESS and finite proposal support constrain inference; KDE tail estimates and bandwidth dependence must be reported.','No model retraining, no new posterior sampling; no uncertainty from network fitting or estimated prior densities is propagated.']}
(O/'provenance.json').write_text(json.dumps(record,indent=2),encoding='utf8')
print('COMPLETE',flush=True)
