import numpy as np
import pandas as pd
import pytest

from lg_npe.features import add_stellar_representations
from lg_npe.final_design import (shift_stellar_features, equal_domain_weights,
    fit_training_scaler, balanced_batch_indices, assert_partition,
    calibrate_delta, posterior_summary, summary_metrics, paired_component_intervals)
from lg_npe.observation import normalized_importance_weights, gaussian_linear_observation
from lg_npe.evaluation import posterior_metrics


def test_shift_matches_physical_multiplication_and_preserves_ratio():
    f = pd.DataFrame(dict(mstar_a=[2.,5.],mstar_b=[3.,1.],id_a=[1,2],id_b=[3,4]))
    names=['log_mstar_primary','log_mstar_secondary','log_stellar_sum','log_stellar_ratio']
    before = add_stellar_representations(f)[names].to_numpy()
    f[['mstar_a','mstar_b']] *= 10**.27
    after = add_stellar_representations(f)[names].to_numpy()
    np.testing.assert_allclose(shift_stellar_features(before,names,.27),after)
    np.testing.assert_array_equal(shift_stellar_features(before,names,0),before)


def test_equal_objective_balancing_and_scaler_training_only():
    from lg_npe.controlled_training import draw_batch_from_pools
    d=np.array(['F']*1000+['T']*10)
    w=equal_domain_weights(d)
    assert w[d=='F'].sum()==pytest.approx(.5)
    assert w[d=='T'].sum()==pytest.approx(.5)
    losses=np.where(d=='F',2.,8.)
    assert np.dot(w,losses)==pytest.approx(5.)
    rng=np.random.default_rng(1)
    for _ in range(100):
        ix=balanced_batch_indices(d,64,rng)
        assert (d[ix]=='T').sum()==32
    a=balanced_batch_indices(d,64,np.random.default_rng(7))
    np.testing.assert_array_equal(a,balanced_batch_indices(d,64,np.random.default_rng(7)))
    pools=[np.flatnonzero(d==label) for label in np.unique(d)]
    np.testing.assert_array_equal(a,draw_batch_from_pools(pools,64,np.random.default_rng(7)))
    scaler=fit_training_scaler(losses[:,None],d)
    assert scaler.mean_[0]==pytest.approx(5.)
    scaler.transform([[1e9]])
    assert scaler.mean_[0]==pytest.approx(5.)


def test_host_leakage_detects_transitive_connections():
    f=pd.DataFrame(dict(host_i=[1,2,10,20],host_j=[2,3,11,21]))
    with pytest.raises(ValueError,match='host'):
        assert_partition(f,dict(train=[0,2],validation=[1],test=[3]))
    assert_partition(f,dict(train=[0,1],validation=[2],test=[3]))


def test_calibration_recovers_horizontal_shift_without_m200c():
    rng=np.random.default_rng(4); n=500
    x=rng.normal(size=(n,4)); same=rng.random(n)>.5
    f=pd.DataFrame(x,columns=['separation','vr','vt','log_stellar_sum'])
    f['same_host']=same; f['host_i']=np.arange(n); f['host_j']=np.arange(n)
    f['log_bound_subhalo_sum']=12+.1*x[:,0]-.2*x[:,1]+1.5*x[:,3]+.1*same
    t=f.copy(); t['log_bound_subhalo_sum']+=.45
    note,draws=calibrate_delta(dict(flamingo=f,tng=t),repeats=10)
    assert note['delta']==pytest.approx(.3)
    np.testing.assert_allclose(draws,.3,atol=1e-10)


def test_summary_metrics_and_pairing_match_saved_draws():
    rng=np.random.default_rng(9); y=np.arange(12.)
    s=rng.normal(y,1.,size=(500,len(y)))
    a=posterior_summary(s,y,np.arange(12),np.arange(12)//2,np.arange(12)%2==0)
    for k,v in summary_metrics(a).items():
        assert v==pytest.approx(posterior_metrics(s,y)[k])
    ci,raw=paired_component_intervals(a,a,repeats=10)
    np.testing.assert_array_equal(raw.value,0)
    b=a.iloc[::-1]
    with pytest.raises(AssertionError):paired_component_intervals(a,b,repeats=2)


def test_observation_integration_recovers_analytic_posterior():
    from scipy.stats import norm
    rng=np.random.default_rng(20)
    x=rng.normal(2,1,100000)
    w,ess=normalized_importance_weights(norm.logpdf(2,loc=x),norm.logpdf(x),norm.logpdf(x,loc=2))
    mean,var=gaussian_linear_observation(1,1,2)
    assert np.dot(w,x)==pytest.approx(mean,abs=.02)
    assert np.dot(w,(x-mean)**2)==pytest.approx(var,abs=.02)
    assert ess>10000
    assert abs(x.mean()-mean)>.9  # naive perturbation targets a different distribution


def test_optimizer_restores_best_and_uses_equal_validation_domains():
    import torch
    from lg_npe.controlled_training import optimize_density
    class Density(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.mean=torch.nn.Parameter(torch.tensor(0.))
        def log_prob(self,theta,condition):
            return -.5*(theta[:,0]-self.mean)**2
    d=Density(); x=torch.zeros(20,1); y=torch.zeros(20,1)
    labels=np.array(['F']*18+['T']*2)
    vy=torch.tensor([[0.]]*18+[[2.]]*2)
    spec=dict(batch_size=4,learning_rate=.01,max_num_epochs=3,stop_after_epochs=1)
    result=optimize_density(d,(y,x,labels),(vy,x,labels),spec,42,epoch_steps=2)
    assert result['best_validation_nll']==pytest.approx(1.)
    assert result['best_epoch']==1 and result['stopped_early']


def test_final_configs_are_frozen():
    import json
    from pathlib import Path
    from lg_npe.controlled_training import validate_final_config
    paths=list((Path(__file__).parents[1]/'configs/final').glob('*.json'))
    assert len(paths)==6
    for p in paths:
        c=json.loads(p.read_text()); validate_final_config(c)
        c['observation_noise']='invented'
        with pytest.raises(ValueError):validate_final_config(c)
