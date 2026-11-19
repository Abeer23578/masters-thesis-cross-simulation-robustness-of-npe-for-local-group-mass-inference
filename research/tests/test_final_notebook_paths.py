"""Exercise production notebook branches with explicitly synthetic model fixtures.

No sbi import, real model load, CUDA operation or scientific result is produced.
The source cells are executed unchanged; only external training/sampling is replaced.
"""
from pathlib import Path
import json
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from lg_npe.final_design import (sha256, posterior_summary, paired_component_intervals,
    shift_stellar_features)
from lg_npe.features import feature_names
from lg_npe.evaluation import posterior_metrics
from lg_npe.registry import write_json

ROOT=Path(__file__).parents[1]


def cell(path, marker):
    n=json.loads((ROOT/path).read_text(encoding='utf-8'))
    matches=[''.join(c['source']) for c in n['cells'] if c['cell_type']=='code'
             and marker in ''.join(c['source'])]
    assert len(matches)==1, f'{marker}: {len(matches)} source cells'
    return compile(matches[0],f'{path}: {marker}','exec')


def fixture_context(tmp_path):
    rng=np.random.default_rng(17)
    frames={}
    for domain in ['flamingo','tng']:
        f=pd.DataFrame(rng.normal(size=(32,3)),columns=['separation','vr','vt'])
        f['log_stellar_sum']=rng.normal(11,.1,len(f))
        f['log_mstar_primary']=f.log_stellar_sum-.2
        f['log_mstar_secondary']=f.log_stellar_sum-.4
        f['log_m200c']=12.4+.1*f.separation
        f['pair_index']=np.arange(32)
        f['same_host']=np.arange(32)%2==0
        frames[domain]=f
    splits={d:dict(train=np.arange(14),validation=np.arange(14,20),test=np.arange(20,32),
                    groups=np.arange(32)//2) for d in frames}
    output=tmp_path/'results'; output.mkdir()
    bundle=tmp_path/'bundle'; bundle.mkdir()
    np.savez(bundle/'splits.npz',synthetic=np.arange(3))
    configurations={p.stem:json.loads(p.read_text()) for p in (ROOT/'configs/final').glob('*.json')}
    def samples(posterior,X,draws,batch_size,device):
        assert draws==500 and device=='cuda'  # requested settings; no CUDA is used here
        return np.random.default_rng(9).normal(12.4+posterior, .3,(draws,len(X)))
    def paired(a,b,repeats=2000,**kwargs):
        assert repeats==2000
        return paired_component_intervals(a,b,repeats=4,**kwargs)
    return dict(PROJECT=tmp_path,OUTPUT=output,BUNDLE=bundle,frames=frames,splits=splits,
        configs=configurations,identity={'fixture':'synthetic'},STAMP='synthetic',
        Path=Path,json=json,pickle=pickle,np=np,pd=pd,plt=plt,write_json=write_json,sha256=sha256,
        feature_names=feature_names,posterior_metrics=posterior_metrics,
        posterior_summary=posterior_summary,posterior_draws=samples,
        paired_component_intervals=paired,shift_stellar_features=shift_stellar_features,
        require_saved_model_environment=lambda:None,environment_record=lambda:{'fixture':'synthetic'},
        display=lambda *args:None,audit={'source_sha256':{},'design_sha256':'synthetic'})


@pytest.mark.parametrize('name',['flamingo_dynamics','tng_stellar_sum','combined_stellar_sum'])
def test_training_notebook_passes_only_declared_partitions(tmp_path,name):
    ctx=fixture_context(tmp_path); ctx.update(MODE='train',CONFIG_NAME=name)
    c=ctx['configs'][name]; names=feature_names('dynamics',c['mass_feature'])
    def train(train,validation,config):
        assert config==c
        for actual,part in [(train,'train'),(validation,'validation')]:
            expected=[ctx['frames'][d].iloc[ctx['splits'][d][part]] for d in c['training_domains']]
            np.testing.assert_array_equal(actual[0],np.concatenate([f[names] for f in expected]))
            np.testing.assert_array_equal(actual[1],np.concatenate([f.log_m200c for f in expected]))
            assert len(actual[2])==len(actual[1])
        return None,None,None,dict(history=[dict(epoch=1,validation_nll=0)],best_epoch=1)
    ctx['train_controlled_npe']=train
    exec(cell('notebooks/final/final_six_model_experiment.ipynb','posterior,density,scaler,history='),ctx)
    run=tmp_path/c['output_root']/c['experiment_id']/'synthetic'
    m=json.loads((run/'run_manifest.json').read_text())
    assert m['config']==c and m['model_sha256']==sha256(run/'model.pkl')


def test_six_model_evaluation_notebook_writes_complete_aligned_outputs(tmp_path):
    ctx=fixture_context(tmp_path); ctx['MODE']='evaluate'; registry={}; models={}
    for i,(label,c) in enumerate(ctx['configs'].items()):
        run=tmp_path/'models'/label; run.mkdir(parents=True)
        (run/'model.pkl').write_bytes(b'synthetic fixture, not a serialized model')
        write_json(run/'run_manifest.json',dict(config=c,data_identity=ctx['identity'],
            split_sha256=sha256(ctx['BUNDLE']/'splits.npz'),model_sha256=sha256(run/'model.pkl')))
        names=feature_names('dynamics',c['mass_feature'])
        scaler=StandardScaler().fit(ctx['frames']['flamingo'][names].to_numpy(float))
        models[run/'model.pkl']=dict(posterior=i*.01,feature_names=names,scaler=scaler)
        registry[label]=str(run.relative_to(tmp_path))
    registry_path=tmp_path/'registry.json'; write_json(registry_path,registry)
    ctx.update(MODEL_REGISTRY=str(registry_path),load_model=lambda path:models[path])
    exec(cell('notebooks/final/final_six_model_experiment.ipynb','metrics=[]; summaries={};'),ctx)
    table=pd.read_csv(ctx['OUTPUT']/'six_model_metrics.csv')
    assert len(table)==24 and table.model.nunique()==6
    assert set(table.groupby('model').size())=={4}
    assert table[['rmse','bias','coverage68','coverage95']].notna().all().all()
    exec(cell('notebooks/final/final_six_model_experiment.ipynb','contrasts=['),ctx)
    paired=pd.read_csv(ctx['OUTPUT']/'paired_component_intervals.csv')
    assert len(paired[['model_A','model_B']].drop_duplicates())==7
    assert len(list(ctx['OUTPUT'].glob('*_draws.npz')))==12
    assert (ctx['OUTPUT']/'evaluation_manifest.json').is_file()
    plt.close('all')


def test_delta_evaluation_notebook_retains_every_shift_and_host_subset(tmp_path):
    ctx=fixture_context(tmp_path);ctx.update(MODE='evaluate',DELTA=.361689,
        DELTAS=np.array([-.1,0.,.361689,.6]),model_provenance={})
    runs={}; models={}
    for label in ['stellar_sum','stellar_individual']:
        run=tmp_path/'historical'/label;run.mkdir(parents=True)
        train=ctx['splits']['flamingo']['train']
        np.savez(run/'split_indices.npz',train=train)
        names=feature_names('dynamics',label)
        scaler=StandardScaler().fit(ctx['frames']['flamingo'].iloc[train][names].to_numpy(np.float32))
        models[run/'model.pkl']=dict(posterior=0.,scaler=scaler,feature_names=names)
        runs[label]=run
    calibration_path=tmp_path/'calibration.json';calibration_path.write_text('{}')
    ctx.update(RUNS=runs,load_model=lambda path:models[path],calibration_path=calibration_path)
    exec(cell('notebooks/diagnostics/stellar_delta_shift.ipynb','metrics=[]; summaries={}; support=[]'),ctx)
    table=pd.read_csv(ctx['OUTPUT']/'delta_metrics.csv')
    assert len(table)==2*4*3
    assert set(table.n)=={6,12}
    assert table.groupby(['model','delta']).size().eq(3).all()
    exec(cell('notebooks/diagnostics/stellar_delta_shift.ipynb','intervals=[]; zero='),ctx)
    assert len(list(ctx['OUTPUT'].glob('*_draws.npz')))==8
    assert (ctx['OUTPUT']/'completion_manifest.json').is_file()
    plt.close('all')
