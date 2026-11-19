#!/usr/bin/env python
"""Thin Roihu notebook launcher, including a versioned synthetic compatibility gate."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from lg_npe.final_design import sha256, require_saved_model_environment
from lg_npe.registry import write_json, environment_record


def input_hashes():
    paths=list((PROJECT/'src/lg_npe').glob('*.py'))+list((PROJECT/'configs/final').glob('*.json'))
    paths += list((PROJECT/'tests').glob('*.py'))
    paths += [Path(__file__),PROJECT/'cluster/roihu/run_final_stage.sh',
              PROJECT/'requirements-roihu.txt',
              PROJECT/'notebooks/final/final_six_model_experiment.ipynb',
              PROJECT/'notebooks/diagnostics/stellar_delta_shift.ipynb',
              PROJECT/'experiment_design.ipynb']
    return {str(p.relative_to(PROJECT)):sha256(p) for p in sorted(paths)}


def smoke(output):
    import numpy as np
    import torch
    from lg_npe.controlled_training import train_controlled_npe
    from lg_npe.inference import posterior_draws
    from lg_npe.evaluation import posterior_metrics
    subprocess.run([sys.executable,'-m','pytest','tests','-q'],cwd=PROJECT,check=True)
    rng=np.random.default_rng(42)
    results={}
    for label,dimension in [('combined_dynamics',3),('combined_stellar_sum',4)]:
        c=json.loads((PROJECT/f'configs/final/{label}.json').read_text())
        x=rng.normal(size=(128,dimension)); vx=rng.normal(size=(32,dimension))
        theta=12.5+.15*x[:,0]+rng.normal(0,.08,len(x))
        vt=12.5+.15*vx[:,0]+rng.normal(0,.08,len(vx))
        d=np.array(['flamingo']*112+['tng']*16)
        vd=np.array(['flamingo']*24+['tng']*8)
        posterior,density,scaler,history=train_controlled_npe((x,theta,d),(vx,vt,vd),c)
        samples=posterior_draws(posterior,scaler.transform(vx),500,16,'cuda')
        assert samples.shape==(500,32) and np.isfinite(samples).all()
        assert samples.min()>=11.5 and samples.max()<=14.5
        # Verify the restored checkpoint against exact full-domain validation means.
        vt_tensor = torch.tensor(vt[:, None], dtype=torch.float32, device='cuda')
        vx_tensor = torch.tensor(scaler.transform(vx), dtype=torch.float32, device='cuda')
        scores = []
        with torch.no_grad():
            for domain in np.unique(vd):
                indices = np.flatnonzero(vd == domain)
                idx = torch.as_tensor(indices, device='cuda')
                score = float(-density.log_prob(
                    vt_tensor[idx], condition=vx_tensor[idx]
                ).sum()) / len(indices)
                scores.append(score)
        loss = float(np.mean(scores))
        assert np.isclose(loss,history['best_validation_nll'],atol=1e-5)
        results[label]=dict(metrics=posterior_metrics(samples,vt),best_epoch=history['best_epoch'])
    receipt=dict(status='passed',input_sha256=input_hashes(),environment=environment_record(),results=results)
    write_json(output/'smoke_receipt.json',receipt)
    print('SMOKE RECEIPT:',output/'smoke_receipt.json')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['smoke','prepare','delta','train','evaluate'])
    p.add_argument('--config',default='flamingo_dynamics')
    p.add_argument('--receipt',type=Path)
    p.add_argument('--registry',type=Path)
    args=p.parse_args()
    if os.environ.get('LG_RUN_RESEARCH_WORKFLOW') != '1':
        raise RuntimeError('Optional workflow disabled. Set LG_RUN_RESEARCH_WORKFLOW=1 after reading the reproduction notebook.')
    os.chdir(PROJECT); os.environ['LG_PROJECT']=str(PROJECT)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output=PROJECT/'out/final_stage_jobs'/stamp
    output.mkdir(parents=True,exist_ok=False)
    if args.stage!='prepare':require_saved_model_environment()
    if args.stage=='smoke':
        smoke(output); return
    if args.stage in ['train','delta','evaluate']:
        if args.receipt is None:raise ValueError('Supply --receipt from the completed final-stage smoke')
        receipt=json.loads(args.receipt.read_text())
        if receipt['status']!='passed' or receipt['input_sha256']!=input_hashes():
            raise ValueError('Smoke receipt does not match current source/configs/notebooks')
    import papermill as pm
    notebook='notebooks/final/final_six_model_experiment.ipynb'
    parameters={'MODE':args.stage if args.stage in ['prepare','train','evaluate'] else 'evaluate'}
    if args.stage=='delta':notebook='notebooks/diagnostics/stellar_delta_shift.ipynb'
    if args.stage=='train':parameters['CONFIG_NAME']=args.config
    if args.stage=='evaluate':
        if args.registry is None:raise ValueError('Supply --registry with exact six run directories')
        parameters['MODEL_REGISTRY']=str(args.registry.resolve())
    destination=output/Path(notebook).name
    pm.execute_notebook(str(PROJECT/notebook),str(destination),parameters=parameters,
        cwd=str(PROJECT),kernel_name='python3',log_output=True)
    write_json(output/'job_manifest.json',dict(stage=args.stage,parameters=parameters,
        input_sha256=input_hashes(),executed_notebook=str(destination),environment=environment_record()))
    print('EXECUTED NOTEBOOK:',destination)


if __name__=='__main__':main()
