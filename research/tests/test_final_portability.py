"""Regression checks for the final-stage v3 portability repair (no GPU/models)."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lg_npe.final_design import (FINAL_IDENTITY_COLUMNS, frame_digest, sha256,
                               validate_frozen_calibration)

ROOT = Path(__file__).parents[1]
REQUIRED = ('pair_index', 'id_a', 'id_b', 'host_i', 'host_j', 'same_host',
            'separation', 'vr', 'vt', 'mstar_a', 'mstar_b', 'log_m200c')


def frame():
    result = pd.DataFrame({c: np.arange(6, dtype=np.int64) for c in REQUIRED[:5]})
    result['same_host'] = np.arange(6) % 2 == 0
    for i, name in enumerate(REQUIRED[6:]):
        result[name] = np.arange(6, dtype=float) + i + .25
    result['log_bound_subhalo_sum'] = np.arange(6.) + 12
    return result


def test_identity_has_exact_canonical_columns_and_ignores_runtime_bound_log():
    assert FINAL_IDENTITY_COLUMNS == REQUIRED
    f = frame(); h = hashlib.sha256()
    for name in REQUIRED:
        a = f[name].to_numpy()
        h.update(name.encode())
        h.update(np.asarray(a, dtype='<i8' if a.dtype.kind in 'biu' else '<f8').tobytes())
    assert frame_digest(f) == h.hexdigest()
    assert frame_digest(f.drop(columns='log_bound_subhalo_sum')) == h.hexdigest()
    f['log_bound_subhalo_sum'] = np.nextafter(f.log_bound_subhalo_sum, np.inf)
    assert frame_digest(f) == h.hexdigest()
    f['log_bound_subhalo_sum'] = np.nan
    assert frame_digest(f) == h.hexdigest()


@pytest.mark.parametrize('column', REQUIRED)
def test_each_relevant_column_changes_identity(column):
    f = frame(); original = frame_digest(f)
    if column == 'same_host':
        f.loc[0, column] = not f.loc[0, column]
    elif f[column].dtype.kind == 'f':
        f.loc[0, column] = np.nextafter(f.loc[0, column], np.inf)
    else:
        f.loc[0, column] += 1
    assert frame_digest(f) != original


@pytest.mark.parametrize('notebook', [
    'notebooks/final/final_six_model_experiment.ipynb',
    'notebooks/diagnostics/stellar_delta_shift.ipynb'])
@pytest.mark.parametrize('column', ['mstar_a', 'vr', 'log_m200c'])
def test_preparation_rejects_real_changes_without_rewriting_split(tmp_path, notebook, column):
    cells = json.loads((ROOT/notebook).read_text(encoding='utf-8'))['cells']
    code = next(''.join(c['source']) for c in cells
                if c['cell_type']=='code' and 'Frozen data identities changed' in ''.join(c['source']))
    bundle=tmp_path/'data/derived/final_experiment_v1'; bundle.mkdir(parents=True)
    historical=tmp_path/'historical'; historical.mkdir()
    (historical/'split_indices.npz').write_bytes(b'synthetic historical reference')
    frames={'flamingo':frame(), 'tng':frame()}
    splits={d:dict(train=np.arange(3),validation=np.array([3]),test=np.array([4,5]),
                   groups=np.arange(6)) for d in frames}
    np.savez(bundle/'splits.npz', **{f'{d}_{k}':v for d,p in splits.items() for k,v in p.items()})
    split_sha=sha256(bundle/'splits.npz')
    identity={d:frame_digest(f) for d,f in frames.items()}
    identity['historical_split_sha256']=sha256(historical/'split_indices.npz')
    (bundle/'identity.json').write_text(json.dumps(identity))
    def no_write(*args, **kwargs):
        raise AssertionError('Existing frozen artifacts must never be rewritten')
    ctx=dict(PROJECT=tmp_path, frames=frames, splits=splits, np=np, json=json,
             frame_digest=frame_digest, sha256=sha256, HISTORICAL_SUM='historical',write_json=no_write)
    frames['tng']['log_bound_subhalo_sum'] += .01
    exec(compile(code, notebook, 'exec'), ctx)
    frames['tng'].loc[0,column]=np.nextafter(frames['tng'].loc[0,column], np.inf)
    with pytest.raises(AssertionError, match='Frozen data identities changed'):
        exec(compile(code, notebook, 'exec'), ctx)
    assert sha256(bundle/'splits.npz') == split_sha


def test_frozen_artifacts_remain_byte_identical_and_memberships_exact():
    required = [
        ROOT/'out/final_stage_audit/portability_v3/repair_verification.json',
        ROOT/'data/derived/final_experiment_v1/splits.npz',
    ]
    if not all(path.is_file() for path in required):
        pytest.skip('Requires the optional archived run bundle; see docs/reproduction.ipynb')
    report=json.loads((ROOT/'out/final_stage_audit/portability_v3/repair_verification.json').read_text())
    for path, expected in report['immutable_file_sha256'].items():
        resolved = ROOT/'out/final_stage_audit/portability_v3/original_design.txt' if path == 'FINAL_EXPERIMENT_DESIGN_2026-10-01.md' else ROOT/path
        assert sha256(resolved) == expected
    bundle=ROOT/'data/derived/final_experiment_v1'
    assert sha256(bundle/'splits.npz') == '69a09e48f0abec5aef8ad9ecb70b877f6fbaa50c0ca85afbe2af7e78c9f2df0d'
    identity=json.loads((bundle/'identity.json').read_text())
    assert identity['historical_split_sha256'] == 'fc38a9b561cf72bf7cb9d342385414133a732bc043adc1e80533d079ac5854b3'
    with np.load(bundle/'splits.npz') as saved:
        for key, expected in report['membership_sha256'].items():
            assert hashlib.sha256(np.asarray(saved[key],dtype='<i8').tobytes()).hexdigest() == expected


def test_calibration_provenance_rejects_changed_inputs_and_artifacts(tmp_path):
    bundle=tmp_path/'data/derived/final_experiment_v1';bundle.mkdir(parents=True)
    training={'flamingo':frame(),'tng':frame()}
    paths=['bound.npz','bound.json','delta_calibration.json','delta_calibration_draws.npy','code.py','splits.npz']
    for name in paths:(tmp_path/name).write_bytes(name.encode())
    record=dict(schema='frozen-calibration-portability-v3',
        training_identity={d:frame_digest(f) for d,f in training.items()},
        file_sha256={name:sha256(tmp_path/name) for name in paths})
    (bundle/'calibration_integrity_v3.json').write_text(json.dumps(record))
    training['tng']['log_bound_subhalo_sum'] += .001
    validate_frozen_calibration(tmp_path,training)
    training['tng'].loc[0,'mstar_a'] += 1
    with pytest.raises(AssertionError,match='training inputs changed'):
        validate_frozen_calibration(tmp_path,training)
    training['tng']=frame()
    for name in paths:
        (tmp_path/name).write_bytes(b'changed')
        with pytest.raises(AssertionError,match='dependency changed'):
            validate_frozen_calibration(tmp_path,training)
        (tmp_path/name).write_bytes(name.encode())


@pytest.mark.parametrize('shape', ['vector','row','column'])
def test_smoke_validation_indexes_inputs_before_shape_independent_reduction(shape):
    import torch
    source=ast.parse((ROOT/'scripts/run_final_stage.py').read_text())
    smoke=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='smoke')
    loop=next(n for n in smoke.body if isinstance(n,ast.For))
    start=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Assign)
               and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='vt_tensor')
    stop=next(i for i in range(start,len(loop.body)) if isinstance(loop.body[i],ast.Assert))
    code=compile(ast.Module(body=loop.body[start:stop+1],type_ignores=[]),'smoke verification','exec')
    calls=[]
    class Density:
        def log_prob(self,theta,condition):
            np.testing.assert_array_equal(theta.cpu(),condition.cpu())
            calls.append(len(theta))
            score=-theta[:,0]
            return score if shape=='vector' else score.reshape(1,-1) if shape=='row' else score[:,None]
    cpu_torch=SimpleNamespace(float32=torch.float32,no_grad=torch.no_grad,
        tensor=lambda x,**kw:torch.tensor(x,**{**kw,'device':'cpu'}),
        as_tensor=lambda x,**kw:torch.as_tensor(x,**{**kw,'device':'cpu'}))
    values=np.array([2.]*24+[8.]*8)
    ctx=dict(torch=cpu_torch,np=np,vt=values,vx=values[:,None],vd=np.array(['F']*24+['T']*8),
        scaler=SimpleNamespace(transform=lambda x:x),density=Density(),history={'best_validation_nll':5.})
    exec(code,ctx)
    assert calls==[24,8] and ctx['loss']==5.
