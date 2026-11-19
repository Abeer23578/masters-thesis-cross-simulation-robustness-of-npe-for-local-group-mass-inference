"""Audit-only CPU deserialization of user-owned frozen checkpoints.

Does not use or modify the frozen environment guard. A separate, explicitly
non-pinned inspection path remaps tensor storages to CPU; no model inference.
"""
import os, sys, io, pickle, json, hashlib, traceback
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import repository_root, research_root, archive_root, diagnostic_artifacts, output_dir, require_opt_in
sys.dont_write_bytecode=True
os.environ['OMP_NUM_THREADS']='2'
import torch
torch.set_num_threads(2)
R=archive_root()
class CPUUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        if module=='torch.storage' and name=='_load_from_bytes':
            return lambda b:torch.load(io.BytesIO(b),map_location='cpu',weights_only=False)
        return super().find_class(module,name)
def load_cpu(path):
    require_opt_in("deserialize a trusted archived checkpoint on CPU")
    with Path(path).open('rb') as f: return CPUUnpickler(f).load()
if __name__=='__main__':
    O=output_dir('classifier')
    records={}
    for key,rel in json.loads((R/'out/final_six/model_registry_v1.json').read_text()).items():
        try:
            m=load_cpu(R/rel/'model.pkl');d=m['density_estimator'];d.eval()
            h=hashlib.sha256()
            for name,a in sorted(d.state_dict().items()):
                h.update(name.encode());h.update(a.detach().cpu().numpy().tobytes())
            records[key]=dict(keys=list(m),config=m['config'],features=m['feature_names'],target=m['target'],
                density_class=str(type(d)),parameter_hash=h.hexdigest(),n_parameters=sum(p.numel() for p in d.parameters()),
                scaler_mean=m['scaler'].mean_.tolist(),scaler_scale=m['scaler'].scale_.tolist(),
                architecture=str(d),state_shapes={k:list(v.shape) for k,v in d.state_dict().items()})
            print(key,records[key]['density_class'],records[key]['n_parameters'],flush=True)
        except Exception:
            records[key]={'error':traceback.format_exc()};print(key,records[key]['error'],flush=True)
    (O/'checkpoint_inspection.json').write_text(json.dumps(records,indent=2),encoding='utf8')
