#!/usr/bin/env python
"""Create an explicit six-model registry; reject ambiguous or incomplete runs."""
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
args=p.parse_args()
registry={}
for path in sorted((ROOT/'configs/final').glob('*.json')):
    config=json.loads(path.read_text())
    matches=[]
    for manifest_path in (ROOT/config['output_root']/config['experiment_id']).glob('*/run_manifest.json'):
        manifest=json.loads(manifest_path.read_text())
        if manifest['config']==config and (manifest_path.parent/'model.pkl').is_file():
            matches.append(manifest_path.parent)
    if len(matches)!=1:
        raise RuntimeError(f'{path.stem}: expected one completed run, found {matches}; choose exact runs explicitly')
    registry[path.stem]=str(matches[0].relative_to(ROOT)).replace('\\','/')
assert len(registry)==6
args.output.parent.mkdir(parents=True,exist_ok=True)
with args.output.open('x',encoding='utf-8') as f:json.dump(registry,f,indent=2)
print(args.output)
