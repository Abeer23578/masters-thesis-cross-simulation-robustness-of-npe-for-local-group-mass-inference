"""Copy and verify known local artifacts."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

ROOT=Path(__file__).resolve().parents[1]

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--source-label',choices=['original_project','final_archive','thesis_workspace'],required=True)
    args=p.parse_args()
    records=json.loads((ROOT/'provenance/artifacts.json').read_text(encoding='utf-8'))['artifacts']
    copied=verified=missing=0
    source_root=args.source.resolve()
    by_size=None
    source_digests={}
    for item in records:
        if item['source_label']!=args.source_label: continue
        destination=(ROOT/item['destination']).resolve()
        if not destination.is_relative_to(ROOT):
            raise ValueError('Artifact path leaves the configured root')
        if destination.exists():
            if digest(destination)!=item['sha256']:
                raise FileExistsError(f'Refusing to replace differing file: {destination.relative_to(ROOT)}')
            verified+=1;continue
        if item.get('metadata_template'):
            template=(ROOT/item['metadata_template']).resolve()
            if not template.is_relative_to(ROOT) or digest(template)!=item['sha256']:
                raise ValueError('Artifact metadata template checksum mismatch')
            destination.parent.mkdir(parents=True,exist_ok=True)
            with template.open('rb') as src,destination.open('xb') as dst:shutil.copyfileobj(src,dst)
            copied+=1;continue
        if by_size is None:
            if not source_root.is_dir():raise NotADirectoryError(source_root)
            by_size={}
            for candidate in source_root.rglob('*'):
                if candidate.is_symlink() or not candidate.is_file() or '.git' in candidate.parts:continue
                by_size.setdefault(candidate.stat().st_size,[]).append(candidate)
        source=None
        for candidate in by_size.get(item['bytes'],[]):
            if candidate not in source_digests:source_digests[candidate]=digest(candidate)
            if source_digests[candidate]==item.get('source_sha256',item['sha256']):
                source=candidate;break
        if source is None:
            print('Unavailable artifact:',item['destination']);missing+=1;continue
        destination.parent.mkdir(parents=True,exist_ok=True)
        with source.open('rb') as src,destination.open('xb') as dst: shutil.copyfileobj(src,dst)
        if digest(destination)!=item['sha256']: raise ValueError('Copied artifact checksum mismatch')
        copied+=1
    print(f'Copied {copied}; verified existing {verified}; unavailable {missing}. No inputs moved or removed.')

if __name__=='__main__': main()
