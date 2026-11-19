"""Validate and optionally execute the reading notebooks."""
from pathlib import Path
import argparse
import ast
import base64
import contextlib
import io
import json
import os
import re
import subprocess
import sys
import nbformat

ROOT=Path(__file__).resolve().parents[1]
READING=sorted((ROOT/'notebooks').glob('[01][0-9]_*.ipynb'))

def execute_one(path):
    """Run a notebook in a fresh IPython process, collecting ordinary rich outputs."""
    import matplotlib
    matplotlib.use('module://matplotlib_inline.backend_inline')
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    from traitlets.config import Config
    from matplotlib_inline.backend_inline import configure_inline_support
    config=Config()
    config.HistoryManager.hist_file=':memory:'
    shell=InteractiveShell.instance(config=config)
    configure_inline_support(shell,'inline')
    notebook=nbformat.read(path,as_version=4)
    os.chdir(ROOT)
    for count,cell in enumerate((c for c in notebook.cells if c.cell_type=='code'),1):
        with capture_output(stdout=True,stderr=True,display=True) as captured:
            result=shell.run_cell(cell.source,store_history=False)
        if result.error_before_exec or result.error_in_exec:
            raise RuntimeError(f'{path.name}, code cell {count}: {result.error_before_exec or result.error_in_exec}')
        cell.execution_count=count
        outputs=[]
        if captured.stdout: outputs.append(nbformat.v4.new_output('stream',name='stdout',text=captured.stdout))
        if captured.stderr: outputs.append(nbformat.v4.new_output('stream',name='stderr',text=captured.stderr))
        for out in captured.outputs:
            outputs.append(nbformat.v4.new_output('display_data',data=out.data,metadata=out.metadata))
        cell.outputs=outputs
    nbformat.validate(notebook)
    dest=ROOT/'outputs/executed'/path.name
    dest.parent.mkdir(parents=True,exist_ok=True)
    nbformat.write(notebook,dest)
    print('Executed',path.name,flush=True)

def excluded(relative):
    parts=relative.parts
    name=relative.name
    if name.endswith(('.pkl','.pickle','.npz','.npy','.hdf5','.h5','.pt','.pth','.zip','.pyc','.log')): return True
    if any(x in parts for x in ['.venv','__pycache__','.ipynb_checkpoints','.pytest_cache','.git']): return True
    s=relative.as_posix()
    return any(s.startswith(prefix) for prefix in ['outputs/','releases/','data/frames/','data/local/',
        'research/analogues/cache/','research/analogues/tng_for_comparison/','research/analogues/tng_for_comparison_v3/',
        'research/data/','research/out/','research/pickles','research/logs/','research/analysis_artifacts/'])

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.worker:
        execute_one(args.worker.resolve());return
    failures=[];files=[];notebooks=0
    for path in ROOT.rglob('*'):
        if not path.is_file() or excluded(path.relative_to(ROOT)): continue
        relative=path.relative_to(ROOT)
        if path.suffix=='.md' and relative.as_posix()!='README.md':
            failures.append(f'Only README.md should be delivered: {relative}')
        if path.stat().st_size>=90_000_000: failures.append(f'Oversized public file: {relative}')
        if path.suffix=='.py':
            try:ast.parse(path.read_text(encoding='utf-8-sig'))
            except SyntaxError as e: failures.append(f'Python syntax {relative}: {e}')
        if path.suffix=='.ipynb':
            nb=nbformat.read(path,as_version=4);nbformat.validate(nb);notebooks+=1
            if nb.metadata.get('authors')!=[{'name':'Abeer Ayyaz'}]: failures.append(f'Author metadata: {relative}')
            for cell in nb.cells:
                if cell.cell_type=='code':
                    try:ast.parse(cell.source)
                    except SyntaxError:
                        if not re.search(r'(?m)^\s*[!%]',cell.source):failures.append(f'Cell syntax: {relative}')
                else:
                    for link in re.findall(r'\]\(([^)]+)\)',cell.source):
                        if '://' in link or link.startswith(('#','mailto:')):continue
                        target=link.split('#')[0].split('?')[0]
                        if target and not (path.parent/target).exists():failures.append(f'Broken link in {relative}: {link}')
        files.append(dict(path=relative.as_posix(),bytes=path.stat().st_size))
    if len(READING)!=14:failures.append(f'Expected 14 reading notebooks, got {len(READING)}')
    print(f'Checked {notebooks} notebooks and {len(files)} public files.')
    if failures:
        print('\n'.join(failures));raise SystemExit(1)
    if args.execute:
        for path in READING+[ROOT/'notebooks/90_generate_catalogue_caches.ipynb']:
            subprocess.run([sys.executable,'-X','utf8','-B',str(Path(__file__)), '--worker',str(path)],check=True)
    print('All requested checks passed.')

if __name__=='__main__':main()
