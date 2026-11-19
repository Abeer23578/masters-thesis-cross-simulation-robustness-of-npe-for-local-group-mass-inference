"""Portable paths for the retained optional analyses."""
from pathlib import Path
import os


def repository_root():
    candidates = [Path(__file__).resolve(), *Path(__file__).resolve().parents]
    return next(p for p in candidates if (p / 'pyproject.toml').is_file())


def research_root():
    return Path(os.environ.get('LG_PROJECT', repository_root() / 'research')).expanduser().resolve()


def archive_root():
    return Path(os.environ.get('LG_FINAL_ARCHIVE', research_root())).expanduser().resolve()


def diagnostic_artifacts():
    return Path(os.environ.get('LG_DIAGNOSTIC_ARTIFACTS',
                research_root() / 'analysis_artifacts' / 'classifier')).expanduser().resolve()


def require_opt_in(purpose):
    if os.environ.get('LG_RUN_RESEARCH_WORKFLOW') != '1':
        raise RuntimeError('Optional research workflow is disabled. Set '
            'LG_RUN_RESEARCH_WORKFLOW=1 after reading the reproduction notebook. Action: ' + purpose)


def output_dir(group):
    require_opt_in('write new ' + group + ' analysis outputs')
    base = Path(os.environ.get('LG_SUPPLEMENT_OUTPUT',
                repository_root() / 'outputs' / 'research_reanalysis')).expanduser().resolve()
    out = base / group
    out.mkdir(parents=True, exist_ok=True)
    return out
