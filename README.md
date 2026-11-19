# Cross-Simulation Robustness of Neural Posterior Estimation for Local Group Mass Inference

**Abeer Ayyaz**

This thesis investigates the robustness of neural posterior estimation under
cross-simulation transfer between FLAMINGO and TNG300. The notebooks cover catalogue
selection, exploratory analysis, prediction and calibration, then test the features
and population differences associated with transfer failure. The final notebooks
apply the models to the observed Local Group and explore classifier-weighted mixtures.

For a short overview of the findings, see [Key results from the thesis](notebooks/key_results_from_the_thesis.ipynb).

## Setup

Use Python 3.11 or newer. From the repository folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m jupyter lab
```

The [setup notebook](docs/setup.ipynb) includes Linux/macOS commands.
`requirements.txt` runs the result notebooks, `requirements-data.txt` adds catalogue
extraction, and `requirements-research.txt` records the model-training environment.

## Notebooks

Start with [00: overview](notebooks/00_start_here.ipynb), then follow the numbered sequence.
The notebooks use the small tables in `results/` and run on a CPU without model pickles.

| Notebook | Topic |
|---|---|
| [01](notebooks/01_catalogues_and_pair_selection.ipynb) | Catalogues, selection and pair finding |
| [02](notebooks/02_features_and_population_shift.ipynb) | EDA, feature distributions and host composition |
| [03](notebooks/03_npe_and_experimental_design.ipynb) | NPE, feature blocks, splits and training |
| [04](notebooks/04_within_and_cross_simulation.ipynb) | Within-simulation performance and transfer |
| [05](notebooks/05_calibration.ipynb) | Coverage and rank diagnostics |
| [06](notebooks/06_bootstrap_and_sample_size.ipynb) | Sample-size and training-noise bootstrap references |
| [07](notebooks/07_feature_diagnostics_and_ablation.ipynb) | Feature-space diagnostics and ablation |
| [08](notebooks/08_alternative_mass_targets.ipynb) | Alternative mass inputs and targets |
| [09](notebooks/09_stellar_normalization.ipynb) | Stellar-mass normalization |
| [10](notebooks/10_common_support.ipynb) | Common support and conditional shift |
| [11](notebooks/11_prior_and_support_sensitivity.ipynb) | Empirical-prior and support sensitivity |
| [12](notebooks/12_local_group_and_interpretation.ipynb) | The observed Local Group and physical interpretation |
| [13](notebooks/13_classifier_mixture_posteriors.ipynb) | Classifier-weighted posterior mixtures |

The [thesis map](docs/thesis_map.ipynb) connects the tables and figures to the notebooks.
[Scientific notes](docs/scientific_notes.ipynb) give further definitions and caveats.

## Data and reproduction

[Notebook 90](notebooks/90_generate_catalogue_caches.ipynb) generates the FLAMINGO and
TNG300 sample/pair caches using the catalogue pipelines. The [data guide](docs/data.ipynb)
lists the official sources, output files and TNG API setup. Pickles, raw arrays,
checkpoints and large posterior archives are excluded from Git. There is no public
release of the derived artifact bundle yet; the result notebooks use the included summaries. Exact model refits and checkpoint analyses also require the archived split and calibration files described in the reproduction guide.

[Notebook 92](notebooks/92_reproduce_saved_analyses.ipynb) gives the command order for
rerunning the prior, support-bound and classifier analyses. See [reproduction](docs/reproduction.ipynb)
for checkpoint reuse and model refits.

## Files

| Folder | Contents |
|---|---|
| `notebooks/` | Main analyses and data-generation notebooks |
| `docs/` | Setup, data, reproduction and thesis-reference notebooks |
| `results/` | Aggregate tables and bootstrap records |
| `research/notebooks/original/` | Detailed EDA, training and diagnostic workflows |
| `research/src/lg_npe/` | Feature, split, training and evaluation helpers |
| `research/analogues/`, `research/TNG300/` | Catalogue selection and particle-feature extraction |
| `scripts/` | Artifact import and repository checks |
| `provenance/` | Input checksums and file records |

Run `python scripts/check_repository.py` to check notebook structure and links.
Add `--execute` to run the result notebooks and save copies in `outputs/`.

## AI use

AI was used to organize this repository, clean up the directory structure, and add
comments and explanations to make the work easier to follow. The ideas,
experiments, and conclusions remain my own. The original working directory was
messy and had a difficult-to-follow trail; this repository presents the same work
in a clearer form.
