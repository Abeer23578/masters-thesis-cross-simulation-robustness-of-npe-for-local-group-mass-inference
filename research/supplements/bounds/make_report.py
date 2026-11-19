from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import repository_root, research_root, archive_root, diagnostic_artifacts, output_dir, require_opt_in
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

O=output_dir('bounds')
h=pd.read_csv(O/'heldout_bound_sensitivity.csv');hp=h[h.subset=='pooled']
l=pd.read_csv(O/'local_group_bound_sensitivity.csv')
t=pd.read_csv(O/'training_support_checks.csv')
sc=json.loads((O/'scenarios.json').read_text())
rows=[]
for spec in sc:
    s=spec['scenario'];a=l[l.scenario==s];b=hp[hp.scenario==s]
    rows.append(dict(**spec,lg_max_abs_median_change_dex=a.median_shift_dex.abs().max(),
        lg_max_abs_mass_change_percent=a.median_mass_change_percent.abs().max(),
        lg_max_abs_width68_change_percent=a.width68_change_percent.abs().max(),
        heldout_max_abs_bias_change_dex=b.change_bias.abs().max(),
        heldout_max_abs_rmse_change_dex=b.change_rmse.abs().max(),
        heldout_max_abs_coverage68_change_percentage_points=100*b.change_coverage68.abs().max(),
        heldout_max_individual_median_change_dex=b.max_abs_individual_median_change.max(),
        max_training_rows_excluded=t[t.scenario==s].training_outside_box.max(),
        max_test_rows_excluded=b.truth_outside_box.max()))
s=pd.DataFrame(rows);s.to_csv(O/'summary_across_models.csv',index=False)
wide=hp[hp.scenario=='wide'];wide.to_csv(O/'wide_box_changes_by_model.csv',index=False)

selected=['wide','near_population','lower_12','upper_13','tight_12_13']
z=s.set_index('scenario').loc[selected];x=np.arange(len(z))
labels=[f'[{a:g}, {b:g}]' for a,b in zip(z.lower,z.upper)]
plt.rcParams.update({'font.family':'serif','font.size':10,'pdf.fonttype':42,'savefig.dpi':220,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
aa=z.lg_max_abs_median_change_dex.to_numpy()
ax[0].bar(x,aa,color='#0072B2',width=.6)
for xx,yy in zip(x,aa):ax[0].annotate(f'{yy:.2g}',(xx,yy),xytext=(0,4),textcoords='offset points',ha='center',fontsize=8)
ax[0].set(yscale='log',ylabel='Maximum absolute median change (dex)',title='Local Group: all six models and both input scenarios')
ax[0].set_ylim(1e-6,.02)
ax[1].bar(x-.18,z.heldout_max_abs_bias_change_dex,width=.36,label='Mean bias',color='#D55E00')
ax[1].bar(x+.18,z.heldout_max_abs_rmse_change_dex,width=.36,label='RMSE',color='#009E73')
ax[1].set(yscale='log',ylabel='Maximum absolute metric change (dex)',title='Both full held-out test sets: all six models')
ax[1].set_ylim(2e-4,.2);ax[1].legend(frameon=False,loc='upper left')
for a in ax:
    a.set_xticks(x,labels,rotation=20);a.set_xlabel(r'Box limits in $\log_{10}(M/M_\odot)$')
    a.grid(axis='y',alpha=.15);a.set_axisbelow(True)
fig.suptitle('Changes relative to the original [11.5, 14.5] box; frozen networks, CPU analytic calculation',fontsize=11)
for ext in ['pdf','png']:fig.savefig(O/f'box_bound_sensitivity.{ext}')
plt.close(fig)

def fm(x):return f'{x:.3g}'
table='| Box | Largest LG median change (dex) | Largest LG 68% width change (%) | Largest test RMSE change (dex) | Largest test C68 change (percentage points) |\n|---|---:|---:|---:|---:|\n'
for r in rows:
    table+=f"| [{r['lower']:g}, {r['upper']:g}] | {fm(r['lg_max_abs_median_change_dex'])} | {fm(r['lg_max_abs_width68_change_percent'])} | {fm(r['heldout_max_abs_rmse_change_dex'])} | {fm(r['heldout_max_abs_coverage68_change_percentage_points'])} |\n"
report='''# Box-bound sensitivity of the final six models — 10 November 2026

The original box does not materially determine the Local Group results. Widening [11.5,14.5] to [11.0,15.0] changes every nominal or observational-mixture median by at most 0.00000612 dex (0.00141% in linear mass), and every 68% interval width by at most 0.00235%. The largest absolute change in pooled test-set RMSE is 0.00146 dex, in mean bias 0.00153 dex, and in C68 0.734 percentage points. The stellar-sum transfer failures persist.

## Scope and calculation

All six frozen models were tested on both component-disjoint evaluation populations (21409 FLAMINGO and231 TNG pairs), with pooled, same-host and different-host summaries. Local Group calculations include the nominal input and the original5000 approximate observational inputs with their saved weights. Ten boxes were defined before evaluating their effects; scenarios include wider, modestly tighter and intentionally restrictive supports. These are exploratory sensitivity settings, not an independently preregistered new experiment and not optimized on test performance.

For these one-dimensional affine MAFs, the conditional density is Gaussian. Its mean and scale were recovered by inverting the flow at standard-normal coordinates0,+1,-1, and linearity was checked at2.3. The density was then normalized separately over each proposed box; all quantiles were calculated analytically. Mixture quantiles were obtained by inverting the weighted average of the normalized conditional CDFs. No flattened-mixture reweighting was performed.

No GPU, model fitting, retraining, fresh posterior draws, or edits to original checkpoints were used. CUDA was disabled and all model parameters were checked to be on CPU. Source hashes and versions are in provenance.json. The local software versions differ from the archived run; checkpoint identity, original validation NLL and saved posterior distributions were therefore verified. All262 checks passed; the largest validation-NLL discrepancy was6.19e-8. Saved-draw comparisons also passed for held-out cases and both Local Group scenarios.

The baseline here uses exact conditional quantiles for the recovered model. The archived evaluation used500 draws per pair, so its empirical metrics differ slightly even at the same box. Every sensitivity difference is measured relative to the same analytic baseline. `analytic_vs_saved500_baseline.csv` records both versions; archived headline numbers are not overwritten.

## Maximum changes across models

LG maxima include all six models and both input scenarios. Test maxima include all12 model-by-test-population combinations, using pooled test sets. Different columns can attain their maxima in different models. These are sensitivity differences, not confidence intervals.

'''+table+'''
## Interpretation

- The original box removes at most0.00207% of the untruncated Local Group distribution, averaged over input components. The upper edge is particularly remote from the Local Group posteriors. Bounds [10,16] produce essentially the same LG result as [11,15].
- Stellar-sum cross-simulation mean-bias changes under [11,15] are -8.92e-6 dex for FLAMINGO -> TNG and +8.14e-7 dex for TNG -> FLAMINGO. Analytic baseline biases remain about -0.316 and+0.321 dex, respectively. C68 is unchanged in these two cases. The box is not an explanation of their large transfer errors.
- Small pooled differences do not guarantee negligible effects for every pair: widening to [11,15] changes an individual test posterior median by up to0.0339 dex. The largest p95 change among the pooled test cases is0.0102 dex.
- [11.54,13.87] contains every frozen training and test mass, but clipping tails can still move individual medians by up to0.242 dex. This maximum occurs for TNG stellar sum evaluated on FLAMINGO; its p95 change is0.000327 dex and1.32% of those pairs shift by more than0.01 dex. Thus the worst effect is not representative of most pairs. Keeping training values inside a box is not sufficient to make posterior truncation irrelevant.
- [12,13] is an intentional stress test: it excludes some training and true test masses. It changes the largest pooled RMSE by0.0621 dex and C68 by15.2 percentage points; it is not proposed as a suitable support. LG medians happen to remain inside it, but 68% widths can change by5.43%.
- This test changes support only. It does not replace the effective training mass distribution or repeat the earlier empirical-prior swap. Widening exposes mathematical tails of the trained density outside its original support, not newly validated physical predictions.

## Thesis-ready wording

The sensitivity to the declared mass bounds was tested on CPU using the six frozen models. Expanding the support from11.5–14.5 to11.0–15.0 in log10(M/Msun) changed the nominal and observational-mixture Local Group medians by less than7e-6 dex and the68% interval widths by less than0.003%. Across both held-out test populations, pooled RMSE changed by at most0.0015 dex and68% coverage by0.74 percentage points. The stellar-sum transfer biases were effectively unchanged, indicating that these failures are not driven by truncation at the adopted bounds. Narrower supports can affect individual posterior tails even when they contain all training masses.

## Files

- `run_sensitivity.py`: complete CPU calculation and validation.
- `make_report.py`: summary, figure and LaTeX generation from saved sensitivity outputs.
- `scenarios.json`: ten box settings and classifications.
- `summary_across_models.csv`: aggregate maxima shown above.
- `local_group_bound_sensitivity.csv`:120 rows, all bounds/models/input scenarios, intervals and excluded probability.
- `heldout_bound_sensitivity.csv`:360 rows, all bounds/models/domains/host subsets.
- `wide_box_changes_by_model.csv`: twelve pooled comparisons for [11,15].
- `training_support_checks.csv`: masses excluded under each box and training population.
- `*_per_pair.csv.gz`: original and selected alternative-box quantiles and conditional Gaussian parameters.
- `conditional_gaussian_parameters.npz`: all conditional parameters, including LG components, for additional CPU post-processing without loading models.
- `validation_checks.csv`, `analytic_vs_saved500_baseline.csv`, `provenance.json`.
- `box_bound_sensitivity.pdf/png`, `box_bound_table.tex`.
'''
(O/'BOX_BOUND_SENSITIVITY_REPORT.md').write_text(report,encoding='utf8')
latex=r'''% Required packages: float, booktabs.
\begin{table}[H]
\centering
\small
\caption{Sensitivity to the declared log-mass bounds, relative to the original $[11.5,14.5]$ support. Local Group (LG) maxima include all six models and both nominal and observational-mixture inputs; test maxima include both full held-out populations for all six models. Changes are computed relative to an analytic baseline with the same frozen networks. Maxima in different columns need not refer to the same model. These are sensitivity results, not confidence intervals.}
\label{tab:box_bound_sensitivity}
\begin{tabular}{@{}lrrrr@{}}
\toprule
Box & Max. $|\Delta m_{\rm LG}|$ & Max. LG $|\Delta W_{68}|$ & Max. $|\Delta\mathrm{RMSE}|$ & Max. $|\Delta C_{68}|$ \\
 & (dex) & (\%) & (dex) & (percentage points) \\
\midrule
'''
for name in ['wide','very_wide','near_population','tight_12_13']:
    r=s.set_index('scenario').loc[name]
    v=r.lg_max_abs_median_change_dex
    power=int(np.floor(np.log10(v)));mant=v/10**power
    latex+=f"$[{r.lower:g},{r.upper:g}]$ & ${mant:.2f}\\times10^{{{power}}}$ & {r.lg_max_abs_width68_change_percent:.4f} & {r.heldout_max_abs_rmse_change_dex:.5f} & {r.heldout_max_abs_coverage68_change_percentage_points:.3f} \\\\\n"
latex+=r'''\bottomrule
\end{tabular}
\par\smallskip
\begin{minipage}{\linewidth}
\footnotesize
$m_{\rm LG}$ is the posterior median log mass and $W_{68}$ the central 68\% interval width in dex; width changes are relative percentages. The $[12,13]$ support is a deliberately restrictive stress test that excludes some training and true test masses. The $[11.54,13.87]$ support excludes no training or test masses, but still affects some posterior tails.
\end{minipage}
\end{table}
'''
(O/'box_bound_table.tex').write_text(latex,encoding='utf8')
print(s[['scenario','lg_max_abs_median_change_dex','heldout_max_abs_rmse_change_dex','heldout_max_abs_coverage68_change_percentage_points']].to_string(index=False))
