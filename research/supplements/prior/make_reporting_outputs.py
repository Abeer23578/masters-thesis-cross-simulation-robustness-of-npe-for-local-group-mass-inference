from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import repository_root, research_root, archive_root, diagnostic_artifacts, output_dir, require_opt_in
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

O=output_dir('prior')
d=pd.read_csv(O/'prior_swap_test_metrics.csv')
d=d[d.subset=='pooled']
plt.rcParams.update({'font.family':'serif','font.size':10,'pdf.fonttype':42,'savefig.dpi':250,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained',gridspec_kw={'width_ratios':[1,1.45]})
with np.load(O/'training_prior_density_grids.npz') as g:
    for domain,color,label in [('flamingo','#0072B2','FLAMINGO'),('tng','#D55E00','TNG300'),('combined','#009E73','Equal-domain mixture')]:
        axes[0].plot(g['grid'],np.exp(g[f'{domain}_1.0']),color=color,label=label)
axes[0].set(xlim=(11.5,14),xlabel=r'$\theta=\log_{10}(M/M_\odot)$',ylabel='Training mass density',title='Training-only KDE priors')
axes[0].legend(frameon=False,fontsize=8)
names=['flamingo_dynamics','tng_dynamics','flamingo_stellar_sum','tng_stellar_sum']
for i,name in enumerate(names):
    sub=d[d.model==name]
    axes[1].scatter(sub[sub.treatment=='original'].bias,i,color='black',marker='s',label='Original' if i==0 else None,zorder=4)
    for b,color,offset in [(.5,'#0072B2',-.14),(1.,'#D55E00',0),(2.,'#009E73',.14)]:
        row=sub[(sub.treatment=='prior_swap')&(sub.bandwidth_factor==b)].iloc[0]
        axes[1].scatter(row.bias,i+offset,color=color,label=f'Prior swap: {b:g} x Scott' if i==0 else None,zorder=3)
axes[1].axhspan(.65,1.35,color='#F0E442',alpha=.22,zorder=0)
axes[1].set_yticks(range(4),['F to T: dynamics','T to F: dynamics*','F to T: stellar sum','T to F: stellar sum'])
axes[1].invert_yaxis();axes[1].axvline(0,color='.5',ls=':')
axes[1].set(xlabel='Mean posterior-median error (dex)',title='Prior-swap bandwidth sensitivity')
axes[1].legend(frameon=False,fontsize=8,loc='upper left',bbox_to_anchor=(0,-.13),ncol=2)
fig.suptitle('* Reverse dynamics is unstable: report bandwidth and importance-weight ESS.',fontsize=10)
for ext in ['pdf','png']:fig.savefig(O/f'prior_swap_bias.{ext}')
plt.close(fig)

# Verify that the log-density interpolation never extrapolated beyond its grid.
import json
prov=json.loads((O/'provenance.json').read_text())
for path in prov['sources']:
    if path.endswith('_draws.npz'):
        with np.load(path) as z:
            assert z['samples'].min()>=11.5 and z['samples'].max()<=14.5
lp=json.loads((O/'local_group_prior_swap_provenance.json').read_text())
for path in lp['sources']:
    with np.load(path) as z:
        assert z['theta_log10_msun'].min()>=11.5 and z['theta_log10_msun'].max()<=14.5
print('Figure saved; all test and Local Group draws lie within the density grid.')
