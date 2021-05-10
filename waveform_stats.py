#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This script generates Figure S1C-D in the manuscript
"Graded Remapping of Hippocampal Ensembles under Sensory Conflicts" written by
D. Fetterhoff, A. Sobolev & C. Leibold.

All analysis code was written by D. Fetterhoff

"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
import seaborn as sns

# Load data from this folder
hdf5Dir = '/home/fetterhoff/Graded_Remapping/'

combinedResultDir = hdf5Dir+'waveform_stats/' # Save in subdirectory
if not os.path.exists(combinedResultDir):
    os.makedirs(combinedResultDir)

wavedf = pd.read_csv(hdf5Dir+'allwavecombined_final.csv')
wavedf['neuron_type'] = 0
fil=(wavedf.spike_ratio < 1.5)
wavedf.loc[fil, 'neuron_type'] = 'INT'
fil=(wavedf.spike_ratio >= 1.5)
wavedf.loc[fil,'neuron_tpye'] = 'PC'


#%% scatter spike ratio vs firing rate

pl.rcParams.update({'font.size': 6})

ajp = sns.jointplot(x=wavedf.spike_ratio[wavedf.spike_ratio < 10],y=np.log10(wavedf.mfr[wavedf.spike_ratio < 10]),alpha=0.25,joint_kws={"s": 1}, height=2.2)
yl1, yl2 = ajp.ax_joint.get_ylim()
ajp.ax_joint.fill_between([0,1.5],yl1, yl2, facecolor='C1',alpha=0.2)
ajp.ax_joint.fill_between([1.5,10],np.log10(5.0), yl2, facecolor='C1',alpha=0.2)
ajp.ax_joint.set_ylim(yl1, yl2)
ajp.ax_joint.set_xlabel('Spike Ratio')
ajp.ax_joint.set_ylabel('Log of Mean Firing Rate (Hz)')
ajp.ax_marg_x.axvline(1.5,color='C1',alpha=0.5)
ajp.ax_joint.set_ylim(yl1, yl2)
ajp.savefig(combinedResultDir+'fig_S1C_mfr_spike_ratio.pdf',format='pdf', dpi=300, bbox_inches = 'tight', pad_inches = 0.05)
pl.close()
perc_lost = (wavedf.spike_ratio > 10).sum() / float(len(wavedf)) *100
print('{}% of neurons not shown (SR > 10)'.format(perc_lost))

#%% Box plot of spike width by neuron group (PC vs INT)

ax = sns.catplot(x='neuron_type', y='spike_width_ms', kind='boxen', data=wavedf, height=2.2)
ax.set_ylabels('Spike Width (ms)')
ax.set_xlabels('Neuron Type')
ax.savefig(combinedResultDir+'fig_S1D_spike_width.pdf',format='pdf', dpi=300, bbox_inches = 'tight', pad_inches = 0.05)
pl.close()
