#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This script generates all panels of Figure 2 and Tables S2-S3 in the manuscript
"Graded Remapping of Hippocampal Ensembles under Sensory Conflicts" written by
D. Fetterhoff, A. Sobolev & C. Leibold.
"""
import os
import numpy as np
import matplotlib.pyplot as pl
import itertools as it
import pandas as pd
import scipy.stats as ss

fileList = [
    ['g0395_d1'],
    ['g0395_d2'],
    ['g0395_d3'],
    ['g0397_d1'],
    ['g0397_d2'],
    ['g0397_d3'],
    ['g2017_d1'],
    ['g2017_d2'],
    ['g2017_d3'],
    ['g2018_d1'],
    ['g2018_d2'],
    ['g2018_d3'],
    ['g2783_d1'],
    ['g2783_d2'],
    ['g2783_d3'],
    ['g2784_d1'],
    ['g2784_d2'],
    ['g2784_d3']
    ]

speedThresh = 5 # cm/s, to discard spikes during stillness
dt = 0.05 # Sampling rate of the virtual trajectory

#hdf5Dir = '/home/dustin/Documents/hdf5_v1/' # Load data from this folder
hdf5Dir = '/home/fetterhoff/atlas/RM_45max/combined_hdf5_from_raw_gsp2_removedInfSR/'

combinedResultDir = hdf5Dir+'rate_remapping/' # Save in subdirectory
if not os.path.exists(combinedResultDir):
    os.makedirs(combinedResultDir)

bd = [187.5, 275, 412.5, 500] # boundaries for all maze segments

totalMazeLength = 622 # measured from the setup
xBins = np.linspace(1,24,80) # old: np.arange(1,24,0.29) np.linspace(1,24,80)
totalMazeLength = 622 # Measured from physical virtual reality setup
xCenters = (xBins + np.diff(xBins)[0]/2)[:-1]
Nbins = xBins.size -1
xBinsReal = np.linspace(0, totalMazeLength, len(xBins))
xCentersReal = (xBinsReal + np.diff(xBinsReal)[0]/2)[:-1]

ibd = [np.abs(xCentersReal - b).argmin() for b in bd] # index of boundaries
itmp = ibd[:]
itmp.insert(0,0), itmp.append(len(xCentersReal)-1)
iparts = [i+1 for i in itmp]

# Overlap and rate remapping data frames
df_overlap_all = pd.DataFrame([])
df_peakComp_all = pd.DataFrame([])
df_count = pd.DataFrame([])

pl.rcParams.update({'font.size': 6, 'xtick.labelsize':6, 'ytick.labelsize':6, 'legend.fontsize':6, 'axes.facecolor':'white', 'lines.linewidth': 1.25, 'lines.markersize': 2.0, 'axes.labelsize': 6, 'figure.titlesize' : 6, 'axes.titlesize' : 'medium'})

#%%
for il in np.arange(len(fileList)):
    folderName = fileList[il]
    session = folderName[0][-9:]
    print(session)

    # Load the necessary files
    f1 = hdf5Dir+session+'_dat.h5'
    spikeDF = pd.read_hdf(f1, 'spikeDF')

    f2 = hdf5Dir+session+'_laps_traj.h5'
    lapsDF = pd.read_hdf(f2, 'lapsDF')
    trajDF = pd.read_hdf(f2, 'trj')
    lapsDB = np.array(lapsDF)

    f3 = hdf5Dir+session+'_resultsDB.h5'
    cellResultsDB = pd.read_hdf(f3, 'cellResultsDB')

    nPlaceFields = 0 # Count the number of place fields
    for i in spikeDF.FieldPeakLoc:
        nPlaceFields += len(i)

    # Table S1
    sumN = pd.DataFrame({'session': session, 'nPlaceCells' : len(spikeDF), 'nPlaceFields' : nPlaceFields}, index=[il])
    df_count = pd.concat([df_count, sumN])

    #%% Identify cell types based on peak locations: Create a single cell summary pie chart and bar chart

    ivalidspeeds = trajDF.speed > speedThresh # use speed calculated in place_cell.py

    segment_type_code = np.zeros([len(spikeDF), 5]) # 5 different maze segements
    for ii, cell_id in enumerate(spikeDF.T):
        segment_type_code[ii,:] = spikeDF.loc[cell_id].segment_types

#    binned_traj, xbin = np.histogram(trajDF.places_cm[ivalidspeeds], bins=xBinsReal)
#    binned_traj1, xbin = np.histogram(trajDF.places_cm[np.logical_and(trajDF.maze_type==1, ivalidspeeds)], bins=xBinsReal)
#    binned_traj_1, xbin = np.histogram(trajDF.places_cm[np.logical_and(trajDF.maze_type==-1, ivalidspeeds)], bins=xBinsReal)
#    binned_traj2, xbin = np.histogram(trajDF.places_cm[np.logical_and(trajDF.maze_type==2, ivalidspeeds)], bins=xBinsReal)
#    binned_traj_2, xbin = np.histogram(trajDF.places_cm[np.logical_and(trajDF.maze_type==-2, ivalidspeeds)], bins=xBinsReal)

    pc_mat1, pc_mat_1 = np.zeros([len(spikeDF),len(xCentersReal)]), np.zeros([len(spikeDF),len(xCentersReal)])
    pc_mat2, pc_mat_2 = np.zeros([len(spikeDF),len(xCentersReal)]), np.zeros([len(spikeDF),len(xCentersReal)])

    for q, cid in enumerate(spikeDF.T):
        if cid > -1:

                sp = spikeDF.loc[cid] # need to use stList so indices match with CellDB
                cDB = cellResultsDB[cellResultsDB.cell_id == cid]
                pc_mat1[q], pc_mat_1[q] = np.array(cDB[cDB.choice == 1].normRate), np.array(cDB[cDB.choice == -1].normRate)
                pc_mat2[q], pc_mat_2[q] = np.array(cDB[cDB.choice == 2].normRate), np.array(cDB[cDB.choice == -2].normRate)

#                #spike_places is almost the same as cDB.xCount, so we need to divide it by the binned trajectory
#                spike_places1, xbin = np.histogram(sp.real_cm[np.logical_and(sp.maze_type==1, \
#                                    sp.speed_cms > speedThresh)], bins=xBinsReal)
#                spike_places_1, xbin = np.histogram(sp.real_cm[np.logical_and(sp.maze_type==-1, \
#                                    sp.speed_cms > speedThresh)], bins=xBinsReal)
#                spike_places2, xbin = np.histogram(sp.real_cm[np.logical_and(sp.maze_type==2, \
#                                    sp.speed_cms > speedThresh)], bins=xBinsReal)
#                spike_places_2, xbin = np.histogram(sp.real_cm[np.logical_and(sp.maze_type==-2, \
#                                    sp.speed_cms > speedThresh)], bins=xBinsReal)
#
#                spike_rate_norm1 = np.nan_to_num(spike_places1 / (binned_traj1 * dt))
#                spike_rate_norm_1 = np.nan_to_num(spike_places_1 / (binned_traj_1 * dt))
#                spike_rate_norm2 = np.nan_to_num(spike_places2 / (binned_traj2 * dt))
#                spike_rate_norm_2 = np.nan_to_num(spike_places_2 / (binned_traj_2 * dt))
#
#                pc_mat1[q], pc_mat_1[q] = spike_rate_norm1, spike_rate_norm_1
#                pc_mat2[q], pc_mat_2[q] = spike_rate_norm2, spike_rate_norm_2

    #%% create a data frame of all single rate firing peak information
    pvc_tochoice = {1:1, 2:-1, 3:2, 4:-2}
    g = 2 # number of bins to search for peak in either direction
    singlePeakMask = np.logical_and(segment_type_code > 0, segment_type_code < 5) # Find only single-peak neurons
    nrmsk = singlePeakMask.sum(axis=1) # neuron mask
    for i, cid in enumerate(spikeDF.T):
        if nrmsk[i] > 0:
            cDB = cellResultsDB[cellResultsDB.cell_id == cid]
            # loop through locations with single fields
            for ipf, pf in enumerate(np.where(np.logical_and(segment_type_code[i] > 0, segment_type_code[i] < 5))[0]):
                ch = pvc_tochoice[segment_type_code[i][pf]] # This will be the type of field
                # index of the peak in this maze segment
                idx = cDB.normRate[cDB.choice == ch].reset_index()[iparts[pf]:iparts[pf+1]].idxmax().normRate
                # Find where is the max in that part and then get the range for the whole peak to compare in other mazes
                peakRateCH = cDB.normRate[cDB.choice == ch].reset_index().iloc[idx].normRate
                id1, id2 = np.max([idx-g, 0]), np.min([idx+g+1, 78]) # add one to g to get odd number of bins covered
                peakRate1 = cDB.normRate[cDB.choice == 1].reset_index().iloc[id1:id2].normRate.max()
                peakRate_1 = cDB.normRate[cDB.choice == -1].reset_index().iloc[id1:id2].normRate.max()
                peakRate2 = cDB.normRate[cDB.choice == 2].reset_index().iloc[id1:id2].normRate.max()
                peakRate_2 = cDB.normRate[cDB.choice == -2].reset_index().iloc[id1:id2].normRate.max()
                dictPeak = pd.DataFrame({'gerbilID': session[:5], 'data': session[6:], 'cell_name' : spikeDF.file[cid],
                                         'cell_id' : cid, 'peakLocation' : cDB.XpositionReal[idx], 'peak_segment' : pf,
                                         'peak_choice': ch, 'peak_rate_choice' : peakRateCH, 'peak1': peakRate1, 'peak-1' : peakRate_1,
                                         'peak2' : peakRate2, 'peak-2' : peakRate_2}, index = [i])
                df_peakComp_all = pd.concat([df_peakComp_all, dictPeak])

    #%% Measure Overlap
    win = [pc_mat1, pc_mat1, pc_mat1,  pc_mat_1, pc_mat_1]
    qin = [pc_mat_1, pc_mat2, pc_mat_2, pc_mat2, pc_mat_2]
    labelsqw = ['1_1','12','1_2','_12','_1_2']

    part_labels = ['fh_','fc_','mh_','lc_','lh_']

    # correlations done for each spatial bin, not each neuron
    overlap_dict = {}
    for i, (w, q) in enumerate(zip(win, qin)):
        meandata1, meandata2 = np.mean(w, axis=1), np.mean(q, axis=1)
        overlap = (np.c_[meandata1, meandata2].min(axis=1) / np.c_[meandata1, meandata2].max(axis=1)) # overlap, Leutgeb...Moser 2004
        lab = 'overlap'+labelsqw[i] # label
        overlap_dict[lab] = []
        overlap_dict[lab].append(overlap)
        overlap_dict[lab] = overlap_dict[lab][0]

        for ip, pp, pb in zip(iparts, iparts[1:], part_labels):
            labe = pb+lab
            maxdata1s, maxdata2s = np.mean(w[:, ip-1:pp], axis=1), np.mean(q[:, ip-1:pp], axis=1) # s for segment of maze
            overlapp = (np.c_[maxdata1s, maxdata2s].min(axis=1) / np.c_[maxdata1s, maxdata2s].max(axis=1))
            overlap_dict[labe] = []
            overlap_dict[labe].append(overlapp)
            overlap_dict[labe] = overlap_dict[labe][0]

    overlap_dict['cell_num'] = np.array(spikeDF.index)
    overlap_dict['sess_cell'] = np.array(spikeDF.file)[:]
    overlap_dict['fh_type'] = segment_type_code[:,0]
    overlap_dict['fc_type'] = segment_type_code[:,1]
    overlap_dict['mh_type'] = segment_type_code[:,2]
    overlap_dict['lc_type'] = segment_type_code[:,3]
    overlap_dict['lh_type'] = segment_type_code[:,4]

    overlapDF = pd.DataFrame(overlap_dict)
    df_overlap_all = pd.concat([df_overlap_all, overlapDF])

#%% Save to csv files
df_count.to_csv(combinedResultDir+'table_S1_place_cell_field_counts.csv')
#df_overlap_all.to_csv(combinedResultDir+'df_overlap_all.csv')
#df_peakComp_all.to_csv(combinedResultDir+'df_peakComp_all.csv')

#%% Peak comparison data to assess rate remapping
df_ci_rateRemapping = pd.DataFrame([])
choiceToMazeType = {1:'R', -1:'L',2:'R*', -2:'L*'}
def r_to_z(r):
    return np.log((1 + r) / (1 - r)) / 2.0

def z_to_r(z):
    e = np.exp(2 * z)
    return((e - 1) / (e + 1))

def r_confidence_interval(r, n, alpha=0.05):
    z = r_to_z(r)
    se = 1.0 / np.sqrt(n - 3)
    z_crit = ss.norm.ppf(1 - alpha/2)  # 2-tailed z critical value

    lo = z - z_crit * se
    hi = z + z_crit * se

    # Return a sequence
    return (z_to_r(lo), z_to_r(hi))

fig, ax = pl.subplots(4,4,figsize=(3,4),sharex=True,sharey=True)
ax = ax.ravel()

fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axis
pl.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
fig.suptitle('Rate Remapping in Single Maze-Type Fields', y=.92, fontsize = 6)
pl.ylabel("Peak Rate (Hz) in Single Maze-Type Fields", labelpad=8)
pl.xlabel("Peak Rate (Hz) in surrounding 5 bins of other Maze-Type", labelpad=4)

choiceList = [1, -1, 2, -2]
# Fisher's Z https://medium.com/@shandou/how-to-compute-confidence-interval-for-pearsons-r-a-brief-guide-951445b9cb2d

for i, (rch, cch) in enumerate(it.product(choiceList, repeat=2)):
    ax[i].set(adjustable='box', aspect='equal')
    ax[i].tick_params(axis=u'both', which=u'both',length=0)
    if i not in [0,5,10,15]:
        mk = df_peakComp_all[df_peakComp_all.peak_choice == rch]
        r, p = ss.pearsonr(mk.peak_rate_choice, mk['peak'+str(cch)])
        CI_r = r_confidence_interval(r, len(mk))

        r, p = np.around(r,2), np.around(p,2)
        ax[i].plot([0,30],[0,30],'C1',alpha=0.5)

        ax[i].plot( mk['peak'+str(cch)], mk.peak_rate_choice,'k.',alpha=0.3, markersize=1)

        ax[i].set_xlim([0,30]); ax[i].set_ylim([0,30])
        ax[i].set_xticks([0, 30]); ax[i].set_yticks([0, 30]);

        ax[i].set_title('r = {}'.format(r),pad=1, fontsize=6)

        cidict = pd.DataFrame({'Field':choiceToMazeType[rch],'Data':choiceToMazeType[cch],'n':len(mk), 'CI_lo':CI_r[0],'CI_hi':CI_r[1]},index=[i])
        df_ci_rateRemapping = pd.concat([df_ci_rateRemapping, cidict])
    else:
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['left'].set_visible(False)
        ax[i].spines['bottom'].set_visible(False)

ax[0].set_ylabel('R', labelpad=0)
ax[4].set_ylabel('L', labelpad=0)
ax[8].set_ylabel('R*', labelpad=0)
ax[12].set_ylabel('L*', labelpad=0)
ax[12].set_xlabel('R', labelpad=0)
ax[13].set_xlabel('L', labelpad=0)
ax[14].set_xlabel('R*', labelpad=0)
ax[15].set_xlabel('L*', labelpad=0)

df_ci_rateRemapping.to_csv(combinedResultDir+'table_S2_rate_remapping_single_peaks_CI.csv')
pl.savefig(combinedResultDir+'fig_2A_rate_remapping_single_peaks.pdf',format='pdf', bbox_inches = 'tight', pad_inches = 0.05)
pl.close()

#%% Overlap summary figure
fig, ax = pl.subplots(3,5, sharex=True, sharey=True, figsize=(3.75,4))
ax = ax.ravel()
ax[0].set_yticks([0,0.5,1.0])

pi = [0, 1, 2, 3, 4]
df_overlap_stats = pd.DataFrame([])

obin = np.linspace(0,1,100)
hds = ['fh_','fc_','mh_','lc_','lh_']
titles = ['First Hallway','First Corner','Middle Hallway','Last Corner','Last Hallway']

# For single maze type cells
for i, h in enumerate(hds):
    mk = (df_overlap_all[h+'type'] > 0) & (df_overlap_all[h+'type'] < 5)

    values, _ = np.histogram(df_overlap_all[h+'overlap1_1'][mk], obin)
    cum = np.cumsum(values).astype(float) / np.cumsum(values).max()
    ax[pi[i]].plot(obin[1:], cum, label='R & L', alpha=0.75, c='C1') # Plot all data on the first plot

    values, _ = np.histogram(df_overlap_all[h+'overlap12'][mk], obin)
    cum = np.cumsum(values).astype(float) / np.cumsum(values).max()
    ax[pi[i]].plot(obin[1:], cum, label='R & R*', alpha=0.75, c='C3') # Plot all data on the first plot

    values, _ = np.histogram(df_overlap_all[h+'overlap_1_2'][mk], obin)
    cum = np.cumsum(values).astype(float) / np.cumsum(values).max()
    ax[pi[i]].plot(obin[1:], cum, label='L & L*', alpha=0.75, c='C0') # Plot all data on the first plot

    values, _ = np.histogram(df_overlap_all[h+'overlap1_2'][mk], obin)
    cum = np.cumsum(values).astype(float) / np.cumsum(values).max()
    ax[pi[i]].plot(obin[1:], cum, label='R & L*', alpha=0.75, c='C2') # Plot all data on the first plot

    values, _ = np.histogram(df_overlap_all[h+'overlap_12'][mk], obin)
    cum = np.cumsum(values).astype(float) / np.cumsum(values).max()
    ax[pi[i]].plot(obin[1:], cum, label='L & R*', alpha=0.75,c='C4') # Plot all data on the first plot


    ax[pi[i]].set_title('{}'.format(titles[i]))
    n = mk.sum()
    KWstat, KWp =  ss.mstats.kruskalwallis(df_overlap_all[h+'overlap1_1'][mk].values, df_overlap_all[h+'overlap12'][mk].values, df_overlap_all[h+'overlap1_2'][mk].values, \
                            df_overlap_all[h+'overlap_12'][mk].values, df_overlap_all[h+'overlap_1_2'][mk].values)
    if KWp < 0.05:
        MWUstat12vs1_2, MWUp12vs1_2 = ss.mannwhitneyu(df_overlap_all[h+'overlap12'][mk].values, df_overlap_all[h+'overlap1_2'][mk].values)
        MWUstat_12vs_1_2, MWUp_12vs_1_2 = ss.mannwhitneyu(df_overlap_all[h+'overlap_12'][mk].values, df_overlap_all[h+'overlap_1_2'][mk].values)
    else:
        MWUstat12vs1_2, MWUp12vs1_2 = np.nan, np.nan
        MWUstat_12vs_1_2, MWUp_12vs_1_2 = np.nan, np.nan

    df_overlap_stats = pd.concat([df_overlap_stats, pd.DataFrame({'celltype':'single','segment':h, 'n':n, 'KWs':KWstat, 'KWp':KWp, 'p R & R* vs R & L*' : MWUp12vs1_2, 'p L & R* vs L & L*' : MWUp_12vs_1_2,}, index=[i])])

ax[0].set_ylabel('CDF',fontsize=6); ax[5].set_ylabel('CDF',fontsize=6); ax[10].set_ylabel('CDF',fontsize=6);
ax[10].set_xlabel('Overlap',fontsize=6); ax[11].set_xlabel('Overlap',fontsize=6); ax[12].set_xlabel('Overlap',fontsize=6); ax[13].set_xlabel('Overlap',fontsize=6); ax[14].set_xlabel('Overlap',fontsize=6);

# Directional cells
obin = np.linspace(0,1,100)
hds = ['fh_','fc_','mh_','lc_','lh_']
titles = ['First Hallway','First Corner','Middle Hallway','Last Corner','Last Hallway']
pi = [5, 6, 7, 8, 9]
for i, h in enumerate(hds):

    mk = np.logical_or(df_overlap_all[h+'type'] == 6, df_overlap_all[h+'type'] == 9)

    values, _ = np.histogram(df_overlap_all[h+'overlap1_1'][mk], obin)
    cum = np.cumsum(values).astype(float) / np.cumsum(values).max()
    ax[pi[i]].plot(obin[1:], cum, label='R & L', alpha=0.75, c='C1') # Plot all data on the first plot

    values, _ = np.histogram(df_overlap_all[h+'overlap12'][mk], obin)
    cum = np.cumsum(values).astype(float) / np.cumsum(values).max()
    ax[pi[i]].plot(obin[1:], cum, label='R & R*', alpha=0.75, c='C3') # Plot all data on the first plot

    values, _ = np.histogram(df_overlap_all[h+'overlap1_2'][mk], obin)
    cum = np.cumsum(values).astype(float) / np.cumsum(values).max()
    ax[pi[i]].plot(obin[1:], cum, label='R & L*', alpha=0.75, c='C2') # Plot all data on the first plot

    values, _ = np.histogram(df_overlap_all[h+'overlap_12'][mk], obin)
    cum = np.cumsum(values).astype(float) / np.cumsum(values).max()
    ax[pi[i]].plot(obin[1:], cum, label='L & R*', alpha=0.75, c='C4')# Plot all data on the first plot

    values, _ = np.histogram(df_overlap_all[h+'overlap_1_2'][mk], obin)
    cum = np.cumsum(values).astype(float) / np.cumsum(values).max()
    ax[pi[i]].plot(obin[1:], cum, label='L & L*', alpha=0.75, c='C0') # Plot all data on the first plot

    n = mk.sum()
    KWstat, KWp =  ss.mstats.kruskalwallis(df_overlap_all[h+'overlap1_1'][mk].values, df_overlap_all[h+'overlap12'][mk].values, df_overlap_all[h+'overlap1_2'][mk].values, \
                            df_overlap_all[h+'overlap_12'][mk].values, df_overlap_all[h+'overlap_1_2'][mk].values)
    if KWp < 0.05:
        MWUstat12vs1_2, MWUp12vs1_2 = ss.mannwhitneyu(df_overlap_all[h+'overlap12'][mk].values, df_overlap_all[h+'overlap1_2'][mk].values)
        MWUstat_12vs_1_2, MWUp_12vs_1_2 = ss.mannwhitneyu(df_overlap_all[h+'overlap_12'][mk].values, df_overlap_all[h+'overlap_1_2'][mk].values)
    else:
        MWUstat12vs1_2, MWUp12vs1_2 = np.nan, np.nan
        MWUstat_12vs_1_2, MWUp_12vs_1_2 = np.nan, np.nan

    df_overlap_stats = pd.concat([df_overlap_stats, pd.DataFrame({'celltype':'directional','segment':h, 'n':n, 'KWs':KWstat,  'KWp':KWp, 'p R & R* vs R & L*' : MWUp12vs1_2, 'p L & R* vs L & L*' : MWUp_12vs_1_2,}, index=[i])])


# Image cells
obin = np.linspace(0,1,100)
hds = ['fh_','mh_','lh_']
titles = ['First Hallway','Middle Hallway','Last Hallway']
pi = [10, 12, 14]
for i, h in enumerate(hds): # image cells

    mk = np.logical_or(df_overlap_all[h+'type'] == 5, df_overlap_all[h+'type'] == 10)
    values, _ = np.histogram(df_overlap_all[h+'overlap1_1'][mk], obin)
    cum = np.cumsum(values).astype(float) / np.cumsum(values).max()
    ax[pi[i]].plot(obin[1:], cum, label='rA_L', alpha=0.75, c='C1') # Plot all data on the first plot

    values, _ = np.histogram(df_overlap_all[h+'overlap12'][mk], obin)
    cum = np.cumsum(values).astype(float) / np.cumsum(values).max()
    ax[pi[i]].plot(obin[1:], cum, label='rA_rA*', alpha=0.75, c='C3') # Plot all data on the first plot

    values, _ = np.histogram(df_overlap_all[h+'overlap1_2'][mk], obin)
    cum = np.cumsum(values).astype(float) / np.cumsum(values).max()
    ax[pi[i]].plot(obin[1:], cum, label='rA_L*', alpha=0.75, c='C2') # Plot all data on the first plot

    values, _ = np.histogram(df_overlap_all[h+'overlap_12'][mk], obin)
    cum = np.cumsum(values).astype(float) / np.cumsum(values).max()
    ax[pi[i]].plot(obin[1:], cum, label='L_rA*', alpha=0.75, c='C4') # Plot all data on the first plot

    values, _ = np.histogram(df_overlap_all[h+'overlap_1_2'][mk], obin)
    cum = np.cumsum(values).astype(float) / np.cumsum(values).max()
    ax[pi[i]].plot(obin[1:], cum, label='L_L*', alpha=0.75, c='C0') # Plot all data on the first plot

    n = mk.sum()
    KWstat, KWp =  ss.mstats.kruskalwallis(df_overlap_all[h+'overlap1_1'][mk].values, df_overlap_all[h+'overlap12'][mk].values, df_overlap_all[h+'overlap1_2'][mk].values, \
                            df_overlap_all[h+'overlap_12'][mk].values, df_overlap_all[h+'overlap_1_2'][mk].values)
    if KWp < 0.05:
        MWUstat12vs1_2, MWUp12vs1_2 = ss.mannwhitneyu(df_overlap_all[h+'overlap12'][mk].values, df_overlap_all[h+'overlap1_2'][mk].values)
        MWUstat_12vs_1_2, MWUp_12vs_1_2 = ss.mannwhitneyu(df_overlap_all[h+'overlap_12'][mk].values, df_overlap_all[h+'overlap_1_2'][mk].values)
    else:
        MWUstat12vs1_2, MWUp12vs1_2 = np.nan, np.nan
        MWUstat_12vs_1_2, MWUp_12vs_1_2 = np.nan, np.nan

    df_overlap_stats = pd.concat([df_overlap_stats, pd.DataFrame({'celltype':'image_overlap','segment':h, 'n':n, 'KWs':KWstat, 'KWp':KWp, 'p R & R* vs R & L*' : MWUp12vs1_2, 'p L & R* vs L & L*' : MWUp_12vs_1_2,}, index=[i])])

ax[0].legend(fontsize=6)
for i in [11, 13]:
    ax[i].spines['top'].set_visible(False)
    ax[i].spines['right'].set_visible(False)
    ax[i].spines['left'].set_visible(False)
    ax[i].spines['bottom'].set_visible(False)
    ax[i].tick_params(axis=u'both', which=u'both',length=0)

df_overlap_stats.to_csv(combinedResultDir+'table_S3_overlap_KW_MWU_stats.csv')
pl.savefig(combinedResultDir+'fig_2BCD_rate_remap_overlap_summary_.pdf',format='pdf', bbox_inches = 'tight', pad_inches = 0.05)
pl.close()