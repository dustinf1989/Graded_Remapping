#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

This script generates all panels of Figure 3, 5B, S4, and Tables S1, S4-S6 in the manuscript
"Graded Remapping of Hippocampal Ensembles under Sensory Conflicts" written by
D. Fetterhoff, A. Sobolev & C. Leibold.

"""

def bin_spikes(spike_times,dt,wdw_start,wdw_end):
    """

    Function taken from [manuscript] (https://arxiv.org/abs/1708.00909)

    Function that puts spikes into bins

    Parameters
    ----------
    spike_times: an array of arrays
        an array of neurons. within each neuron's array is an array containing all the spike times of that neuron
    dt: number (any format)
        size of time bins
    wdw_start: number (any format)
        the start time for putting spikes in bins
    wdw_end: number (any format)
        the end time for putting spikes in bins

    Returns
    -------
    neural_data: a matrix of size "number of time bins" x "number of neurons"
        the number of spikes in each time bin for each neuron
    """
    edges=np.arange(wdw_start,wdw_end,dt) #Get edges of time bins
    num_bins=edges.shape[0]-1 #Number of bins
    num_neurons=spike_times.shape[0] #Number of neurons
    neural_data=np.empty([num_bins,num_neurons]) #Initialize array for binned neural data
    #Count number of spikes in each bin for each neuron, and put in array
    for i in range(num_neurons):
        neural_data[:,i]=np.histogram(spike_times[i],edges)[0]
    return neural_data

import os
import numpy as np
import matplotlib.pyplot as pl
import itertools as it
import scipy.stats as ss
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import ImageGrid
import scikit_posthocs
import seaborn as sns

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

#fileList = [fileList[11]] # Uncomment to test one file only; g2018d3 used for Fig 3A-E

toPlotNeuralPop = True # To plot the firing rate vectors for each maze type to ensure using the same data here as for population vector analysis
toPlotAllSessions = True # to plot figures for all sessions combined
totalMazeLength = 622 # cm, measured from physical virtual reality setup
speedThresh = 5 # cm/s, to discard spikes during stillness

#hdf5Dir = '/home/dustin/Documents/hdf5_v1/' # Load data from this folder
hdf5Dir = '/home/fetterhoff/atlas/RM_45max/combined_hdf5_from_raw_gsp2_removedInfSR/'

combinedResultDir = hdf5Dir+'reactivation_analysis/' # Save in subdirectory
if not os.path.exists(combinedResultDir):
    os.makedirs(combinedResultDir)

# Initialize to save data over all sessions
df_count = pd.DataFrame()
react_stats_df = pd.DataFrame() # Save the stats about the reactivation bins in a dataframe and as csv file

z_react11, z_react1_1, z_react12, z_react1_2 = np.array([]), np.array([]), np.array([]), np.array([])
z_react_11, z_react_1_1, z_react_12, z_react_1_2 = np.array([]), np.array([]), np.array([]), np.array([])
z_react21, z_react2_1, z_react22, z_react2_2 = np.array([]), np.array([]), np.array([]), np.array([])
z_react_21, z_react_2_1, z_react_22, z_react_2_2 = np.array([]), np.array([]), np.array([]), np.array([])

iti_react11, iti_react1_1, iti_react12, iti_react1_2 = np.array([]), np.array([]), np.array([]), np.array([])
iti_react_11, iti_react_1_1, iti_react_12, iti_react_1_2 = np.array([]), np.array([]), np.array([]), np.array([])
iti_react21, iti_react2_1, iti_react22, iti_react2_2 = np.array([]), np.array([]), np.array([]), np.array([])
iti_react_21, iti_react_2_1, iti_react_22, iti_react_2_2 = np.array([]), np.array([]), np.array([]), np.array([])

pl.rcParams.update({'font.size': 6, 'xtick.labelsize':6, 'ytick.labelsize':6, 'legend.fontsize':6, 'axes.facecolor':'white', 'lines.linewidth': 1.0, 'lines.markersize': 2.0, 'axes.labelsize': 6, 'figure.titlesize' : 6, 'axes.titlesize' : 'medium'})

#%% Loop through all sessions
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

    nPlaceFields = 0 # Count the number of place fields
    for i in spikeDF.FieldPeakLoc:
        nPlaceFields += len(i)

    # Table S1
    sumN = pd.DataFrame({'session': session, 'nPlaceCells' : len(spikeDF), 'nPlaceFields' : nPlaceFields}, index=[il])
    df_count = pd.concat([df_count, sumN])

    all_spike_times =  []
    for cell_id in spikeDF.T:
        all_spike_times.append(spikeDF.loc[cell_id].times)

    #%%Bin neural data using "bin_spikes" function
    t_start = trajDF.times.iloc[0] #Time to start extracting data - here the first time position was recorded
    t_end = trajDF.times.iloc[-1] + 20 #Time to finish extracting data
    dt = 0.1
    timeAx = np.arange(t_start-t_start, t_end-t_start, dt) # trajecotry time axis that starts at zero
    trajTimeAx = np.arange(t_start, t_end, dt)
    mazeTypeTimeAx = np.zeros(len(trajTimeAx)) # axis that matches binned time with maze-type labels
    speedTimeAx = np.zeros(len(trajTimeAx)) # axis that matches binned time with maze-type labels
    for i, tr in enumerate(trajTimeAx):
        chi = lapsDB[np.logical_and((lapsDB[:,0] - tr) <= 0, (lapsDB[:, 1] - tr) > 0), 3]
        itrajplace = np.abs(trajDF.times - tr).idxmin() # CHECK: swapped argmin for idxmin
        if len(chi) > 0:
            mazeTypeTimeAx[i] = int(chi[0])
            speedTimeAx[i] = trajDF.speed[int(itrajplace)]
    itiMask = mazeTypeTimeAx == 0
    spdMask = speedTimeAx > speedThresh

    # Use this to get the boolean array to make sure a neuron has a spike during each maze
    all_spike_times = np.array(all_spike_times)
    neural_data = bin_spikes(all_spike_times, dt, t_start, t_end+dt) # Add dt so length of axes match trajTime and mazeTypeTimeAx
#    run_neural_data = bin_spikes(all_run_spike_times, dt, t_start, t_end+dt) # Add dt so length of axes match trajTime and mazeTypeTimeAx

    neural_data1 = neural_data[spdMask & (mazeTypeTimeAx == 1)]
    neural_data_1 = neural_data[spdMask & (mazeTypeTimeAx == -1)]
    neural_data2 = neural_data[spdMask & (mazeTypeTimeAx == 2)]
    neural_data_2 = neural_data[spdMask & (mazeTypeTimeAx == -2)]

    neural_data_iti = neural_data[mazeTypeTimeAx == 0]
    z_neural_data_iti = ((neural_data_iti - neural_data_iti.mean(axis=0)) / neural_data_iti.std(axis=0))

    # Apply speed mask before getting the z-score on moving data from all mazes
    spd_neural_data = neural_data[spdMask, :] # add the speed mask - this vector has more data than when filtered for speed earlier due to speed downspampling
    z_neural_data = ((spd_neural_data - spd_neural_data.mean(axis=0)) / spd_neural_data.std(axis=0))

    z_neural_data1 = z_neural_data[mazeTypeTimeAx[spdMask] == 1, :]
    z_neural_data_1 = z_neural_data[mazeTypeTimeAx[spdMask] == -1, :]
    z_neural_data2 = z_neural_data[mazeTypeTimeAx[spdMask] == 2, :]
    z_neural_data_2 = z_neural_data[mazeTypeTimeAx[spdMask] == -2, :]

    def fill_zeros_with_last(arr):
        # fill zeros with the previous non-zero value
        prev = np.arange(len(arr))
        prev[arr == 0] = 0
        prev = np.maximum.accumulate(prev)
        return arr[prev]

    iti_mask_filled = fill_zeros_with_last(mazeTypeTimeAx)
    iti_mask_short = iti_mask_filled[mazeTypeTimeAx == 0]

    def get_reactivation_zScored(z_neur_dat):

        '''
        Input must already be z-scored based on entire session or dataset

        Analysis based on https://doi.org/10.1007/s10827-009-0154-6
        '''

        z_neur_data = z_neur_dat.T

        C = np.cov(z_neur_data)
        evals, D = np.linalg.eig(C)
        idx = evals.argsort()[::-1]
        evals = evals[idx]
        D = D[:, idx]
        q = float(z_neur_data.shape[1]) / float(z_neur_data.shape[0]) # how many time steps / how many neurons
        lambda_max = (1+np.sqrt(1/q))**2 # lamba max is the theorectical highest eignvalue. Encoding strength is eval/lambda_max = how much stronger is it?
        idsig = np.where(np.abs(evals) > lambda_max)[0]

        patterns = np.real(D[:, idsig]) # When doing cross-comparisions, can be different if some neurons never spike in a single maze
        reactivation = np.zeros([z_neur_data.shape[1], len(idsig)])
        for ii, pt in enumerate(patterns.T): #
            Q = np.dot(np.expand_dims(pt, 1), np.expand_dims(pt, 1).T)
            reactivation[:, ii] = np.sum(z_neur_data * np.dot(Q, z_neur_data), axis=0)
        return reactivation, idsig, evals, patterns

    _, idsig1, _, patterns1 = get_reactivation_zScored(z_neural_data1)
    _, idsig_1, _, patterns_1 = get_reactivation_zScored(z_neural_data_1)
    _, idsig2, _, patterns2 = get_reactivation_zScored(z_neural_data2)
    _, idsig_2, _, patterns_2 = get_reactivation_zScored(z_neural_data_2)

    # No idea why patterns_1 has imaginary components in 002018_2019-01-07
#    patterns_1 = np.real(patterns_1)

    #%% Get the reactivtion for each significant pattern
    sig_reactivation1 = np.zeros([z_neural_data.shape[0], len(idsig1)])
    sig_reactivation_iti1 = np.zeros([z_neural_data_iti.shape[0], len(idsig1)])
    for iin, nl in enumerate(patterns1.T): #
        Ql = np.dot(np.expand_dims(nl, 1), np.expand_dims(nl, 1).T) # Ql seems right except backwards
        sig_reactivation1[:, iin] = np.sum(z_neural_data.T * np.dot(Ql, z_neural_data.T), axis=0)
        z_react11 = np.append(z_react11, sig_reactivation1[mazeTypeTimeAx[spdMask] == 1, iin].mean())
        z_react1_1 = np.append(z_react1_1, sig_reactivation1[mazeTypeTimeAx[spdMask] == -1, iin].mean())
        z_react12 = np.append(z_react12, sig_reactivation1[mazeTypeTimeAx[spdMask] == 2, iin].mean())
        z_react1_2 = np.append(z_react1_2, sig_reactivation1[mazeTypeTimeAx[spdMask] == -2, iin].mean())

        sig_reactivation_iti1[:, iin] = np.sum(z_neural_data_iti.T * np.dot(Ql, z_neural_data_iti.T), axis=0)
        iti_react11 = np.append(iti_react11, sig_reactivation_iti1[iti_mask_short == 1, iin].mean())
        iti_react1_1 = np.append(iti_react1_1, sig_reactivation_iti1[iti_mask_short == -1, iin].mean())
        iti_react12 = np.append(iti_react12, sig_reactivation_iti1[iti_mask_short == 2, iin].mean())
        iti_react1_2 = np.append(iti_react1_2, sig_reactivation_iti1[iti_mask_short == -2, iin].mean())

    sig_reactivation_1 = np.zeros([z_neural_data.shape[0], len(idsig_1)])
    sig_reactivation_iti_1 = np.zeros([z_neural_data_iti.shape[0], len(idsig_1)])
    for iin, nl in enumerate(patterns_1.T): #
        Ql = np.dot(np.expand_dims(nl, 1), np.expand_dims(nl, 1).T) # Ql seems right except backwards
        sig_reactivation_1[:, iin] = np.sum(z_neural_data.T * np.dot(Ql, z_neural_data.T), axis=0)
        z_react_11 = np.append(z_react_11, sig_reactivation_1[mazeTypeTimeAx[spdMask] == 1, iin].mean())
        z_react_1_1 = np.append(z_react_1_1, sig_reactivation_1[mazeTypeTimeAx[spdMask] == -1, iin].mean())
        z_react_12 = np.append(z_react_12, sig_reactivation_1[mazeTypeTimeAx[spdMask] == 2, iin].mean())
        z_react_1_2 = np.append(z_react_1_2, sig_reactivation_1[mazeTypeTimeAx[spdMask] == -2, iin].mean())

        sig_reactivation_iti_1[:, iin] = np.sum( z_neural_data_iti.T * np.dot(Ql, z_neural_data_iti.T), axis=0)
        iti_react_11 = np.append(iti_react_11, sig_reactivation_iti_1[iti_mask_short== 1, iin].mean())
        iti_react_1_1 = np.append(iti_react_1_1, sig_reactivation_iti_1[iti_mask_short == -1, iin].mean())
        iti_react_12 = np.append(iti_react_12, sig_reactivation_iti_1[iti_mask_short == 2, iin].mean())
        iti_react_1_2 = np.append(iti_react_1_2, sig_reactivation_iti_1[iti_mask_short == -2, iin].mean())

    sig_reactivation2 = np.zeros([z_neural_data.shape[0], len(idsig2)])
    sig_reactivation_iti2 = np.zeros([z_neural_data_iti.shape[0], len(idsig2)])
    for iin, nl in enumerate(patterns2.T): #
        Ql = np.dot(np.expand_dims(nl, 1), np.expand_dims(nl, 1).T) # Ql seems right except backwards
        sig_reactivation2[:, iin] = np.sum(z_neural_data.T * np.dot(Ql, z_neural_data.T), axis=0)
        z_react21 = np.append(z_react21, sig_reactivation2[mazeTypeTimeAx[spdMask] == 1, iin].mean())
        z_react2_1 = np.append(z_react2_1, sig_reactivation2[mazeTypeTimeAx[spdMask] == -1, iin].mean())
        z_react22 = np.append(z_react22, sig_reactivation2[mazeTypeTimeAx[spdMask] == 2, iin].mean())
        z_react2_2 = np.append(z_react2_2, sig_reactivation2[mazeTypeTimeAx[spdMask] == -2, iin].mean())

        sig_reactivation_iti2[:, iin] = np.sum( z_neural_data_iti.T * np.dot(Ql, z_neural_data_iti.T), axis=0)
        iti_react21 = np.append(iti_react21, sig_reactivation_iti2[iti_mask_short == 1, iin].mean())
        iti_react2_1 = np.append(iti_react2_1, sig_reactivation_iti2[iti_mask_short == -1, iin].mean())
        iti_react22 = np.append(iti_react22, sig_reactivation_iti2[iti_mask_short == 2, iin].mean())
        iti_react2_2 = np.append(iti_react2_2, sig_reactivation_iti2[iti_mask_short == -2, iin].mean())

    sig_reactivation_2 = np.zeros([z_neural_data.shape[0], len(idsig_2)])
    sig_reactivation_iti_2 = np.zeros([z_neural_data_iti.shape[0], len(idsig_2)])
    for iin, nl in enumerate(patterns_2.T): #
        Ql = np.dot(np.expand_dims(nl, 1), np.expand_dims(nl, 1).T) # Ql seems right except backwards
        sig_reactivation_2[:, iin] = np.sum(z_neural_data.T * np.dot(Ql, z_neural_data.T), axis=0)
        z_react_21 = np.append(z_react_21, sig_reactivation_2[mazeTypeTimeAx[spdMask] == 1, iin].mean())
        z_react_2_1 = np.append(z_react_2_1, sig_reactivation_2[mazeTypeTimeAx[spdMask] == -1, iin].mean())
        z_react_22 = np.append(z_react_22, sig_reactivation_2[mazeTypeTimeAx[spdMask] == 2, iin].mean())
        z_react_2_2 = np.append(z_react_2_2, sig_reactivation_2[mazeTypeTimeAx[spdMask] == -2, iin].mean())

        sig_reactivation_iti_2[:, iin] = np.sum(z_neural_data_iti.T * np.dot(Ql, z_neural_data_iti.T), axis=0)
        iti_react_21 = np.append(iti_react_21, sig_reactivation_iti_2[iti_mask_short == 1, iin].mean())
        iti_react_2_1 = np.append(iti_react_2_1, sig_reactivation_iti_2[iti_mask_short == -1, iin].mean())
        iti_react_22 = np.append(iti_react_22, sig_reactivation_iti_2[iti_mask_short == -2, iin].mean())
        iti_react_2_2 = np.append(iti_react_2_2, sig_reactivation_iti_2[:, iin].mean())

    react_stats_df = pd.concat([react_stats_df, pd.DataFrame({'session' : session, 'nData' : len(trajTimeAx), 'nTask' : len(trajTimeAx[~itiMask]), 'nITI' : len(trajTimeAx[itiMask]),
                                                              'nR' : len(trajTimeAx[mazeTypeTimeAx == 1]), 'nL' : len(trajTimeAx[mazeTypeTimeAx == -1]), 'nR*': len(trajTimeAx[mazeTypeTimeAx == 2]),
                                                              'nL*' : len(trajTimeAx[mazeTypeTimeAx == -2]), 'sp_nTask' : len(trajTimeAx[(mazeTypeTimeAx != 0) & spdMask]), 'sp_nR' : len(trajTimeAx[(mazeTypeTimeAx == 1) & (spdMask)]),
                                                              'sp_nL' : len(trajTimeAx[(mazeTypeTimeAx == -1) & (spdMask)]), 'sp_nR*' : len(trajTimeAx[(mazeTypeTimeAx == 2) & (spdMask)]),
                                                              'sp_nL*' : len(trajTimeAx[(mazeTypeTimeAx == -2) & (spdMask)]),
                                                              'pct_nTask' : np.around(len(trajTimeAx[(mazeTypeTimeAx !=0) & (spdMask)]) / float(len(trajTimeAx[(mazeTypeTimeAx !=0)])), 3),
                                                              'pct_nR' : np.around(len(trajTimeAx[(mazeTypeTimeAx == 1) & (spdMask)]) / float(len(trajTimeAx[(mazeTypeTimeAx == 1)])), 3),
                                                              'pct_nL'  : np.around(len(trajTimeAx[(mazeTypeTimeAx == -1) & (spdMask)]) / float(len(trajTimeAx[(mazeTypeTimeAx == -1)])), 3),
                                                              'pct_nR*' : np.around(len(trajTimeAx[(mazeTypeTimeAx == 2) & (spdMask)]) / float(len(trajTimeAx[(mazeTypeTimeAx == 2)])), 3),
                                                              'pct_nL*' : np.around(len(trajTimeAx[(mazeTypeTimeAx == -2) & (spdMask)]) / float(len(trajTimeAx[(mazeTypeTimeAx == -2)])), 3),
                                                              'nSig_patt_R' : len(idsig1), 'nSig_patt_L' : len(idsig_1), 'nSig_patt_R*' : len(idsig2), 'nSig_patt_L*' : len(idsig_2)}, index=[il] )])

    #%% Fig 3B: Plot 20 second example firing

    ip = 3000
    fig, ax = pl.subplots(1,1,figsize=(1.3,2))
    im = ax.imshow(neural_data[ip:ip+200,:].T, 'Greys', extent=[timeAx[ip],timeAx[ip+200],0,neural_data.shape[1]], origin='lower')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="10%", pad=0.1)
    cb = pl.colorbar(im, cax=cax, ticks=[0, neural_data[ip:ip+200,:].max()])
    cb.set_label('# of Spikes', labelpad=-9)

    ax.set_xlabel('Seconds')
    ax.set_yticks([0,neural_data.shape[1]])
    ax.set_xticks([timeAx[ip], timeAx[ip+200]])
    ax.set_ylabel('Neuron #', labelpad=-10)
    pl.savefig(combinedResultDir+'fig_3B_example_firing_{}.pdf'.format(session),format='pdf', dpi=300, bbox_inches = 'tight', pad_inches = 0.05)
    pl.close()

    #%% Fig 3C: Show the patterns

    data = [patterns1, patterns_1, patterns2, patterns_2]
    nrows = [x.shape[0] for x in data]
    fig = pl.figure(figsize=(2.5, 1.9))
    axs = ImageGrid(fig, 111, nrows_ncols=(4, 1), axes_pad=0.1)
    for ix, d in enumerate(data):
        ax = axs[ix]
        im = ax.imshow(d.T, 'viridis', vmax=1, vmin=-1)
        _ = ax.yaxis.set_ticks([])
        _ = ax.xaxis.set_ticks([])
    axs[3].xaxis.set_ticks(np.arange(0, patterns1.shape[0], 10))
    axs[0].set_ylabel('R'); axs[1].set_ylabel('L')
    axs[2].set_ylabel('R*'); axs[3].set_ylabel('L*')
    axs[3].set_xlabel('Neuron #')
    axs[0].set_title('Patterns')

    fig.subplots_adjust(right = 0.95)
    cbar_ax = fig.add_axes([0.98, 0.125, 0.02, 0.755])
    fig.colorbar(im, cax=cbar_ax, ticks=[-1, 1]) # must match the vmin and vmax above
    cbar_ax.set_ylabel("Load (a.u.)", labelpad=-12)

    fig.savefig(combinedResultDir+'fig_3C_patterns_{}.pdf'.format(session),format='pdf', dpi=300, bbox_inches = 'tight', pad_inches = 0.05)
    pl.close(fig)

    #%% Publication quality pattern correlation matrices

    comb_pats = np.vstack((patterns1.T, patterns_1.T, patterns2.T, patterns_2.T))
    lenvector = np.array([0, patterns1.shape[1], patterns1.shape[1]+patterns_1.shape[1],
                          patterns1.shape[1]+patterns_1.shape[1]+patterns2.shape[1], comb_pats.shape[0]])

    midlen = [(a + b) / 2.0 for a, b in zip(lenvector[::], lenvector[1::])]
    allvec = np.array(sorted(np.concatenate((lenvector, midlen))))
    lb = [' ', 'R',' ', 'L', ' ', 'R*', ' ', 'L*', ' ']

    mat_r_patterns = np.zeros([comb_pats.shape[0] * comb_pats.shape[0]])
    mat_p_patterns = np.zeros([comb_pats.shape[0] * comb_pats.shape[0]])
    for i, (mat1, mat2) in enumerate(it.product(comb_pats, repeat=2)):
        mat_r_patterns[i], mat_p_patterns[i] = ss.pearsonr( np.real(mat1), np.real(mat2))
    mat_r_patterns = np.reshape(mat_r_patterns, (comb_pats.shape[0], comb_pats.shape[0]))
    mat_p_patterns = np.reshape(mat_p_patterns, (comb_pats.shape[0], comb_pats.shape[0]))

    fig, ax = pl.subplots(2, 1, figsize=(2.1, 3.0))
    ax = ax.ravel()
    posneg = np.sign(mat_r_patterns)
    sigfil = mat_p_patterns < 0.001
    posneg[~sigfil] = 0
    im = ax[0].imshow(mat_r_patterns, vmin=-1, vmax=1)
    im1 = ax[1].imshow(posneg, vmin=-1, vmax=1)

    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    pl.colorbar(im, cax = cax, ticks=[-1,1], label='r')
    cax.set_ylabel("r", labelpad=-13)

    divider = make_axes_locatable(ax[1])
    ca1 = divider.append_axes("right", size="5%", pad=0.05)
    pl.colorbar(im1, cax = ca1, ticks=[-1,1], label = 'p < 0.001')
    ca1.set_ylabel('p < 0.001', labelpad = -13)
#
    for i in range(2):
        ax[i].set_xticks(allvec-0.5)
        ax[i].set_xticklabels(lb)
        ax[i].set_yticks(allvec-0.5)
        ax[i].set_yticklabels(lb)
        ax[i].hlines(lenvector-0.5, -.5, comb_pats.shape[0]-0.5, alpha = 0.5)
        ax[i].vlines(lenvector-0.5, -.5, comb_pats.shape[0]-0.5, alpha = 0.5)
    ax[0].set_title('Pattern Correlation Matrix', pad=2)
    ax[1].set_title('Significant Correlations', pad=2)
    pl.tight_layout(h_pad=1)

    fig.savefig(combinedResultDir+'fig_3D_patterns_correlation_matrix_pub3_{}.pdf'.format(session), format='pdf', dpi=300, bbox_inches = 'tight', pad_inches = 0.05)
    pl.close(fig)

    #%% Publication quality pattern correlation matrices - SMALLER FOR SUPPLEMENT

    comb_pats = np.vstack((patterns1.T, patterns_1.T, patterns2.T, patterns_2.T))
    lenvector = np.array([0, patterns1.shape[1], patterns1.shape[1]+patterns_1.shape[1],
                          patterns1.shape[1]+patterns_1.shape[1]+patterns2.shape[1], comb_pats.shape[0]])

    midlen = [(a + b) / 2.0 for a, b in zip(lenvector[::], lenvector[1::])]
    allvec = np.array(sorted(np.concatenate((lenvector, midlen))))
    lb = [' ', 'R',' ', 'L', ' ', 'R*', ' ', 'L*', ' ']

    mat_r_patterns = np.zeros([comb_pats.shape[0] * comb_pats.shape[0]])
    mat_p_patterns = np.zeros([comb_pats.shape[0] * comb_pats.shape[0]])
    for i, (mat1, mat2) in enumerate(it.product(comb_pats, repeat=2)):
        mat_r_patterns[i], mat_p_patterns[i] = ss.pearsonr( np.real(mat1), np.real(mat2))
    mat_r_patterns = np.reshape(mat_r_patterns, (comb_pats.shape[0], comb_pats.shape[0]))
    mat_p_patterns = np.reshape(mat_p_patterns, (comb_pats.shape[0], comb_pats.shape[0]))

    fig, ax = pl.subplots(2, 1, figsize=(1.3, 2.0))
    ax = ax.ravel()
    posneg = np.sign(mat_r_patterns)
    sigfil = mat_p_patterns < 0.001
    posneg[~sigfil] = 0
    im = ax[0].imshow(mat_r_patterns, vmin=-1, vmax=1)
    im1 = ax[1].imshow(posneg, vmin=-1, vmax=1)

    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    pl.colorbar(im, cax = cax, ticks=[-1,1], label='r')
    cax.set_ylabel("r", labelpad=-13)

    divider = make_axes_locatable(ax[1])
    ca1 = divider.append_axes("right", size="5%", pad=0.05)
    pl.colorbar(im1, cax = ca1, ticks=[-1,1], label = 'p < 0.001')
    ca1.set_ylabel('p < 0.001', labelpad = -13)

    for i in range(2):
        ax[i].set_xticks(allvec-0.5)
        ax[i].set_xticklabels(lb)
        ax[i].set_yticks(allvec-0.5)
        ax[i].set_yticklabels(lb)
        ax[i].hlines(lenvector-0.5, -.5, comb_pats.shape[0]-0.5, alpha = 0.5)
        ax[i].vlines(lenvector-0.5, -.5, comb_pats.shape[0]-0.5, alpha = 0.5)
    ax[0].set_title('Pattern Correlations', pad=2)
    ax[1].set_title('Significance', pad=2)
    pl.tight_layout(h_pad=1)

    fig.savefig(combinedResultDir+'fig_S4_patterns_correlation_matrix_{}.pdf'.format(session), format='pdf', dpi=300, bbox_inches = 'tight', pad_inches = 0.05)
    pl.close(fig)

    #%% Fig 3E: Plot the reactivation for 2 cross patterns
    fig, ax = pl.subplots(ncols=4,nrows=2,figsize=(4.85,2.35),sharex=True, sharey=True)
    pl.tight_layout()

    for ie in range(2):
        ax[ie,0].plot(timeAx[spdMask], np.where(mazeTypeTimeAx[spdMask] == 1, sig_reactivation1[:,ie], 0),'r',label='R')
        ax[ie,0].plot(timeAx[spdMask], np.where(mazeTypeTimeAx[spdMask] == -1, sig_reactivation1[:,ie], 0),'b',label='L')
        ax[ie,0].plot(timeAx[spdMask], np.where(mazeTypeTimeAx[spdMask] == 2, sig_reactivation1[:,ie], 0),'m',label='R*')
        ax[ie,0].plot(timeAx[spdMask], np.where(mazeTypeTimeAx[spdMask] == -2, sig_reactivation1[:,ie], 0),'c',label='L*')

    for ie in range(2):
        ax[ie,1].plot(timeAx[spdMask], np.where(mazeTypeTimeAx[spdMask] == 1, sig_reactivation_1[:,ie], 0),'r',label='R')
        ax[ie,1].plot(timeAx[spdMask], np.where(mazeTypeTimeAx[spdMask] == -1, sig_reactivation_1[:,ie], 0),'b',label='L')
        ax[ie,1].plot(timeAx[spdMask], np.where(mazeTypeTimeAx[spdMask] == 2, sig_reactivation_1[:,ie], 0),'m',label='R*')
        ax[ie,1].plot(timeAx[spdMask], np.where(mazeTypeTimeAx[spdMask] == -2, sig_reactivation_1[:,ie], 0),'c',label='L*')

    for ie in range(2):
        ax[ie,2].plot(timeAx[spdMask], np.where(mazeTypeTimeAx[spdMask] == 1, sig_reactivation2[:,ie], 0),'r',label='R')
        ax[ie,2].plot(timeAx[spdMask], np.where(mazeTypeTimeAx[spdMask] == -1, sig_reactivation2[:,ie], 0),'b',label='L')
        ax[ie,2].plot(timeAx[spdMask], np.where(mazeTypeTimeAx[spdMask] == 2, sig_reactivation2[:,ie], 0),'m',label='R*')
        ax[ie,2].plot(timeAx[spdMask], np.where(mazeTypeTimeAx[spdMask] == -2, sig_reactivation2[:,ie], 0),'c',label='L*')

    for ie in range(2):
        ax[ie,3].plot(timeAx[spdMask], np.where(mazeTypeTimeAx[spdMask] == 1, sig_reactivation_2[:,ie], 0),'r',label='R')
        ax[ie,3].plot(timeAx[spdMask], np.where(mazeTypeTimeAx[spdMask] == -1, sig_reactivation_2[:,ie], 0),'b',label='L')
        ax[ie,3].plot(timeAx[spdMask], np.where(mazeTypeTimeAx[spdMask] == 2, sig_reactivation_2[:,ie], 0),'m',label='R*')
        ax[ie,3].plot(timeAx[spdMask], np.where(mazeTypeTimeAx[spdMask] == -2, sig_reactivation_2[:,ie], 0),'c',label='L*')

    for ll in range(4): # label only bottom x-axes
        ax[1,ll].set_xlabel('Seconds')
        ax[1,ll].set_xlim([0, timeAx[-1]])

    for i in range(2):
        for w in range(4):
            ax[i,w].spines['top'].set_visible(False)
            ax[i,w].spines['right'].set_visible(False)
            ax[i,w].tick_params()

    ax[0,0].set_title('Patterns from R')
    ax[0,1].set_title('Patterns from L')
    ax[0,2].set_title('Patterns from R*')
    ax[0,3].set_title('Patterns from L*')

    ax[0,0].set_ylabel('Reactivation Strength',  labelpad=0)
    ax[1,0].set_ylabel('Reactivation Strength',  labelpad=0)
    ax[0,0].yaxis.set_ticks([0,300])

    ax[0,2].legend()

    fig.savefig(combinedResultDir+'fig_3E_cross_reactivation_2_{}.pdf'.format(session),format='pdf', dpi=300, bbox_inches = 'tight', pad_inches = 0.05)
    pl.close(fig)

    #%% Fig 3A: Matrices of place cell firing in each maze-type
    if toPlotNeuralPop:
        # Load the place field data if it should be plotted
        f3 = hdf5Dir+session+'_resultsDB.h5'
        cellResultsDB = pd.read_hdf(f3, 'cellResultsDB')

        def remove_smallest_peaks(cid, cell_id_list, peak_list, maxRate_list):
            i_multiple_peaks = np.where(cell_id_list == cid)[0] # based on entire list
            i_biggest_peak = np.argmax(maxRate_list[i_multiple_peaks]) # based on list of peaks with same ID
            i_peaks_to_del = i_multiple_peaks[~i_biggest_peak]
            cell_id_list = np.delete(cell_id_list, i_peaks_to_del)
            peak_list = np.delete(peak_list, i_peaks_to_del)
            maxRate_list = np.delete(maxRate_list, i_peaks_to_del)
            return cell_id_list, peak_list, maxRate_list

        cell_id_list_2, peak_list_2 = np.array([]), np.array([])
        cell_id_list_1, peak_list_1 = np.array([]), np.array([])
        cell_id_list2, peak_list2 = np.array([]), np.array([])
        cell_id_list1, peak_list1 = np.array([]), np.array([])
        maxRate_list1, maxRate_list_1, maxRate_list2, maxRate_list_2 = np.array([]), np.array([]), np.array([]), np.array([])

        for cell_id in spikeDF.T:
            sp = spikeDF.loc[cell_id] # need to use stList so indices match with CellDB
            resultsDB = cellResultsDB[cellResultsDB['cell_id'] == cell_id]

            for i, mt in enumerate(sp.fieldMazeType):
                if (mt == -2):
                    cell_id_list_2, peak_list_2 = np.append(cell_id_list_2, cell_id), np.append(peak_list_2, sp.FieldPeakLoc[i])
                    maxRate_list_2 = np.append(maxRate_list_2, sp.maxFieldRate[i])
                elif (mt == -1):
                    cell_id_list_1, peak_list_1 = np.append(cell_id_list_1, cell_id), np.append(peak_list_1, sp.FieldPeakLoc[i])
                    maxRate_list_1 = np.append(maxRate_list_1, sp.maxFieldRate[i])
                elif (mt == 2):
                    cell_id_list2, peak_list2 = np.append(cell_id_list2, cell_id), np.append(peak_list2, sp.FieldPeakLoc[i])
                    maxRate_list2 = np.append(maxRate_list2, sp.maxFieldRate[i])
                elif (mt == 1):
                    cell_id_list1, peak_list1 = np.append(cell_id_list1, cell_id), np.append(peak_list1, sp.FieldPeakLoc[i])
                    maxRate_list1 = np.append(maxRate_list1, sp.maxFieldRate[i])

            # If a neuron has more than one peak in the same maze-type, only keep the largest one.
            for i in range(4):
                if np.sum(cell_id_list1 == cell_id) > 1:
                    cell_id_list1, peak_list1, maxRate_list1 = remove_smallest_peaks(cell_id, cell_id_list1, peak_list1, maxRate_list1)
                if np.sum(cell_id_list_1 == cell_id) > 1:
                    cell_id_list_1, peak_list_1,maxRate_list_1 = remove_smallest_peaks(cell_id, cell_id_list_1, peak_list_1, maxRate_list_1)
                if np.sum(cell_id_list2 == cell_id) > 1:
                    cell_id_list2, peak_list2, maxRate_list2 = remove_smallest_peaks(cell_id, cell_id_list2, peak_list2, maxRate_list2)
                if np.sum(cell_id_list_2 == cell_id) > 1:
                    cell_id_list_2, peak_list_2, maxRate_list_2 = remove_smallest_peaks(cell_id, cell_id_list_2, peak_list_2, maxRate_list_2)

        mt=1 # maze-type to plot
        peaksort1 = cell_id_list1[peak_list1.argsort()]
        matrix1 = np.zeros( [len(peaksort1), np.sum(np.logical_and(cellResultsDB.choice==mt, cellResultsDB.cell_id==peaksort1[0])) ] )
        for iii, item in enumerate(peaksort1):
            resultsDB = cellResultsDB[cellResultsDB['cell_id'] == item]
            matrix1[iii] = resultsDB[resultsDB.choice==mt].normRate / resultsDB[resultsDB.choice==mt].normRate.max()

        mt=-1 # maze-type to plot
        peaksort_1 = cell_id_list_1[peak_list_1.argsort()]
        matrix_1 = np.zeros( [len(peaksort_1), np.sum(np.logical_and(cellResultsDB.choice==mt, cellResultsDB.cell_id==peaksort_1[0])) ] )
        for iii, item in  enumerate(peaksort_1):
            resultsDB = cellResultsDB[cellResultsDB['cell_id'] == item]
            matrix_1[iii] = resultsDB[resultsDB.choice==mt].normRate / resultsDB[resultsDB.choice==mt].normRate.max()

        mt=2 # maze-type to plot
        peaksort2 = cell_id_list2[peak_list2.argsort()]
        matrix2 = np.zeros( [len(peaksort2), np.sum(np.logical_and(cellResultsDB.choice==mt, cellResultsDB.cell_id==peaksort2[0])) ] )
        for iii, item in  enumerate(peaksort2):
            resultsDB = cellResultsDB[cellResultsDB['cell_id'] == item]
            matrix2[iii] = resultsDB[resultsDB.choice==mt].normRate / resultsDB[resultsDB.choice==mt].normRate.max()

        mt=-2 # maze-type to plot
        peaksort_2 = cell_id_list_2[peak_list_2.argsort()]
        matrix_2 = np.zeros( [len(peaksort_2), np.sum(np.logical_and(cellResultsDB.choice==mt, cellResultsDB.cell_id==peaksort_2[0])) ] )
        for iii, item in  enumerate(peaksort_2):
            resultsDB = cellResultsDB[cellResultsDB['cell_id'] == item]
            matrix_2[iii] = resultsDB[resultsDB.choice==mt].normRate / resultsDB[resultsDB.choice==mt].normRate.max()

        # Place field matrix maps
        pl.figure(figsize=(3.5,2))
        axm1, axm2, axm3, axm4 = pl.subplot(2,2,1), pl.subplot(2,2,2), pl.subplot(2,2,3), pl.subplot(2,2,4)
        axm1.imshow(matrix1, cmap='viridis', aspect='auto', extent=[0,totalMazeLength,0,len(matrix1)])
        axm1.set_title('R, n={} place cells'.format(len(peaksort1)), pad=2)
        axm2.imshow(matrix_1, cmap='viridis', aspect='auto', extent=[0,totalMazeLength,0,len(matrix_1)])
        axm2.set_title('L, n={} place cells'.format(len(peaksort_1)), pad=2)
        axm3.imshow(matrix2, cmap='viridis', aspect='auto', extent=[0,totalMazeLength,0,len(matrix2)])
        axm3.set_title('R*, n={} place cells'.format(len(peaksort2)), pad=2)
        axm4.imshow(matrix_2, cmap='viridis', aspect='auto', extent=[0,totalMazeLength,0,len(matrix_2)])
        axm4.set_title('L*, n={} place cells'.format(len(peaksort_2)), pad=2)
        axm1.grid(False); axm2.grid(False); axm3.grid(False); axm4.grid(False)
        axm1.set_xticks([]); axm2.set_xticks([])
        axm3.set_xlabel('Track Position (cm)'); axm4.set_xlabel('Track Position (cm)')
        axm1.set_ylabel('Neuron #'); axm3.set_ylabel('Neuron #')

        pl.savefig(combinedResultDir+'fig_3A_neural_pop_viridis_pca_{}.pdf'.format(session),format='pdf', dpi=300, bbox_inches = 'tight', pad_inches = 0.05)
        pl.close()

#%% Analysis for all sessions

df_count.to_csv(combinedResultDir+'table_S1_place_cell_counts_reactivation.csv')

if toPlotAllSessions:

    ses = react_stats_df['session']
    react_stats_df.drop(labels=['session'], axis=1, inplace=True)
    react_stats_df.insert(0, 'session', ses)
    react_stats_df.to_csv(combinedResultDir+'pca_stats.csv', index=False, encoding='utf-8-sig')

    #%% Fig 3F: Matrix plot of reactivation
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

    df_ci_react = pd.DataFrame([])

    yy = [z_react11, z_react11,  z_react11, z_react11, z_react_1_1, z_react_1_1, z_react_1_1, z_react_1_1, z_react22, z_react22, z_react22, z_react22, z_react_2_2, z_react_2_2, z_react_2_2, z_react_2_2]
    xx = [z_react11, z_react1_1, z_react12, z_react1_2, z_react_11,  z_react_1_1, z_react_12, z_react_1_2, z_react21, z_react2_1, z_react22, z_react2_2, z_react_21, z_react_2_1, z_react_22, z_react_2_2]
    mx = np.ceil(np.max(np.concatenate((z_react11, z_react_1_1, z_react22, z_react_2_2)))) # Top boundary for plots

    fig, ax = pl.subplots(nrows=4, ncols=4, figsize=(4,4),sharex=True, sharey=True)
    ax = ax.ravel()

    # Create one larger axis for x & y axis labels
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    pl.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    pl.xlabel("Reactivation Strength from Data", labelpad=5)
    pl.ylabel("Reactivation Strength from Patterns", labelpad=5)


    for i, (xi, yi) in enumerate(zip(xx,yy)):
        if i not in [0,5,10,15]:
            ax[i].set(adjustable='box', aspect='equal')
            ax[i].tick_params(axis=u'both', which=u'both',length=0)

            ri, pr = ss.pearsonr(xi, yi)
            CI_r = r_confidence_interval(ri, len(xi))

            cidict = pd.DataFrame({'n':len(xi), 'CI_lo':CI_r[0],'CI_hi':CI_r[1]},index=[i])
            df_ci_react = pd.concat([df_ci_react, cidict])

            ri, pr = np.around(ri,2), np.around(pr,9)

            ax[i].plot([0,mx],[0,mx],'k',alpha=0.25)
            ax[i].scatter(xi,yi, marker='.', alpha=0.6,color='C1', s=2)

            m, b = np.polyfit(xi, yi, 1)
            X_plot = np.linspace(0,mx,100)
            ax[i].plot(X_plot, m*X_plot + b, '-.', color='C0',alpha=0.75)
            ax[i].set_title('r = {}'.format(ri), pad=1)

            if pr < 0.01:
                ax[i].text(0.15,0.88,'p = {0:1.1e}'.format(pr),transform=ax[i].transAxes)
            else:
                ax[i].text(0.18,0.88,'p = {}'.format(np.around(pr,3)),transform=ax[i].transAxes)

            ax[i].set_xlim([0,mx])
            ax[i].set_ylim([0,mx])
            ax[i].set_yticks([0,mx])
            ax[i].set_xticks([0,mx])
        else:
            ax[i].spines['top'].set_visible(False)
            ax[i].spines['right'].set_visible(False)
            ax[i].spines['left'].set_visible(False)
            ax[i].spines['bottom'].set_visible(False)
            ax[i].tick_params(axis=u'both', which=u'both',length=0)

    ax[0].set_ylabel('R')
    ax[4].set_ylabel('L')
    ax[8].set_ylabel('R*')
    ax[12].set_ylabel('L*')

    ax[12].set_xlabel('R', labelpad=0)
    ax[13].set_xlabel('L', labelpad=0)
    ax[14].set_xlabel('R*', labelpad=0)
    ax[15].set_xlabel('L*',labelpad=0)

    fig.savefig(combinedResultDir+'fig_3F_allSessions_zReact_pca_reactivation_matrix.pdf',format='pdf', dpi=300, bbox_inches = 'tight', pad_inches = 0.05)
    pl.close(fig)

    df_ci_react.to_csv(combinedResultDir+'table_s4_ci_zreact.csv')

    #%% Fig 3G: Box plots of reactivation strength on Track

    pl.rcParams.update({'ytick.major.size': 1.5, 'xtick.major.size': 1.5})

    sns.color_palette("bright")
    fig, ax = pl.subplots(4,1, figsize=(0.9,4.0),sharex=True, sharey=True)
    ax = ax.ravel()

    # Create on larger axis for y-label and hide it
    fig.add_subplot(111, frameon=False)
    pl.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    pl.ylabel('Reactivation Strength on Track', labelpad=5)

    mz = ['R', 'L', 'R*', 'L*']
    pal = {'R':'red', 'L':'blue', 'R*':'magenta', 'L*':'cyan'}
    flierprops = dict(markersize=1, linestyle='none', marker='o', alpha=0.5)

    sns.boxplot(x=mz, y =[z_react11, z_react1_1, z_react12, z_react1_2], ax=ax[0], width=0.3, palette= pal, linewidth=0.5, flierprops=flierprops)
    sns.boxplot(x=mz, y =[z_react_11, z_react_1_1, z_react_12, z_react_1_2], ax=ax[1], width=0.3, palette= pal, linewidth=0.5, flierprops=flierprops)
    sns.boxplot(x=mz, y =[z_react21, z_react2_1, z_react22, z_react2_2], ax=ax[2], width=0.3, palette= pal, linewidth=0.5, flierprops=flierprops)
    sns.boxplot(x=mz, y =[z_react_21, z_react_2_1, z_react_22, z_react_2_2], ax=ax[3], width=0.3, palette= pal, linewidth=0.5, flierprops=flierprops)

    ax[0].set_title('R Patterns',pad=1)
    ax[1].set_title('L Patterns',pad=1)
    ax[2].set_title('R* Patterns',pad=1)
    ax[3].set_title('L* Patterns',pad=1)

    fig.savefig(combinedResultDir+'fig_3G_allSessions_react_box_track.pdf',format='pdf', dpi=300, bbox_inches = 'tight', pad_inches = 0.05)
    pl.close(fig)

    #%% Table S5

    # Statistical tests: Kruskal-Wallis
    ss.kruskal(z_react11, z_react1_1, z_react12, z_react1_2)
    ss.kruskal(z_react_11, z_react_1_1, z_react_12, z_react_1_2)
    ss.kruskal(z_react21, z_react2_1, z_react22, z_react2_2)
    ss.kruskal(z_react_21, z_react_2_1, z_react_22, z_react_2_2)

    # Dunn's Posthoc tests
    a=scikit_posthocs.posthoc_dunn([z_react11, z_react1_1, z_react12, z_react1_2], p_adjust='bonferroni')
    b=scikit_posthocs.posthoc_dunn([z_react_11, z_react_1_1, z_react_12, z_react_1_2], p_adjust='bonferroni')
    c=scikit_posthocs.posthoc_dunn([z_react21, z_react2_1, z_react22, z_react2_2], p_adjust='bonferroni')
    d=scikit_posthocs.posthoc_dunn([z_react_21, z_react_2_1, z_react_22, z_react_2_2], p_adjust='bonferroni')

    #%% Fig 5B: Box plots of reactivation strength during ITI

    fig, ax = pl.subplots(1,4, figsize=(4.88,1.1),sharex=True, sharey=True)
    ax = ax.ravel()

    mz = ['R', 'L', 'R*', 'L*'] # maze-types
    pal = {'R':'red', 'L':'blue', 'R*':'magenta', 'L*':'cyan'} # colors
    flierprops = dict(markersize=1, linestyle='none', marker='o', alpha=0.5)

    sns.boxplot(x=mz, y =[iti_react11, iti_react1_1, iti_react12, iti_react1_2], ax=ax[0], width=0.3 , palette= pal, linewidth=0.5, flierprops=flierprops)
    sns.boxplot(x=mz, y =[iti_react_11, iti_react_1_1, iti_react_12, iti_react_1_2], ax=ax[1], width=0.3, palette= pal, linewidth=0.5, flierprops=flierprops)
    sns.boxplot(x=mz, y =[iti_react21, iti_react2_1, iti_react22, iti_react2_2], ax=ax[2], width=0.3, palette= pal, linewidth=0.5, flierprops=flierprops)
    sns.boxplot(x=mz, y =[iti_react_21, iti_react_2_1, iti_react_22, iti_react_2_2], ax=ax[3], width=0.3 , palette= pal, linewidth=0.5, flierprops=flierprops)

    for i, s in enumerate(mz):
        ax[i].set_title('{} Patterns'.format(s), pad=1)
    ax[0].set_ylim([-0.2,6.5])
    ax[0].set_ylabel('Reactivation Strength: ITI')

    fig.savefig(combinedResultDir+'fig_5B_allSessions_react_box_iti.pdf',format='pdf', dpi=300, bbox_inches = 'tight', pad_inches = 0.05)
    pl.close(fig)

    #%% Table S6

    # Statistical tests: Kruskal-Wallis
    ss.kruskal(iti_react11, iti_react1_1, iti_react12, iti_react1_2)
    ss.kruskal(iti_react_11, iti_react_1_1, iti_react_12, iti_react_1_2)
    ss.kruskal(iti_react21, iti_react2_1, iti_react22, iti_react2_2)
    ss.kruskal(iti_react_21, iti_react_2_1, iti_react_22, iti_react_2_2)

    # Dunn's Posthoc tests
    f=scikit_posthocs.posthoc_dunn([iti_react21, iti_react2_1, iti_react22, iti_react2_2], p_adjust='bonferroni')
    g=scikit_posthocs.posthoc_dunn([iti_react_21, iti_react_2_1, iti_react_22, iti_react_2_2], p_adjust='bonferroni')