#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This script generates all panels of Figure 3 in the manuscript
"Graded Remapping of Hippocampal Ensembles under Sensory Conflicts" written by
D. Fetterhoff, A. Sobolev & C. Leibold.

All analysis code was written by D. Fetterhoff

"""

import os
import glob
import numpy as np
import matplotlib.pyplot as pl
import pandas as pd
from scipy.io import loadmat

def bin_spikes(spike_times, dT, wdw_start, wdw_end):
    """

    Function taken from [manuscript] (https://arxiv.org/abs/1708.00909)

    Function that puts spikes into bins

    Parameters
    ----------
    spike_times: an array of arrays
        an array of neurons. within each neuron's array is an array containing all the spike times of that neuron
    dT: number (any format)
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
    edges = np.arange(wdw_start, wdw_end, dT) #Get edges of time bins
    num_bins = edges.shape[0]-1 #Number of bins
    num_neurons = spike_times.shape[0] #Number of neurons
    neural_data = np.empty([num_bins, num_neurons]) #Initialize array for binned neural data
    #Count number of spikes in each bin for each neuron, and put in array
    for i_ in range(num_neurons):
        neural_data[:, i_] = np.histogram(spike_times[i_], edges)[0]
    return neural_data

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

#fileList = [fileList[10]] # Uncomment to test one file only; 10=g2018_d2 used for Fig. 3A

toPlotAllSessions = True # to plot figures for all sessions combined
speedThresh = 5 # cm/s, to discard spikes during stillness
bd = [187.5, 275, 412.5, 500] # boundaries for all maze segments in cm
bins = np.arange(-0.5, 4.5, 1) # Used to determine MLE from log-likelihoods
bins_ex = np.arange(-0.5, 3.5, 1) # Use different bins to exclude the same segment
gamma = 0.005 # regulizer parameter added to eignvalues

# Names used for plotting
mazeTypeList = ['R', 'L', 'R*', 'L*']
colors = ('r', 'b', 'm', 'c') # Colors for each maze-type
mazeSegList = ['Entire Maze', 'First  Hallway', 'First Corner', 'Middle Hallway', 'Last Corner', 'Last Hallway']

# Load data from this folder
hdf5Dir = '/home/fetterhoff/Graded_Remapping/'

# Create a results subfolder inside the data folder
combinedResultDir = hdf5Dir+'mle_results_{}gamma/'.format(gamma) # Save in subdirectory
if not os.path.exists(combinedResultDir):
    os.makedirs(combinedResultDir)

# Initialize to save results over all sessions
df_count = pd.DataFrame()

# MLE percent matrices with all 4 maze-types
mle_sess = np.zeros([len(fileList), len(mazeTypeList), len(mazeTypeList)])
mle_sess_fh = np.zeros([len(fileList), len(mazeTypeList), len(mazeTypeList)]) # First hallway
mle_sess_fc = np.zeros([len(fileList), len(mazeTypeList), len(mazeTypeList)]) # First corner
mle_sess_mh = np.zeros([len(fileList), len(mazeTypeList), len(mazeTypeList)]) # Middle Hallway
mle_sess_lc = np.zeros([len(fileList), len(mazeTypeList), len(mazeTypeList)]) # Last Corner
mle_sess_lh = np.zeros([len(fileList), len(mazeTypeList), len(mazeTypeList)]) # Last Hallway

# MLE percent matrices when excluding the source = pattern comparisions
mle_ex_sess = np.zeros([len(fileList), len(mazeTypeList), len(mazeTypeList)])
mle_ex_sess_fh = np.zeros([len(fileList), len(mazeTypeList), len(mazeTypeList)])
mle_ex_sess_fc = np.zeros([len(fileList), len(mazeTypeList), len(mazeTypeList)])
mle_ex_sess_mh = np.zeros([len(fileList), len(mazeTypeList), len(mazeTypeList)])
mle_ex_sess_lc = np.zeros([len(fileList), len(mazeTypeList), len(mazeTypeList)])
mle_ex_sess_lh = np.zeros([len(fileList), len(mazeTypeList), len(mazeTypeList)])

# Standardize many plotting parmeters
pl.rcParams.update({'font.size': 6, 'xtick.labelsize':6, 'ytick.labelsize':6, 'legend.fontsize':6, 'axes.facecolor':'white', 'lines.linewidth': 1.0, 'lines.markersize': 2.0, 'axes.labelsize': 6, 'figure.titlesize' : 6, 'axes.titlesize' : 'medium'})

#%% Loop through all sessions
for il, s in enumerate(fileList):
    session = s[0]
    print(session) # current session

    sd = hdf5Dir+session+'/' # session directory

    # Build a DataFrame using all tetrode (TT) files
    spikeDF = pd.DataFrame()
    for mat_name in glob.glob(sd+'*TT*.mat'):
        m = loadmat(mat_name)

        frame = pd.DataFrame([[m['file'][0], m['times'][0], m['vr_x'][0], m['vr_y'][0], m['real_cm'][0], m['speed_cms'][0], m['lap_num'][0],
                               m['maze_type'][0], m['spatial_info_index'][0], m['spatial_info'][0], m['numFieldSpikes'][0], m['maxFieldRate'][0],
                               m['fieldMazeType'][0], m['FieldPeakLoc'][0], m['segment_types'][0], m['spike_ratio'][0]]],
                             columns=['file', 'times', 'vr_x', 'vr_y', 'real_cm', 'speed_cms', 'lap_num', 'maze_type', 'spatial_info_index', 'spatial_info',
                                      'numFieldSpikes', 'maxFieldRate', 'fieldMazeType', 'FieldPeakLoc', 'segment_types', 'spike_ratio'], index=m['ni'][0])
        spikeDF = spikeDF.append(frame)
    spikeDF.sort_index(inplace=True)

    f2 = sd+session+'_laps_traj.h5'
    trajDF = pd.read_hdf(f2, 'trj') # DataFrame of times/places/speed for each lap in VR
    # LapsDF maze_type dictionary: {1:R, -1:L, 2: R*, -2: L*}
    lapsDF = pd.read_hdf(f2, 'lapsDF')
    lapsDB = np.array(lapsDF) # Keep values as matrix

    nPlaceFields = 0 # Count the number of place fields
    for i in spikeDF.FieldPeakLoc:
        nPlaceFields += len(i)

    # Table S1: Cell and field counts per session
    sumN = pd.DataFrame({'session': session, 'nPlaceCells' : len(spikeDF), 'nPlaceFields' : nPlaceFields}, index=[il])
    df_count = pd.concat([df_count, sumN])

    #%% Setup data structures before MLE. Create some time axes and z-score neural data for each maze-type separately.

    t_start = trajDF.times.iloc[0] # Time to start extracting data - here the first time position was recorded
    t_end = trajDF.times.iloc[-1] + 20 # Time to finish extracting data
    dt = 0.1 # seconds = 100ms time bins
    timeAx = np.arange(t_start-t_start, t_end-t_start, dt) # trajecotry time axis that starts at zero
    trajTimeAx = np.arange(t_start, t_end, dt) # trajectory time axis starting at recorded trajectory times
    mazeTypeTimeAx = np.zeros(len(trajTimeAx)) # axis that matches binned time with maze-type labels {1:R, -1:L, 2: R*, -2: L*}
    speedTimeAx = np.zeros(len(trajTimeAx)) # axis that matches binned time with speed in cm/s
    placeTimeAx = np.zeros(len(trajTimeAx)) # position samples to match time axis

    # Downsample vectors to match trajectory time axis using lapsDB
    for i, tr in enumerate(trajTimeAx):
        imt = lapsDB[np.logical_and((lapsDB[:, 0] - tr) <= 0, (lapsDB[:, 1] - tr) > 0), 3] # index for maze-type
        itrj = np.abs(trajDF.times - tr).idxmin() # index of the smallest time difference in traj time axis
        if len(imt): # enter if time bin occurred during the maze
            mazeTypeTimeAx[i] = int(imt[0])
            speedTimeAx[i] = trajDF.speed[int(itrj)]
            placeTimeAx[i] = trajDF.places_cm[int(itrj)]
    itiMask = mazeTypeTimeAx == 0 # True when timesteps were during ITI
    spdMask = speedTimeAx > speedThresh

    # Divide data in each maze-type by the maze segment using masks for all maze segments
    fhr = (placeTimeAx < bd[0]) & ~itiMask # first hallway
    fcr = (placeTimeAx >= bd[0]) & (placeTimeAx < bd[1]) & ~itiMask # first corner
    mhr = (placeTimeAx >= bd[1]) & (placeTimeAx < bd[2]) & ~itiMask # middle hallway
    lcr = (placeTimeAx >= bd[2]) & (placeTimeAx < bd[3]) & ~itiMask # last corner
    lhr = (placeTimeAx >= bd[3]) & ~itiMask # last hallway

    # mazeSegTimeAx specifies maze segment where 1=first hall, 2=first corner, 3=middle hallway, 4=last corner, 5=last hallway, 0=ITI
    mazeSegTimeAx = np.zeros(len(placeTimeAx))
    for i, h in enumerate([fhr, fcr, mhr, lcr, lhr]):
        mazeSegTimeAx[h == 1] = i+1

    # Bin all neuronal data from all place cells
    all_spike_times = []
    for cell_id in spikeDF.T:
        all_spike_times.append(spikeDF.loc[cell_id].times)
    all_spike_times = np.array(all_spike_times, dtype='object')
    neur_data = bin_spikes(all_spike_times, dt, t_start, t_end+dt) # Add dt so length of axes match trajTime and mazeTypeTimeAx

    # Apply speed mask before getting the z-score on moving data from all mazes
    spd_neural_data = neur_data[spdMask, :] # add the speed mask - this vector has more data than when filtered for speed earlier due to speed downspampling
    mazeSegTimeAxSpd = mazeSegTimeAx[spdMask]

    # One vector for each maze-type that specifies the maze segment for each timestep
    # Contents of each vector represent maze segment using same dictionary as mazeSegTimeAx
    mazeSeg1 = mazeSegTimeAxSpd[mazeTypeTimeAx[spdMask] == 1] # maze-type R
    mazeSeg_1 = mazeSegTimeAxSpd[mazeTypeTimeAx[spdMask] == -1] # maze-type L
    mazeSeg2 = mazeSegTimeAxSpd[mazeTypeTimeAx[spdMask] == 2] # maze-type R*
    mazeSeg_2 = mazeSegTimeAxSpd[mazeTypeTimeAx[spdMask] == -2] # maze-type L*

    # Ignore the divide by zero warnings that occur if neurons are silent. Silent neurons removed in mle function.
    np.seterr(divide='ignore', invalid='ignore')

    # z-score neural data for each maze individually
    z_neural_data1 = spd_neural_data[mazeTypeTimeAx[spdMask] == 1, :] # apply speed threshold
    z_neural_data1 = ((z_neural_data1 - z_neural_data1.mean(axis=0)) / z_neural_data1.std(axis=0))
    z_neural_data_1 = spd_neural_data[mazeTypeTimeAx[spdMask] == -1, :]
    z_neural_data_1 = ((z_neural_data_1 - z_neural_data_1.mean(axis=0)) / z_neural_data_1.std(axis=0))

    z_neural_data2 = spd_neural_data[mazeTypeTimeAx[spdMask] == 2, :]
    z_neural_data2 = ((z_neural_data2 - z_neural_data2.mean(axis=0)) / z_neural_data2.std(axis=0))
    z_neural_data_2 = spd_neural_data[mazeTypeTimeAx[spdMask] == -2, :]
    z_neural_data_2 = ((z_neural_data_2 - z_neural_data_2.mean(axis=0)) / z_neural_data_2.std(axis=0))

    #%% Log-Likelihood
    def get_log_likelihood(z_source_dat, z_pattern_dat, Gamma):

        '''
        Computes Gaussian Likelihood
        Inputs must already be z-scored based on dataset
        z_neur_dat.shape[0] / z_neur_dat.shape[1]
        how many time steps / how many neurons
        Gamma (small) is a regularization parameter
        '''

        z_source_data = z_source_dat.T
        z_pattern_data = z_pattern_dat.T

        # filter out neurons if they are silent during either matrix
        sil_fil = np.logical_and(~np.isnan(z_pattern_data.sum(axis=1)), ~np.isnan(z_source_data.sum(axis=1)))
        z_pattern_data = z_pattern_data[sil_fil]
        z_source_data = z_source_data[sil_fil]

        C = np.cov(z_pattern_data) # covariance matrix of pattern data
        evals, D = np.linalg.eig(C) # Diagonalization
        evals = evals+Gamma # add regularization parameter
        patterns = np.real(D) # Remove complex compnents

        Q = np.dot(patterns, np.dot(np.diag(1/evals), np.linalg.inv(patterns)))

        loglike = -0.5*np.sum(z_source_data * np.dot(Q, z_source_data), axis=0) - 0.5*np.sum(np.log(evals))

        return loglike

    # Compute log-likelihood for each source and pattern data combinations
    loglike11 = get_log_likelihood(z_neural_data1, z_neural_data1, gamma)
    loglike1_1 = get_log_likelihood(z_neural_data1, z_neural_data_1, gamma)
    loglike12 = get_log_likelihood(z_neural_data1, z_neural_data2, gamma)
    loglike1_2 = get_log_likelihood(z_neural_data1, z_neural_data_2, gamma)

    loglike_11 = get_log_likelihood(z_neural_data_1, z_neural_data1, gamma)
    loglike_1_1 = get_log_likelihood(z_neural_data_1, z_neural_data_1, gamma)
    loglike_12 = get_log_likelihood(z_neural_data_1, z_neural_data2, gamma)
    loglike_1_2 = get_log_likelihood(z_neural_data_1, z_neural_data_2, gamma)

    loglike21 = get_log_likelihood(z_neural_data2, z_neural_data1, gamma)
    loglike2_1 = get_log_likelihood(z_neural_data2, z_neural_data_1, gamma)
    loglike22 = get_log_likelihood(z_neural_data2, z_neural_data2, gamma)
    loglike2_2 = get_log_likelihood(z_neural_data2, z_neural_data_2, gamma)

    loglike_21 = get_log_likelihood(z_neural_data_2, z_neural_data1, gamma)
    loglike_2_1 = get_log_likelihood(z_neural_data_2, z_neural_data_1, gamma)
    loglike_22 = get_log_likelihood(z_neural_data_2, z_neural_data2, gamma)
    loglike_2_2 = get_log_likelihood(z_neural_data_2, z_neural_data_2, gamma)

    #%% Maximum Likelihood Estimation (MLE) from log-likelihoods

    # Create vectors to compare pattern data across the same source data
    mt1 = np.array([loglike11, loglike1_1, loglike12, loglike1_2]) # maze-type R
    mt_1 = np.array([loglike_11, loglike_1_1, loglike_12, loglike_1_2]) # maze-type L
    mt2 = np.array([loglike21, loglike2_1, loglike22, loglike2_2]) # maze-type R*
    mt_2 = np.array([loglike_21, loglike_2_1, loglike_22, loglike_2_2]) # maze-type L*

    # The MLE is the highest log-likelihood at each timestep
    allmle1 = mt1.argmax(axis=0)
    counts_all1, _ = np.histogram(allmle1, bins) # How often was each maze-type the most likely?

    allmle_1 = mt_1.argmax(axis=0)
    counts_all_1, _ = np.histogram(allmle_1, bins)

    allmle2 = mt2.argmax(axis=0)
    counts_all2, _ = np.histogram(allmle2, bins)

    allmle_2 = mt_2.argmax(axis=0)
    counts_all_2, _ = np.histogram(allmle_2, bins)

    cnt_arr = np.array([counts_all1, counts_all_1, counts_all2, counts_all_2])

    # Turn the array of counts into average percents for each session
    mle_sess[il, :, :] = cnt_arr.astype(float) / cnt_arr.sum(axis=1)[:, None]

    for ii in np.arange(1, 6): # loop through all maze segments

        mle1 = mt1[:, mazeSeg1 == ii].argmax(axis=0)
        counts_all1, _ = np.histogram(mle1, bins)

        mle_1 = mt_1[:, mazeSeg_1 == ii].argmax(axis=0)
        counts_all_1, _ = np.histogram(mle_1, bins)

        mle2 = mt2[:, mazeSeg2 == ii].argmax(axis=0)
        counts_all2, _ = np.histogram(mle2, bins)

        mle_2 = mt_2[:, mazeSeg_2 == ii].argmax(axis=0)
        counts_all_2, _ = np.histogram(mle_2, bins)

        cnt_arr_seg = np.array([counts_all1, counts_all_1, counts_all2, counts_all_2])

        # Save mle for each maze segment into respective matrix for each session
        if ii == 1:
            mle_sess_fh[il, :, :] = cnt_arr_seg / cnt_arr_seg.sum(axis=1)[:, None]
        elif ii == 2:
            mle_sess_fc[il, :, :] = cnt_arr_seg / cnt_arr_seg.sum(axis=1)[:, None]
        elif ii == 3:
            mle_sess_mh[il, :, :] = cnt_arr_seg / cnt_arr_seg.sum(axis=1)[:, None]
        elif ii == 4:
            mle_sess_lc[il, :, :] = cnt_arr_seg / cnt_arr_seg.sum(axis=1)[:, None]
        elif ii == 5:
            mle_sess_lh[il, :, :] = cnt_arr_seg / cnt_arr_seg.sum(axis=1)[:, None]

    #%% Fig 3A: Plot the first two trials as examples

    # Get indicies of the last time steps for 1st and 2nd trials
    l1 = np.where(np.diff(mazeSeg1) < -1)[0][:2] +1
    l_1 = np.where(np.diff(mazeSeg_1) < -1)[0][:2] +1
    l2 = np.where(np.diff(mazeSeg2) < -1)[0][:2] +1
    l_2 = np.where(np.diff(mazeSeg_2) < -1)[0][:2] +1

    # get the maximum on time axis as x limit for all subplots
    xlim0 = np.round(np.array([l1, l_1, l2, l_2]).max(), -1)
    if np.array([l1, l_1, l2, l_2]).max() > xlim0:
        xlim0 += 10

    mle_perc = mle_sess[il, :, :]
    fig, ax = pl.subplots(4, 1, sharey=True, sharex=True, figsize=(4.42, 2.8)) #, figsize=(1.3, 2.0)
    fig.tight_layout()

    # Create one larger axis for x & y axis labels
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    pl.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    pl.xlabel("Time (seconds)", labelpad=1)
    pl.ylabel('MLE using Maze-Type Pattern Data')

    # Plot MLE for each timestep
    for o, m in enumerate(np.array([allmle1[:l1[-1]], allmle_1[:l_1[-1]], allmle2[:l2[-1]], allmle_2[:l_2[-1]]], dtype='object')):
        for i, c in enumerate(colors):
            ax[o].plot((np.where(m == i)[0]+1)/10, i*np.ones((m == i).sum()), '|', alpha=0.7, color=c, markersize=4)
            tx = 'R={0}%, L={1}%, R*={2}%, L*={3}%'.format(*(mle_perc[o, :]*100).astype(int))
            ax[o].set_title('Source Data from {}: {}'.format(mazeTypeList[o], tx), pad=1)

    # Plot maze segment data at the bottom
    for o, m in enumerate(np.array([mazeSeg1[:l1[-1]], mazeSeg_1[:l_1[-1]], mazeSeg2[:l2[-1]], mazeSeg_2[:l_2[-1]]], dtype='object')):
        for i, c in enumerate([' ', 'k', ' ', 'k', ' ']):
            if i % 2:
                ax[o].plot((np.where(m == i+1)[0]+1)/10, -1*np.ones((m == i+1).sum()), '|', alpha=0.4, color=c, markersize=4)
                ax[o].set_ylim([-1.7, 3.7])
                ax[o].set_xlim([-1, xlim0/10])
        ax[o].plot([((np.where(np.diff(m[:l1[-1]]) < -1)[0])/10)[0], len(m)], [-1, -1], '|', color='k', markersize=4)

    ax[0].set_yticks([0, 1, 2, 3])
    ax[0].set_yticklabels(mazeTypeList)
    ax[0].set_xticks([0, xlim0/20, xlim0/10])

    fig.savefig(combinedResultDir+'Fig3A_mle_timesteps_{}.pdf'.format(session), format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.05)
    pl.close(fig)

    #%% MLE while excuding the same maze as pattern (ex = excluding same maze-type)

    mex1 = mt1[1:, :]
    exmle1 = mex1.argmax(axis=0) # MLE excluding same maze
    counts_ex1, _ = np.histogram(exmle1, bins_ex)
    counts_ex1 = np.insert(counts_ex1, 0, 0) # Insert 0 for matrix view

    mex_1 = mt_1[[0, 2, 3], :]
    exmle_1 = mex_1.argmax(axis=0)
    counts_ex_1, _ = np.histogram(exmle_1, bins_ex)
    counts_ex_1 = np.insert(counts_ex_1, 1, 0)

    mex2 = mt2[[0, 1, 3], :]
    exmle2 = mex2.argmax(axis=0)
    counts_ex2, _ = np.histogram(exmle2, bins_ex)
    counts_ex2 = np.insert(counts_ex2, 2, 0)

    mex_2 = mt_2[[0, 1, 2], :]
    exmle_2 = mex_2.argmax(axis=0)
    counts_ex_2, _ = np.histogram(exmle_2, bins_ex)
    counts_ex_2 = np.insert(counts_ex_2, 3, 0)

    cnt_ex = np.array([counts_ex1, counts_ex_1, counts_ex2, counts_ex_2])

    mle_ex_sess[il, :, :] = cnt_ex / cnt_ex.sum(axis=1)[:, None]

    for ii in np.arange(1, 6): # go through all maze segments

        # Segment MLE, 1=R
        smle1 = mex1[:, mazeSeg1 == ii].argmax(axis=0)
        counts_ex1, _ = np.histogram(smle1, bins_ex)
        counts_ex1 = np.insert(counts_ex1, 0, 0)

        smle_1 = mex_1[:, mazeSeg_1 == ii].argmax(axis=0) #_1 = L
        counts_ex_1, _ = np.histogram(smle_1, bins_ex)
        counts_ex_1 = np.insert(counts_ex_1, 1, 0)

        smle2 = mex2[:, mazeSeg2 == ii].argmax(axis=0) #2 = R*
        counts_ex2, _ = np.histogram(smle2, bins_ex)
        counts_ex2 = np.insert(counts_ex2, 2, 0)

        smle_2 = mex_2[:, mazeSeg_2 == ii].argmax(axis=0) #_2 = L*
        counts_ex_2, _ = np.histogram(smle_2, bins_ex)
        counts_ex_2 = np.insert(counts_ex_2, 3, 0)

        cnt_ex_seg = np.array([counts_ex1, counts_ex_1, counts_ex2, counts_ex_2])

        if ii == 1:
            mle_ex_sess_fh[il, :, :] = cnt_ex_seg / cnt_ex_seg.sum(axis=1)[:, None]
        elif ii == 2:
            mle_ex_sess_fc[il, :, :] = cnt_ex_seg / cnt_ex_seg.sum(axis=1)[:, None]
        elif ii == 3:
            mle_ex_sess_mh[il, :, :] = cnt_ex_seg / cnt_ex_seg.sum(axis=1)[:, None]
        elif ii == 4:
            mle_ex_sess_lc[il, :, :] = cnt_ex_seg / cnt_ex_seg.sum(axis=1)[:, None]
        elif ii == 5:
            mle_ex_sess_lh[il, :, :] = cnt_ex_seg / cnt_ex_seg.sum(axis=1)[:, None]

#%% Plot MLE over all sessions
if toPlotAllSessions:

    df_count.to_csv(combinedResultDir+'table_S1_place_cell_field_counts.csv')

    # plot as percentages
    pctAll = [mle_sess.mean(axis=0), mle_sess_fh.mean(axis=0), mle_sess_fc.mean(axis=0), mle_sess_mh.mean(axis=0), mle_sess_lc.mean(axis=0), mle_sess_lh.mean(axis=0)]
    pctEx = [mle_ex_sess.mean(axis=0), mle_ex_sess_fh.mean(axis=0), mle_ex_sess_fc.mean(axis=0), mle_ex_sess_mh.mean(axis=0), mle_ex_sess_lc.mean(axis=0), mle_ex_sess_lh.mean(axis=0)]

    fig, axw = pl.subplots(3, 2, figsize=(2.05, 3.6), sharex=True, sharey=True)
    fig.tight_layout()
    axw = axw.ravel()

    for i, pa in enumerate(pctAll):
        im = axw[i].imshow((pa*100), 'Blues', vmin=0, vmax=50)
        axw[i].set(
            xticks=[0, 1, 2, 3],
            xticklabels=(mazeTypeList),
            yticks=[0, 1, 2, 3],
            yticklabels=(mazeTypeList),
            title=mazeSegList[i])

    cbar_x = fig.add_axes([0.99, 0.235, 0.015, 0.53]) # colorbar
    fig.colorbar(im, cax=cbar_x, ticks=[0, 50]) # must match the vmin and vmax above
    cbar_x.set_ylabel("%", rotation=0, labelpad=-5)

    # Create one larger axis for x & y axis labels
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    pl.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    pl.ylabel("Maximum Likelihood Estimate from Pattern Data", labelpad=1)
    pl.xlabel("Maximum Likelihood Estimate from Source Data", labelpad=0)

    fig.savefig(combinedResultDir+'Fig3B_allSessions_withReal_mle_gamma{}.pdf'.format(gamma), format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.05)
    pl.close(fig)

    #%% Plot MLE excluding same maze-type
    fig, axw = pl.subplots(3, 2, figsize=(2.05, 3.6), sharex=True, sharey=True)
    fig.tight_layout()
    axw = axw.ravel()

    # plot as percentages
    for i, p0 in enumerate(pctEx):
        im = axw[i].imshow(p0*100, 'Oranges', vmin=0, vmax=50)
        axw[i].set(
            xticks=[0, 1, 2, 3],
            xticklabels=(mazeTypeList),
            yticks=[0, 1, 2, 3],
            yticklabels=(mazeTypeList),
            title=mazeSegList[i])

    cbar_x = fig.add_axes([0.99, 0.235, 0.015, 0.53])
    fig.colorbar(im, cax=cbar_x, ticks=[0, 50]) # must match the vmin and vmax above
    cbar_x.set_ylabel("%", rotation=0, labelpad=-5)

    # Create one larger axis for x & y axis labels
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    pl.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    pl.ylabel("Maximum Likelihood Estimate from Pattern Data", labelpad=1)
    pl.xlabel("Maximum Likelihood Estimate from Source Data", labelpad=0)

    fig.savefig(combinedResultDir+'Fig3C_allSessions_noReal_mle_gamma{}.pdf'.format(gamma), format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.05)
    pl.close(fig)
