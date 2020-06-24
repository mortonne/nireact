"""Information about the associative inference task."""

import os
import numpy as np
import scipy.stats as st
import pandas as pd


def sem(x):
    """Standard error of the mean."""
    return np.mean(x) / np.sqrt(len(x) - 1)


def get_subjects():
    """Get a list of included subject IDs."""

    # same as in mistr_profile
    subj_numbers = [2, 4, 5, 6, 7, 8, 9, 10,
                    12, 13, 14, 15, 16, 17, 18, 19, 20,
                    22, 23, 24, 25, 26, 27, 28, 29, 30]
    subj_ids = ['mistr_{:02d}'.format(n) for n in subj_numbers]
    return subj_numbers, subj_ids


def _read_phase_all_subj(data_dir, phase):
    """Read data for a given phase for all subjects."""

    # get a list of included subjects
    subj_numbers, subj_ids = get_subjects()

    # get a list of all subject dataframes
    dfs = []
    n = 0
    for subj_no, subj_id in zip(subj_numbers, subj_ids):
        subj_dir = os.path.join(data_dir, subj_id, 'behav', 'log')
        if phase == 'study':
            subj_df = read_study(subj_dir)
        elif phase == 'test':
            subj_df = read_test(subj_dir)
        else:
            raise ValueError(f'Unsupported phase type: {phase}')
        subj_df['subject'] = subj_no
        subj_df['subj_id'] = f'mistr_{subj_no:02d}'
        subj_df['subj_idx'] = n
        n += 1
        dfs.append(subj_df)

    # concatenate all subjects
    df = pd.concat(dfs, axis=0)
    return df


def read_disp_run(subj_dir, run):
    """Read task data for one display run."""

    # read the log file
    run_file = os.path.join(subj_dir, 'log_01-{:02d}-01_disp.txt'.format(run))
    if not os.path.exists(run_file):
        raise IOError('Log file does not exist: {}'.format(run_file))
    df = pd.read_csv(run_file, sep='\t')

    # determine item number
    groups = np.vstack((df.A, df.B, df.C)).T
    itemno = np.array([groups[i, df.stimType[i] - 1]
                       for i in range(groups.shape[0])])

    data = pd.DataFrame({'onset': df['onset (s)'],
                         'triad': df['triadNr'],
                         'train_type': df['trainType'],
                         'item_type': df['stimType'],
                         'itemno': itemno,
                         'run': np.tile(run, itemno.shape)})
    return data


def read_disp(subj_dir):
    """Read all display task data."""

    df_run = []
    for run in range(1, 9):
        df_run.append(read_disp_run(subj_dir, run))
    df = pd.concat(df_run, ignore_index=True)
    return df


def read_study(subj_dir):
    """Read study phase data."""

    log_file = os.path.join(subj_dir, 'log_02-01-01_study.txt')
    df = pd.read_csv(log_file, sep='\t')
    df['itemno1'] = df.item1
    df['itemno2'] = df.item2
    triads = np.unique(df.triad)
    n = 1
    for triad in triads:
        df.item1[(df.triad == triad) & (df.pair_type == 1)] = n
        df.item2[(df.triad == triad) & (df.pair_type == 1)] = n + 1
        df.item1[(df.triad == triad) & (df.pair_type == 2)] = n + 1
        df.item2[(df.triad == triad) & (df.pair_type == 2)] = n + 2
        n += 3
    return df


def read_study_all_subj(data_dir):
    """Read study phase data for all subjects."""

    df = _read_phase_all_subj(data_dir, 'study')
    return df


def study_lag_block(df):
    """Calculate lag between each trial and the last overlapping trial."""

    items = df.loc[:, ['item1', 'item2']].to_numpy()
    triad = df.triad.to_numpy()
    pair_type = df.pair_type.to_numpy()
    lag = np.zeros(df.shape[0])
    for i in range(items.shape[0]):
        match = np.nonzero((triad[:i] == triad[i]) &
                           (pair_type[:i] != pair_type[i]))[0]
        if match.size == 0:
            lag[i] = np.nan
        else:
            lag[i] = i - match[-1]
    return lag


def study_lag(group_df):
    """Calculate lag for all trials."""

    group_df.loc[:, 'lag'] = np.zeros(group_df.shape[0])
    for subj in np.unique(group_df.subj_idx):
        subj_df = group_df.loc[group_df.subj_idx == subj]
        for run in np.unique(subj_df.run.values):
            df = subj_df.loc[subj_df.run == run]
            lag = study_lag_block(df)
            match = ((group_df.subj_idx == subj) &
                     (group_df.run == run))
            group_df.loc[match, 'lag'] = lag
    return group_df


def read_test(subj_dir):
    """Read all test phase data."""

    # read the log file
    log_file = os.path.join(subj_dir, 'log_03-01-01_test.txt')
    if not os.path.exists(log_file):
        raise IOError('Log file does not exist: {}'.format(log_file))
    df = pd.read_csv(log_file, sep='\t')

    # test type is 1 for direct (AB, BC), 2 for indirect (AC)
    test_type = np.ones(df.trialType.shape, dtype=int)
    test_type[df.trialType == 3] = 2

    # 1 if test correct, 0 if incorrect
    correct = (df.CorrectResp == df.Resp).astype('int')

    # dataframe with standard fields
    data = pd.DataFrame({'trial': df.trialNr.astype(int),
                         'triad': df.triadNr.astype(int),
                         'train_type': df.trainType.astype(int),
                         'trial_type': df.trialType.astype(int),
                         'test_type': test_type,
                         'correct': correct,
                         'rt': df['RT'],
                         'response': correct})
    test_names = {1: 'AB', 2: 'BC', 3: 'AC'}
    data['test'] = [test_names[i] for i in data.trial_type]
    return data


def read_test_all_subj(data_dir):
    """Read test phase data for all subjects."""

    df = _read_phase_all_subj(data_dir, 'test')
    return df


def scrub_rt(data, thresh_iqr=5, thresh=None):
    """Remove outliers from data."""

    raw = data.dropna()
    if thresh is None:
        thresh = raw.groupby('test_type').agg(
            lambda x: np.median(x) + thresh_iqr * st.iqr(x))
    include = ((raw.test_type == 1) & (raw.rt <= thresh.loc[1, 'rt']) |
               (raw.test_type == 2) & (raw.rt <= thresh.loc[2, 'rt']))
    scrub = raw.loc[include].copy()
    return scrub


def test_rt(data_dir, scrub=False, thresh_iqr=5):
    """Calculate average test RT for all subjects."""

    # read all test data from log files
    raw = read_test_all_subj(data_dir)

    if scrub:
        test = scrub_rt(raw, thresh_iqr)
    else:
        test = raw

    # mean RT for correct trials, by trial type and subject
    correct = test.loc[test.correct == 1]
    rt_corr = correct.groupby(['test', 'subj_id'], as_index=False).mean()
    rt_mean = rt_corr.pivot(index='subj_id', columns='test', values='rt')
    rt_mean = rt_mean.reindex(['AB', 'BC', 'AC'], axis=1)
    return rt_mean


def disp_vols(subj_dir):
    """Task information for each display task beta series volume."""

    # load display task data
    disp = read_disp(subj_dir)

    # load test data
    test = read_test(subj_dir)
    ac_test = test.loc[test.trial_type == 3, :]
    disp['correct'] = 0
    for _, trial in ac_test.iterrows():
        disp.loc[trial.triad == disp.triad, 'correct'] = int(trial.correct)

    # get post runs, A and C items only
    vols = disp.groupby(['run', 'itemno'], as_index=False).first()
    return vols


def task_triads(n_triad=6, n_traintype=2):
    """Task information by triad."""

    n_triad_tot = n_triad * n_traintype
    a = np.arange(1, (n_triad_tot * 3) + 1, 3)
    b = np.arange(2, (n_triad_tot * 3) + 1, 3)
    c = np.arange(3, (n_triad_tot * 3) + 1, 3)
    triad = np.arange(1, n_triad_tot + 1)
    traintype = np.repeat(np.arange(1, n_traintype + 1), n_triad)

    df = pd.DataFrame({'triad': triad, 'traintype': traintype,
                       'a': a, 'b': b, 'c': c})
    return df


def task_pairs(n_triad=6, n_traintype=2):
    """Task information by pair."""

    n_triad_tot = n_triad * n_traintype
    pair = np.arange(1, (n_triad_tot * 2) + 1)
    triad = np.repeat(np.arange(1, n_triad_tot + 1), 2)
    pairtype = np.tile(np.arange(1, 3), n_triad_tot)
    traintype = np.repeat(np.arange(1, n_traintype + 1), n_triad * 2)

    triads = task_triads(n_triad, n_traintype)
    item1 = np.zeros(n_triad_tot * 2, dtype=int)
    item2 = np.zeros(n_triad_tot * 2, dtype=int)
    item1[pairtype == 1] = triads.a
    item2[pairtype == 1] = triads.b
    item1[pairtype == 2] = triads.b
    item2[pairtype == 2] = triads.c

    type1 = np.zeros(n_triad_tot * 2, dtype=int)
    type2 = np.zeros(n_triad_tot * 2, dtype=int)
    type1[pairtype == 1] = 1
    type2[pairtype == 1] = 2
    type1[pairtype == 2] = 2
    type2[pairtype == 2] = 3

    df = pd.DataFrame({'pair': pair, 'triad': triad,
                       'pairtype': pairtype,
                       'traintype': traintype,
                       'item1': item1, 'item2': item2,
                       'type1': type1, 'type2': type2})
    return df


def task_items(n_triad=6, n_traintype=2):
    """Task information by item."""

    n_triad_tot = n_triad * n_traintype
    itemno = np.arange(1, (n_triad_tot * 3) + 1)
    triad = np.repeat(np.arange(1, n_triad_tot + 1), 3)
    itemtype = np.tile(np.arange(1, 4), n_triad_tot)
    traintype = np.repeat(np.arange(1, n_traintype + 1), n_triad * 3)

    df = pd.DataFrame({'itemno': itemno, 'triad': triad,
                       'item_type': itemtype, 'train_type': traintype})
    return df


def pair_item_patterns(pairs):
    """Generate patterns to represent items for each pair."""

    pairs = pairs.reset_index()
    n_pair = pairs.shape[0]
    items = list(np.unique(np.hstack((pairs.item1, pairs.item2))))
    n_unit = len(items)
    patterns = np.zeros((n_pair, n_unit))
    for i, pair in pairs.iterrows():
        patterns[i, items.index(pair.item1)] = 1
        patterns[i, items.index(pair.item2)] = 1
    return patterns


def pair_episode_patterns(pairs):
    """Generate patterns to represent pair episodes."""

    n_pair = pairs.shape[0]
    patterns = np.zeros((n_pair, n_pair))
    for i in range(n_pair):
        patterns[i, pairs.pair[i] - 1] = 1
    return patterns


def pair_itemtype_patterns(pairs):
    """Generate patterns to represent pair item types."""

    n_pair = pairs.shape[0]
    n_unit = len(np.unique(np.hstack((pairs.type1, pairs.type2))))
    patterns = np.zeros((n_pair, n_unit))
    for i in range(n_pair):
        patterns[i, pairs.type1[i] - 1] = 1
        patterns[i, pairs.type2[i] - 1] = 1
    return patterns
