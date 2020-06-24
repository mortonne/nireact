"""Representational similarity analysis."""

import os
import random
import itertools
import math
import numpy as np
import pandas as pd
import scipy.spatial.distance as sd
import sklearn.manifold as mn
from mindstorm import prsa
from nireact import task


def load_roi_pattern(roi_dir, subject):
    """Load ROI patterns from exported data set."""

    mat_file = os.path.join(roi_dir, f'pattern_{subject}.txt')
    tab_file = os.path.join(roi_dir, f'pattern_{subject}.csv')
    vols = pd.read_csv(tab_file, index_col=0)
    patterns = np.loadtxt(mat_file)
    return patterns, vols


def item_mean_pattern(patterns, vols):
    """Average across presentations and sort by task structure."""

    # average of each item across runs
    items = np.unique(vols.itemno.values)
    m_pat = np.zeros((len(items), patterns.shape[1]))
    for i, number in enumerate(items):
        m_pat[i, :] = np.mean(patterns[vols.itemno.values == number, :], 0)

    # corresponding item information
    item_vols = vols.groupby(['itemno'], as_index=False).first()

    # re-sort to reflect task features
    sort_vols = item_vols.sort_values(by=['train_type', 'triad', 'item_type'])
    ind = np.array(sort_vols.index)
    m_pat_sort = m_pat[ind, :]

    # remove fields that vary over presentations and reset index
    sort_vols = sort_vols.drop(['run', 'onset'], axis=1).reset_index(drop=True)
    return m_pat_sort, sort_vols


def item_mds(patterns, vols, n_dim=2, distance='correlation',
             runs=None, item_types=None, train_type=None, manifold='MDS'):
    """Calculate multi-dimensional scaling of mean item patterns."""

    # filter to get trials of interest
    include = np.ones(patterns.shape[0], dtype=bool)
    if runs is not None:
        if runs == 'post':
            include = include & (vols.run.values >= 5)
        elif runs == 'pre':
            include = include & (vols.run.values <= 4)

    if item_types is not None:
        include = include & np.isin(vols.item_type, item_types)

    if train_type is not None:
        include = include & (vols.train_type.values == train_type)

    # get mean patterns for included items, sorted by task structure
    item_patterns, item_vols = item_mean_pattern(patterns[include],
                                                 vols.loc[include])

    # calculate MDS projection
    if manifold == 'MDS':
        item_rdm = sd.squareform(sd.pdist(item_patterns, distance))
        embedding = mn.MDS(n_components=n_dim, dissimilarity='precomputed')
        mds_fit = embedding.fit_transform(item_rdm)
    elif manifold == 'TSNE':
        mds_fit = mn.TSNE(n_components=2).fit_transform(item_patterns)
    elif manifold == 'LTSA':
        embedding = mn.LocallyLinearEmbedding(n_components=2, method='modified',
                                              n_neighbors=17)
        mds_fit = embedding.fit_transform(item_patterns)
    else:
        raise ValueError(f'Unknown embedding: {manifold}')

    # combine in a data frame
    mds_df = pd.DataFrame({'Dimension 1': mds_fit[:, 0],
                          'Dimension 2': mds_fit[:, 1]})
    df = pd.concat((item_vols, mds_df), axis=1)
    return df


def item_mds_all_subj(patterns_list, vols_list, **mds_args):
    """Run item MDS for a set of patterns."""

    df_list = []
    for i in range(len(patterns_list)):
        df = item_mds(patterns_list[i], vols_list[i], **mds_args)
        df['subj_idx'] = i
        df_list.append(df)

    subj_df = pd.concat(df_list, axis=0, ignore_index=True)
    return subj_df


def item_mds_roi(roi_dir, runs='post', item_types=(1, 3), train_type=None):
    """Run item MDS on an ROI for all subjects."""

    _, subj_ids = task.get_subjects()
    patterns_list = []
    vols_list = []
    for subj_id in subj_ids:
        patterns, vols = load_roi_pattern(roi_dir, subj_id)
        patterns_list.append(patterns)
        vols_list.append(vols)

    df = item_mds_all_subj(patterns_list, vols_list, runs=runs,
                           item_types=item_types, train_type=train_type)
    return df


def pair_and(x):
    """Pairs where conditions are both true."""
    return x[:, None] & x[:, None].T


def pair_eq(x):
    """Pairs where conditions are equal."""
    return x[:, None] == x[:, None].T


def pair_neq(x):
    """Pairs where conditions are not equal."""
    return x[:, None] != x[:, None].T


def perm_within(group, n_perm):
    """Create permutation indices to scramble within trial group."""

    ugroup = np.unique(group)
    rand_ind = []
    for i in range(n_perm):
        perm_ind = np.zeros((len(group),), dtype=int)
        for j in ugroup:
            group_ind = list(np.nonzero(group == j)[0])
            perm_ind[group_ind] = random.sample(group_ind, len(group_ind))
        rand_ind.append(perm_ind)
    return rand_ind


def perm_triad_type(perm_item_type, triad, item_type, train_type, n_perm):
    """Permute items across triads within type."""

    # scramble items across triads. For example, C1 and C2
    # are exchangeable
    u_train_type = np.unique(train_type)
    train_triads = [np.unique(triad[train_type == tt]) for tt in u_train_type]

    # start with ordered indices
    rand_ind = np.tile(np.arange(len(train_type), dtype=int), (n_perm + 1, 1))
    for i in range(n_perm):
        for triads in train_triads:
            # randomly order triads for one item type
            perm_triads = np.random.permutation(triads)
            for old_triad, new_triad in zip(triads, perm_triads):
                # replace indices of old triad item with new one
                old_ind = (triad == old_triad) & (item_type == perm_item_type)
                new_ind = (triad == new_triad) & (item_type == perm_item_type)
                rand_ind[i + 1, old_ind] = np.nonzero(new_ind)[0]
    return rand_ind


def perm_triad_type_exhaust(perm_item_type, triad, item_type, train_type):
    """Exhaustively permute items across triads within type."""

    # scramble items across triads. For example, C1 and C2
    # are exchangeable
    u_train_type = np.unique(train_type)
    train_triads = [np.unique(triad[train_type == tt]) for tt in u_train_type]
    n_train_triad = [len(t) for t in train_triads]

    # generate all possible orderings of triad within training type
    comb_train = [itertools.permutations(triads) for triads in train_triads]

    # generate all combinations across training type
    n_perm = np.prod([math.factorial(n) for n in n_train_triad])
    rand_ind = np.zeros((n_perm, len(train_type)), dtype=int)
    for i, comb in enumerate(itertools.product(*comb_train)):
        # write indices to get the new order
        rand_ind[i, :] = np.arange(len(train_type))
        for old_train, new_train in zip(train_triads, comb):
            for old_triad, new_triad in zip(old_train, new_train):
                # replace indices of old triad item with new one
                old_ind = (triad == old_triad) & (item_type == perm_item_type)
                new_ind = (triad == new_triad) & (item_type == perm_item_type)
                rand_ind[i, old_ind] = np.nonzero(new_ind)[0]
    return rand_ind


def perm_item_triad(run, triad, item, n_perm):
    """Permute items within triad."""

    # scramble items within triad. For example, A1 and C1 are
    # exchangeable
    u_run = np.unique(run)
    item_run = item[run == u_run[0]]
    triad_run = triad[run == u_run[0]]
    rand_item_ind = perm_within(triad_run, n_perm)

    # permute the items the same way for every run
    rand_ind = np.zeros((n_perm + 1, len(run)), dtype=int)
    rand_ind[0] = np.arange(len(run))
    for i in range(n_perm):
        rand_item_run = item_run[rand_item_ind[i]]
        for r in u_run:
            for old_item, new_item in zip(item_run, rand_item_run):
                old_ind = (run == r) & (item == old_item)
                new_ind = (run == r) & (item == new_item)
                rand_ind[i + 1, old_ind] = np.nonzero(new_ind)[0]
    return rand_ind


def perm_item_triad_exhaust(train_type, triad, item_type, shuffle_types):
    """Permute items within triad exhaustively."""

    # all within-triad flip combinations within training condition
    train_triad = [np.unique(triad[train_type == tt])
                   for tt in np.unique(train_type)]
    n_train = [len(t) for t in train_triad]
    comb_train = [itertools.product([0, 1], repeat=n) for n in n_train]

    # all combinations of the within-training condition flips
    n_perm = np.prod(np.power(2, n_train))
    rand_ind = np.zeros((n_perm, len(train_type)), dtype=int)
    for i, comb in enumerate(itertools.product(*comb_train)):
        rand_ind[i, :] = np.arange(len(train_type))
        for triads, train_switch in zip(train_triad, comb):
            for this_triad, switch in zip(triads, train_switch):
                items1 = np.nonzero((item_type == shuffle_types[0]) &
                                    (triad == this_triad))[0]
                items2 = np.nonzero((item_type == shuffle_types[1]) &
                                    (triad == this_triad))[0]
                if switch == 1:
                    rand_ind[i, items1] = items2
                    rand_ind[i, items2] = items1
    return rand_ind


def prep_triad_sim(vols, item_comp, stat_type, n_perm=None):
    """Prepare triad similarity change analysis."""

    post = vols[vols.run >= 5]
    n_trial = post.shape[0]
    upper_tri = np.triu(np.ones((n_trial, n_trial)), 1)
    include = (pair_neq(post.run.values) &
               pair_eq(post.train_type.values) &
               pair_and(np.isin(post.item_type.values, item_comp)) &
               pair_and(post.correct.values == 1) &
               (upper_tri == 1))

    if stat_type == 'item_type':
        if n_perm is not None:
            raise ValueError('n_perm should not be set for item_type analysis.')

        # permute items within triad, exhaustively
        rand_ind = perm_item_triad_exhaust(post.train_type.values,
                                           post.triad.values,
                                           post.item_type.values, item_comp)
        signal = post.item_type.values
    elif stat_type == 'triad':
        # permute items across triad, within item type
        rand_ind = perm_triad_type(item_comp[1], post.triad.values,
                                   post.item_type.values,
                                   post.train_type.values, n_perm)
        signal = post.triad.values
    else:
        raise ValueError(f'Invalid stat_type: {stat_type}')

    # define trial groups of interest
    trial_ind = {'within': include & pair_eq(signal),
                 'across': include & pair_neq(signal),
                 'block': include & pair_and(post.train_type.values == 1),
                 'inter': include & pair_and(post.train_type.values == 2)}

    # get indices for each condition
    bin_trial = {'wb': ('within', 'block'), 'wi': ('within', 'inter'),
                 'ab': ('across', 'block'), 'ai': ('across', 'inter')}
    cond_ind = {}
    for bin_name, trial_names in bin_trial.items():
        # get trial conjunction
        match = trial_ind[trial_names[0]] & trial_ind[trial_names[1]]

        # scrambled indices for this comparison
        bin_ind = []
        for ind in rand_ind:
            match_perm = match[np.ix_(ind, ind)]
            bin_ind.append(np.nonzero(match_perm))
        cond_ind[bin_name] = bin_ind
    return cond_ind


def zstat_triad_sim(patterns, cond_ind, distance='correlation'):
    """Calculate z-statistic for similarity difference."""

    rho = 1 - sd.squareform(sd.pdist(patterns), distance)

    # mean statistic for each condition and permutation
    cond_stat = {}
    n_perm = len(cond_ind[list(cond_ind.keys())[0]])
    for cond, ind in cond_ind.items():
        m = np.zeros(n_perm)
        for i in range(n_perm):
            m[i] = np.mean(rho[ind[i]])
        cond_stat[cond] = m

    # calculate difference statistic for each permutation
    stat = np.zeros((n_perm, 2))
    stat[:, 0] = cond_stat['wb'] - cond_stat['ab']
    stat[:, 1] = cond_stat['wi'] - cond_stat['ai']
    zstat = zstat_contrasts(stat)
    return zstat


def prep_xval_ind(perm_ind, xval_runs, run, triads, triad, item_type,
                  item_comp):
    """Prepare cross-validation of triads for a set of runs."""

    # array of [perm x triad x run pair x run order x item type] indices
    xval_ind = np.zeros((len(perm_ind), len(triads), len(xval_runs), 2, 2),
                        dtype=int)
    for i, ind in enumerate(perm_ind):
        for j, this_triad in enumerate(triads):
            include = triad[ind] == this_triad
            for k, run_list in enumerate(xval_runs):
                for n, run1 in enumerate(run_list):
                    run2 = [r for r in run_list if r != run1][0]
                    # find items in this pair
                    item1 = np.nonzero(include & (run[ind] == run1) &
                                       (item_type[ind] == item_comp[0]))[0][0]
                    item2 = np.nonzero(include & (run[ind] == run2) &
                                       (item_type[ind] == item_comp[1]))[0][0]
                    xval_ind[i, j, k, n, :] = [item1, item2]
    return xval_ind


def prep_triad_xval(run, triad, train_type, item_type, item_comp, rand_ind,
                    sep_perm=False):
    """Prepare double cross-validation over runs and triads."""

    # split runs into train and test pairs
    u_run = np.unique(run)
    train_runs = list(itertools.combinations(u_run, 2))
    test_runs = [[r for r in u_run if r not in t] for t in train_runs]

    # trains in each training type
    train_triad = [np.unique(triad[train_type == t])
                   for t in np.unique(train_type)]

    # get item pair indices for each training condition
    # [train type] list of
    # [perm x triad x run pair x run order x item type] indices
    train_ind = []
    test_ind = []
    for i, triads in enumerate(train_triad):
        if sep_perm:
            ind = rand_ind[i]
        else:
            ind = rand_ind

        # training
        train_xval = prep_xval_ind(ind, train_runs, run, triads, triad,
                                   item_type, item_comp)
        train_ind.append(train_xval)

        # testing
        test_xval = prep_xval_ind(ind, test_runs, run, triads, triad,
                                  item_type, item_comp)
        test_ind.append(test_xval)
    return train_ind, test_ind


def prep_triad_vector(vols, item_comp, n_perm=1000, perm_type='item_type',
                      exhaust=True):
    """Prepare triad vector analysis."""

    # scramble items within triad
    if perm_type == 'item_type':
        if exhaust:
            rand_ind = perm_item_triad_exhaust(vols.train_type.values,
                                               vols.triad.values,
                                               vols.item_type.values, item_comp)
        else:
            rand_ind = perm_item_triad(vols.run.values, vols.triad.values,
                                       vols.itemno.values, n_perm)
    elif perm_type == 'triad':
        rand_ind = perm_triad_type(item_comp[1], vols.triad.values,
                                   vols.item_type.values,
                                   vols.train_type.values, n_perm)
    else:
        raise ValueError(f'Invalid permutation type: {perm_type}')

    # get indices for trials to use in cross-validation
    train_ind, test_ind = prep_triad_xval(vols.run.values, vols.triad.values,
                                          vols.train_type.values,
                                          vols.item_type.values,
                                          item_comp, rand_ind)
    return train_ind, test_ind


def vector_error(patterns, train_ind, test_ind, stat_type='err'):
    """Mean distance from predicted pattern to observed pattern."""

    n_triad = train_ind.shape[1]

    # training and testing data
    # [perm x triad x xval x run pair x item type x samples]
    train_xmat = patterns[train_ind]
    test_xmat = patterns[test_ind]

    # difference between item types, averaged over run pairs
    # perm x triad x xval x samples
    train_dmat = np.mean(train_xmat[..., 1, :] -
                         train_xmat[..., 0, :], 3)

    # for each triad, get the average over all the other triads
    # perm x triad x xval x samples
    ind = np.array([[i for i in range(n_triad) if i != j]
                    for j in range(n_triad)])
    train_dmat_avg = np.mean(train_dmat[:, ind], 2)

    # itemtype1 + average difference vector (same used for
    # both run pairs)
    pred = test_xmat[..., 0, :] + train_dmat_avg[:, :, :, np.newaxis, :]
    obs = test_xmat[..., 1, :]

    # Euclidean distance between predicted and observed item type2
    err = np.sqrt(np.sum((pred - obs) ** 2, -1))

    if stat_type == 'err':
        # distance between predicted and observed for the correct triad
        stat = err
    elif stat_type == 'diff':
        # for each triad, get the observed patterns for other triads
        obs_other = obs[:, ind]

        # mean distance to other observed patterns
        diff_other = pred[:, :, np.newaxis, ...] - obs_other
        err_other = np.sqrt(np.sum(diff_other ** 2, -1))
        mean_err_other = np.mean(err_other, 2)

        # difference between error for correct triad and other triads
        stat = err - mean_err_other
    else:
        raise ValueError(f'Unknown stat type: {stat_type}')

    # average over triad, xval, and run pair
    stat_vec = stat.reshape((stat.shape[0], np.prod(stat.shape[1:])))
    return np.mean(stat_vec, 1)


def zstat_contrasts(stat):
    """Calculate z-statistics for a standard set of contrasts."""

    zstat = {'pos': prsa.perm_z(np.nansum(stat, 1)),
             'neg': prsa.perm_z(-np.nansum(stat, 1)),
             'block': prsa.perm_z(stat[:, 0]),
             'inter': prsa.perm_z(stat[:, 1]),
             'block_neg': prsa.perm_z(-stat[:, 0]),
             'inter_neg': prsa.perm_z(-stat[:, 1]),
             'block_inter': prsa.perm_z(stat[:, 0] - stat[:, 1]),
             'inter_block': prsa.perm_z(stat[:, 1] - stat[:, 0])}
    return zstat


def zstat_triad_vector(patterns, train_ind, test_ind, stat_type):
    """Calculate z-statistic for triad vector analysis."""

    # calculate negative error for all permutations and training types
    n_perm = len(train_ind[0])
    n_train = len(train_ind)
    stat = np.zeros((n_perm, n_train))
    for i in range(n_train):
        err = vector_error(patterns, train_ind[i], test_ind[i], stat_type)
        stat[:, i] = -err

    zstat = zstat_contrasts(stat)
    return zstat


def load_zstat(res_dir, roi, subj_ids=None):
    """Load z-statistic from permutation test results."""

    if subj_ids is None:
        _, subj_ids = task.get_subjects()

    z = []
    for subj_id in subj_ids:
        subj_file = os.path.join(res_dir, 'roi', roi, f'zstat_{subj_id}.txt')
        zstat = np.loadtxt(subj_file)
        z.append(zstat)
    return np.asarray(z)


def load_zstat_df(res_dir, rois, roi_names=None, var_names=None, subj_ids=None):
    """Load z-statistics into a data frame."""

    if subj_ids is None:
        _, subj_ids = task.get_subjects()

    if roi_names is None:
        roi_names = rois

    if var_names is None:
        var_names = ['pos', 'neg', 'block', 'inter',
                     'block_neg', 'inter_neg', 'block_inter', 'inter_block']

    df_list = []
    for roi, roi_name in zip(rois, roi_names):
        z = load_zstat(res_dir, roi, subj_ids)
        zdf = pd.DataFrame(z, columns=var_names)
        zdf['subj_id'] = subj_ids
        zdf['subj_idx'] = np.arange(len(subj_ids))
        zdfm = zdf.melt(id_vars=['subj_id'], value_name='z',
                        var_name='condition')
        zdfm['roi'] = roi_name
        df_list.append(zdfm)
    df = pd.concat(df_list, axis=0, ignore_index=True)
    return df


def load_zstat_full(res_dir):
    """Load all results for an ROI analysis."""

    _, subj_ids = task.get_subjects()
    roi_names = ['mmpfc', 'lhpc', 'lphc', 'llpc', 'rdlpfc', 'lfpo', 'rfpo']
    rois = [f'vec_triad_{roi}_dil1' for roi in roi_names]
    contrasts = ['pos', 'neg', 'block', 'inter',
                 'block_neg', 'inter_neg', 'block_inter', 'inter_block']
    df = load_zstat_df(res_dir, rois, roi_names, contrasts, subj_ids)
    return df


def zstat_pivot(zdf, condition):
    """Z-statistic pivot table with subjects x rois."""
    zdf_cond = zdf.loc[zdf.condition == condition]
    pivot = zdf_cond.pivot(index='subj_id', columns='roi', values='z')
    return pivot
