#!/usr/bin/env python
#
# Evaluate a statistic on display task data for an ROI.

import os
import numpy as np

import mindstorm.subjutil as su
from nireact import rsa


def main(subject, study_dir, roi, res_name, stat, items='ac',
         suffix='_stim_fix2', n_perm=10000):

    # load ROI data and volume information
    input_dir = os.path.join(study_dir, 'batch', 'glm', 'disp' + suffix,
                             'roi', roi)
    patterns, vols = rsa.load_roi_pattern(input_dir, subject)

    # unpack the items option to get item type code
    item_names = 'abc'
    item_comp = [item_names.index(name) + 1 for name in items]

    if stat in ['vector', 'vectri']:
        include = ((vols.run >= 5) &
                   np.isin(vols.item_type, item_comp) &
                   (vols.correct == 1))
        if stat == 'vector':
            # within triad permutation
            train_ind, test_ind = rsa.prep_triad_vector(vols.loc[include],
                                                        item_comp, exhaust=True,
                                                        perm_type='item_type')
        else:
            # across triad permutation
            train_ind, test_ind = rsa.prep_triad_vector(vols.loc[include],
                                                        item_comp, n_perm,
                                                        perm_type='triad')

        # prediction error
        zstat = rsa.zstat_triad_vector(patterns[include, :], train_ind,
                                       test_ind, 'err')

    elif stat in ['item_type', 'triad']:
        include = ((vols.run >= 5) &
                   np.isin(vols.item_type, item_comp) &
                   (vols.correct == 1))
        n = n_perm if stat == 'triad' else None
        cond_ind = rsa.prep_triad_sim(vols.loc[include], item_comp, stat, n)
        zstat = rsa.zstat_triad_sim(patterns[include, :], cond_ind)

    else:
        raise ValueError(f'Statistic undefined: {stat}')

    # save results
    res_dir = os.path.join(study_dir, 'batch', 'rsa', res_name, 'roi', roi)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    res_file = os.path.join(res_dir, f'zstat_{subject}.txt')
    contrasts = ['pos', 'neg', 'block', 'inter',
                 'block_neg', 'inter_neg', 'block_inter', 'inter_block']
    mat = np.hstack([zstat[name] for name in contrasts])

    np.savetxt(res_file, mat)


if __name__ == '__main__':

    parser = su.SubjParser(include_log=False)
    parser.add_argument('roi', help="name of roi")
    parser.add_argument('res_name', help="name of results directory")
    parser.add_argument('stat',
                        help="type of statistic to calculate (vector, vectri,"
                             "item_type, triad)")
    parser.add_argument('--items', '-i', default='ac',
                        help="items to compare (['ac'],'ab','bc')")
    parser.add_argument('--suffix', '-b', default='_stim_fix2',
                        help="suffix for beta images (_stim_fix2)")
    parser.add_argument('--n-perm', '-p', type=int, default=10000,
                        help="number of permutations to run (10000)")
    args = parser.parse_args()

    main(args.subject, args.study_dir, args.roi, args.res_name,
         args.stat, items=args.items, suffix=args.suffix, n_perm=args.n_perm)
