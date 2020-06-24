#!/usr/bin/env python
#
# Run a searchlight to evaluate a triad statistic.

import os
import numpy as np

import mindstorm.subjutil as su
from nireact import task


def main(subject, study_dir, mask, stat, res_name, items='ac',
         suffix='_stim_fix2', feature_mask=None, radius=3, n_perm=1000,
         n_proc=None):

    from mvpa2.datasets.mri import map2nifti
    from mvpa2.mappers.zscore import zscore
    from mvpa2.measures.searchlight import sphere_searchlight
    from nireact import mvpa

    # lookup subject directory
    sp = su.SubjPath(subject, study_dir)

    # load task information
    vols = task.disp_vols(sp.path('behav', 'log'))

    # unpack the items option to get item type code
    item_names = 'abc'
    item_comp = [item_names.index(name) + 1 for name in items]

    # get post runs, A and C items only
    include = ((vols.run >= 5) & np.isin(vols.item_type, item_comp) &
               vols.correct == 1)
    post = vols.loc[include, :]

    # load beta series
    ds = mvpa.load_disp_beta(sp, suffix, mask, feature_mask, verbose=1)

    # define measure and contrasts to write out
    contrasts = ['pos', 'block_inter', 'inter_block']
    m = mvpa.TriadVector(post, item_comp, stat, contrasts, n_perm)

    # zscore
    ds.sa['run'] = vols.run.values
    zscore(ds, chunks_attr='run')

    # searchlight
    print('Running searchlight...')
    sl = sphere_searchlight(m, radius=radius, nproc=n_proc)
    sl_map = sl(ds[include])

    # save results
    nifti_include = map2nifti(ds, sl_map[-1])
    for i, contrast in enumerate(contrasts):
        res_dir = sp.path('rsa', f'{res_name}_{contrast}')
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)

        nifti = map2nifti(ds, sl_map[i])
        nifti.to_filename(su.impath(res_dir, 'zstat'))

        nifti_include.to_filename(su.impath(res_dir, 'included'))


if __name__ == '__main__':
    parser = su.SubjParser(include_log=False)
    parser.add_argument('mask', help="name of mask file")
    parser.add_argument('stat',
                        help="type of statistic to calculate (vector, vectri)"),
    parser.add_argument('res_name', help="basename of results directory")
    parser.add_argument('--items', '-i', default='ac',
                        help="items to compare (['ac'],'ab','bc')")
    parser.add_argument('--suffix', '-b', default='_stim_fix2',
                        help="suffix for beta images (_stim_fix2)")
    parser.add_argument('--feature-mask', '-f', default=None,
                        help="name of mask of voxels to include as features")
    parser.add_argument('--radius', '-r', type=int, default=3,
                        help="searchlight radius (3)")
    parser.add_argument('--n-perm', '-p', type=int, default=1000,
                        help="number of permutations to run (1000)")
    parser.add_argument('--n-proc', '-n', type=int, default=None,
                        help="processes for searchlight (None)")
    args = parser.parse_args()

    main(args.subject, args.study_dir, args.mask, args.stat, args.res_name,
         items=args.items, suffix=args.suffix, feature_mask=args.feature_mask,
         radius=3, n_perm=args.n_perm, n_proc=args.n_proc)
