#!/usr/bin/env python
#
# Export display task data from an ROI.

import os
import numpy as np

import mindstorm.subjutil as su
from nireact import task, mvpa


def main(subject, study_dir, mask, suffix='_stim_fix2'):

    from mvpa2.mappers.zscore import zscore

    # load subject data
    sp = su.SubjPath(subject, study_dir)
    vols = task.disp_vols(sp.path('behav', 'log'))

    # load fmri data
    ds = mvpa.load_disp_beta(sp, suffix, mask, verbose=1)

    # zscore
    ds.sa['run'] = vols.run.values
    zscore(ds, chunks_attr='run')

    # save data samples and corresponding volume information
    res_dir = os.path.join(sp.study_dir, 'batch', 'glm', 'disp' + suffix,
                           'roi', mask)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    mat_file = os.path.join(res_dir, f'pattern_{subject}.txt')
    tab_file = os.path.join(res_dir, f'pattern_{subject}.csv')
    np.savetxt(mat_file, ds.samples)
    vols.to_csv(tab_file)


if __name__ == '__main__':
    
    parser = su.SubjParser(include_log=False)
    parser.add_argument('mask', help="name of mask file")
    parser.add_argument('--suffix', '-b', default='_stim_fix2',
                        help="suffix for beta images (_stim_fix2)")
    args = parser.parse_args()

    main(args.subject, args.study_dir, args.mask, suffix=args.suffix)
