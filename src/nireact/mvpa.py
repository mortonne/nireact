"""Measures for use with pymvpa2."""

import os
import numpy as np

from mvpa2.datasets.mri import fmri_dataset
from mvpa2.measures.base import Measure
import mindstorm.subjutil as su
from nireact import rsa


def load_disp_beta(sp, beta_suffix, mask, feature_mask=None, verbose=1):
    """Load display task beta series as a dataset."""

    # file with beta series data
    beta_dir = os.path.join(sp.study_dir, 'batch', 'glm',
                            'disp' + beta_suffix, 'beta')
    beta_file = su.impath(beta_dir, sp.subject + '_beta')
    if not os.path.exists(beta_file):
        raise IOError(f'Beta series file does not exist: {beta_file}')
    if verbose:
        print(f'Loading beta series data from: {beta_file}')

    # mask image to select voxels to load
    mask_file = sp.image_path('anatomy', 'bbreg', 'data', mask)
    if not os.path.exists(mask_file):
        raise IOError(f'Mask file does not exist: {mask_file}')
    if verbose:
        print(f'Masking with: {mask_file}')

    if feature_mask is not None:
        # load feature mask
        feature_file = sp.image_path('anatomy', 'bbreg', 'data', feature_mask)
        if not os.path.exists(feature_file):
            raise IOError(f'Feature mask does not exist: {feature_file}')
        if verbose:
            print(f'Using features within: {feature_file}')

        # label voxels with included flag
        ds = fmri_dataset(beta_file, mask=mask_file,
                          add_fa={'include': feature_file})
        ds.fa.include = ds.fa.include.astype(bool)
    else:
        # mark all voxels as included
        ds = fmri_dataset(beta_file, mask=mask_file)
        ds.fa['include'] = np.ones(ds.shape[1], dtype=bool)

    return ds


class TriadDelta(Measure):
    """Test for similarity changes within and between triads."""

    def __init__(self, vols, n_perm=1000):
        Measure.__init__(self)
        self.vols = vols
        self.cond_ind = rsa.prep_triad_delta(vols, n_perm)

    def _call(self, ds):
        pass

    def __call__(self, dataset):
        if np.count_nonzero(dataset.fa.include) < 10:
            return 0, 0, 0

        zstat = rsa.zstat_triad_delta(dataset.samples, self.vols, self.cond_ind)
        return zstat['block'], zstat['inter'], 1


class TriadVector(Measure):
    """Test for analogous structure between items in different triads."""

    def __init__(self, vols, item_comp, stat, contrasts, n_perm=1000,
                 min_voxels=10):
        Measure.__init__(self)

        # get indices of items to compare
        if stat == 'vector':
            perm_type = 'item_type'
        elif stat == 'vectri':
            perm_type = 'triad'
        else:
            raise ValueError(f'Unknown stat: {stat}')
        train_ind, test_ind = rsa.prep_triad_vector(vols, item_comp, n_perm,
                                                    perm_type=perm_type)
        self.train_ind = train_ind
        self.test_ind = test_ind
        self.stat = stat
        self.contrasts = contrasts
        self.min_voxels = min_voxels

    def _call(self, ds):
        pass

    def __call__(self, dataset):

        if np.count_nonzero(dataset.fa.include) < self.min_voxels:
            return tuple(np.zeros(len(self.contrasts) + 1))
        dataset = dataset[:, dataset.fa.include]

        # calculate z-statistics
        zstat = rsa.zstat_triad_vector(dataset.samples, self.train_ind,
                                       self.test_ind, 'err')

        # pass out requested contrasts in specified order
        z_list = [zstat[c] for c in self.contrasts]
        if np.any(np.isnan(z_list)):
            raise ValueError('Undefined statistic.')

        z_list.append(1)
        return tuple(z_list)
