"""Test vector difference analyses."""

import unittest

import numpy as np

from mistr import task, rsa


class VectorTestCase(unittest.TestCase):
    def setUp(self):
        # set up minimal test case
        self.n_triad = 2
        self.n_train = 2
        self.n_item_type = 3
        self.n_item = self.n_triad * self.n_item_type * self.n_train
        items = task.task_items(self.n_triad, self.n_train)
        items_filt = items.loc[items.item_type != 2]

        # expand item information to fill out runs
        vols = items_filt.iloc[np.tile(np.arange(len(items_filt)), 4)].copy()
        vols['run'] = np.repeat(np.arange(1, 5), len(items_filt))
        self.vols = vols

        # generate permutation test indices for vector statistic
        train_ind, test_ind = rsa.prep_triad_vector(self.vols, (1, 3))
        self.train_ind = train_ind
        self.test_ind = test_ind

    def test_ind_shape(self):
        """Test correct size of indices."""
        self.assertEqual(self.train_ind[0].shape, (16, 2, 6, 2, 2))
        self.assertEqual(self.test_ind[0].shape, (16, 2, 6, 2, 2))

    def test_ind_type(self):
        """Test that item type is in correct order."""
        item_type = self.vols.item_type.values[self.train_ind[0][0]]
        correct_item_type = [[[[1, 3]] * 2] * 6] * 2
        np.testing.assert_array_equal(item_type, correct_item_type)

    def test_ind_triad(self):
        """Test ordering of triads in indices."""
        triad = self.vols.triad.values[self.train_ind[0][0]]
        np.testing.assert_array_equal(triad[0], 1)
        np.testing.assert_array_equal(triad[1], 2)

    def test_ind_unique(self):
        """Test that run is unique within cross-validation fold."""
        train_run = self.vols.run.values[self.train_ind[0][0]]
        test_run = self.vols.run.values[self.test_ind[0][0]]
        assert (train_run != test_run).all()

    def test_ind_perm(self):
        """Test effects of permutation indices."""
        item_type = self.vols.item_type.values
        run = self.vols.run.values
        triad = self.vols.triad.values
        ind = self.train_ind[0]

        # permutation should alter item type
        assert (item_type[ind[0]] != item_type[ind[-1]]).any()

        # permutation should not alter run or triad
        np.testing.assert_array_equal(run[ind[0]], run[ind[-1]])
        np.testing.assert_array_equal(triad[ind[0]], triad[ind[-1]])
