#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `plenoptomos` package."""


import unittest

import numpy as np
# import matplotlib.pyplot as plt

import os

import copy as cp

import plenoptomos as pleno


eps = np.finfo(np.float32).eps
atol = eps * 1e2

data_dir = 'examples/data/'
dpath = os.path.join(data_dir, 'flowers_plants_30_eslf.png')
jpath = os.path.join(data_dir, 'flowers_plants_30.json')

error_msg_lf_files = \
    'Please download the following files from the Stanford light-field archive and put them in %s:\n - %s\n - %s'

if not (os.path.exists(dpath) and os.path.exists(jpath)):
    print('Test image file or metadata do not exist.')
    base_url = 'http://lightfields.stanford.edu/images/flowers_plants/'
    dpath_url = base_url + 'raw/flowers_plants_30_eslf.png'
    jpath_url = base_url + 'metadata/flowers_plants_30.json'
    try:
        import urllib.request as ur

        print('They will now be downloaded:\n - %s' % dpath_url)
        ur.urlretrieve(dpath_url, dpath)
        print(' - %s' % jpath_url)
        ur.urlretrieve(jpath_url, jpath)
    except ImportError:
        raise ValueError(error_msg_lf_files % (data_dir, dpath_url, jpath_url))

lf = pleno.import_lf.from_lytro(dpath, jpath, source='eslf')


class TestPlenoptomos(unittest.TestCase):
    """Tests for `plenoptomos` package."""

    def __init__(self, *args, **kwrds):
        super().__init__(*args, **kwrds)

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""


class TestPsf(TestPlenoptomos):
    """Tests for `plenoptomos.psf` package."""

    def setUp(self):
        """Set up test fixtures, if any."""
        super().setUp()

        self.psf_ml_raw = pleno.psf.PSF.create_theo_psf(lf.camera, coordinates='vu', airy_rings=2)

        self.psf_ml_no = pleno.psf.PSFApply2D(psf_d=self.psf_ml_raw, use_otf=False)
        self.psf_ml_yo = pleno.psf.PSFApply2D(psf_d=self.psf_ml_raw, use_otf=True)

        self.raw_img = lf.get_raw_detector_picture()

    def test_001_useotf_direct(self):
        """Test PSF direct with use_otf=True."""

        conv_no = self.psf_ml_no.apply_psf_direct(cp.deepcopy(self.raw_img))
        conv_yo = self.psf_ml_yo.apply_psf_direct(cp.deepcopy(self.raw_img))

        success = np.all(np.isclose(conv_no, conv_yo, atol=atol))
        if not success:
            print('Max absolute deviation is: {}. '.format(np.max(np.abs(conv_no - conv_yo))),
                  end='', flush=True)
            # (ff, axf) = plt.subplots(2, 2, sharex=True, sharey=True)
            # axf[0, 0].imshow(conv_no)
            # axf[0, 1].imshow(conv_yo)
            # axf[1, 0].imshow(conv_no - conv_yo)
            # axf[1, 1].imshow((conv_no - conv_yo) > atol)
            # plt.show()

        assert success

    def test_002_useotf_adjoint(self):
        """Test PSF adjoint with use_otf=True."""

        conv_no = self.psf_ml_no.apply_psf_adjoint(cp.deepcopy(self.raw_img))
        conv_yo = self.psf_ml_yo.apply_psf_adjoint(cp.deepcopy(self.raw_img))

        success = np.all(np.isclose(conv_no, conv_yo, atol=atol))
        if not success:
            print('Max absolute deviation is: {}. '.format(np.max(np.abs(conv_no - conv_yo))),
                  end='', flush=True)
            # (ff, axf) = plt.subplots(2, 2, sharex=True, sharey=True)
            # axf[0, 0].imshow(conv_no)
            # axf[0, 1].imshow(conv_yo)
            # axf[1, 0].imshow(conv_no - conv_yo)
            # axf[1, 1].imshow((conv_no - conv_yo) > atol)
            # plt.show()

        assert success

    def test_003_useotf_direct_extended(self):
        """Test PSF direct on extended shapes with use_otf=True."""

        self.raw_img = self.raw_img[np.newaxis, ...]
        conv_no = self.psf_ml_no.apply_psf_direct(cp.deepcopy(self.raw_img))
        conv_yo = self.psf_ml_yo.apply_psf_direct(cp.deepcopy(self.raw_img))

        success = np.all(np.isclose(conv_no, conv_yo, atol=atol))
        if not success:
            print('Max absolute deviation is: {}. '.format(np.max(np.abs(conv_no - conv_yo))),
                  end='', flush=True)
            # (ff, axf) = plt.subplots(2, 2, sharex=True, sharey=True)
            # axf[0, 0].imshow(conv_no)
            # axf[0, 1].imshow(conv_yo)
            # axf[1, 0].imshow(conv_no - conv_yo)
            # axf[1, 1].imshow((conv_no - conv_yo) > atol)
            # plt.show()

        assert success


if __name__ == '__main__':
    unittest.main()
