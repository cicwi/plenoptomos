#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `plenoptomos` package."""


import unittest

import numpy as np
#import matplotlib.pyplot as plt

import os

import plenoptomos as pleno

data_dir = 'examples/data/'
dpath = os.path.join(data_dir, 'flowers_plants_30_eslf.png')
jpath = os.path.join(data_dir, 'flowers_plants_30.json')

error_msg_lf_files = 'Please download the following files from the Stanford light-field archive and put them in %s:\n - %s\n - %s'

class TestPlenoptomos(unittest.TestCase):
    """Tests for `plenoptomos` package."""

    def setUp(self):
        """Set up test fixtures, if any."""
        if not os.path.exists(dpath):
            print('Test image file: "%s" does not exist.' % (dpath))
            dpath_url = 'http://lightfields.stanford.edu/images/flowers_plants/raw/flowers_plants_30_eslf.png'
            jpath_url = 'http://lightfields.stanford.edu/images/flowers_plants/metadata/flowers_plants_30.json'
            try:
                import urllib.request as ur

                print('They will now be downloaded:\n - %s' % dpath_url)
                ur.urlretrieve(dpath_url, dpath)
                print(' - %s' % jpath_url)
                ur.urlretrieve(jpath_url, jpath)
            except ImportError:
                raise ValueError(error_msg_lf_files % (data_dir, dpath_url, jpath_url))

#        (self.lf_r ,self.lf_g, self.lf_b) = pleno.import_lf.from_lytro(dpath, jpath, source='eslf', mode='rgb')
        self.lf = pleno.import_lf.from_lytro(dpath, jpath, source='eslf')

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_001_psf_useotf(self):
        """Test the correctness of the use_otf=True option."""
        psf_ml_raw = pleno.psf.PSF.create_theo_psf(self.lf.camera, coordinates='vu', airy_rings=2)

        psf_ml_no = pleno.psf.PSFApply2D(psf_d=psf_ml_raw, use_otf=False)
        psf_ml_yo = pleno.psf.PSFApply2D(psf_d=psf_ml_raw, use_otf=True)

        raw_img = self.lf.get_raw_detector_picture()

        conv_no = psf_ml_no.apply_psf_direct(raw_img)
        conv_yo = psf_ml_yo.apply_psf_direct(raw_img)

#        (ff, axf) = plt.subplots(1, 3, sharex=True, sharey=True)
#        axf[0].imshow(conv_no)
#        axf[1].imshow(conv_yo)
#        axf[2].imshow(conv_no - conv_yo)
#        plt.show()

        self.assertTrue(np.all((conv_no - conv_yo) == 0))

if __name__ == '__main__':
    unittest.main()