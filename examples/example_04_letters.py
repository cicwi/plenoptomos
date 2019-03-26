#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 13:26:37 2019

@author: vigano

The used dataset was collected in the labs of Imagine Optic (Bordeaux, France.
Website: https://www.imagine-optic.com/).
The data collection has been performed by Charlotte Herzog, Pablo Martinez Gil,
and Nicola Viganò.
"""

import numpy as np
import matplotlib.pyplot as plt

import os

import plenoptomos as pleno

data_dir = 'examples/data/'

# This file contains the collected light-field data
raw_file = os.path.join(data_dir, 'letters_ULF_M1.png')

# This file contains the flat-field: it is also used for calibrating the light-field data
raw_file_white = os.path.join(data_dir, 'letters_ULF_M1_white.png')
# This file contains the dark-field: it provides the background, but currently not used
raw_file_dark = os.path.join(data_dir, 'letters_ULF_M1_dark.png')
# This file contains the metadata in ini format. The names and sections follow
# the description of the vox data format
ini_file = os.path.join(data_dir, 'letters_ULF_M1.ini')

vox_file_uncal = os.path.join(data_dir, 'letters_ULF_M1_uncalibrated.vox')
vox_file = os.path.join(data_dir, 'letters_ULF_M1.vox')

print('Creating the uncalibrated vox file..')
pleno.data_format.create_vox_from_raw(raw_file, ini_file, raw_det_white=raw_file_white, raw_det_dark=raw_file_dark, out_file=vox_file_uncal)

# The calibration is interactive, and it requires user intervention
# In both dimensions, the first and last peaks should be discarded.
# The expected fitted values should be:
#pitch = [48.2, 48.2273]
#offset = [46.5125, 30.5504]
print('Calibrating the lenslet data..')
pleno.data_format.calibrate_raw_image(vox_file_uncal, vox_file_out=vox_file) #, pitch=pitch, offset=offset

# Now the file is calibrated, and it can be loaded as a light-field
print('Loading the light-field..')
lfv = pleno.data_format.load(vox_file)

z0 = lfv.camera.get_focused_distance()

psf_ml_raw = pleno.psf.PSF.create_theo_psf(lfv.camera, coordinates='vu', airy_rings=1)
psf_ml = pleno.psf.PSFApply2D(psf_d=psf_ml_raw)

print('Computing refocusing distances..')
#alphas_con = np.array((0.95, 1.0))
alphas_con = np.array((0.95, ))
alphas_par = lfv.camera.get_alphas(alphas_con, beam_geometry_in='cone', beam_geometry_out='parallel')
z0s = z0 * alphas_par

print('Refocusing with Back-projection...')
imgs_bpj = pleno.tomo.compute_refocus_iterative(lfv, z0s, beam_geometry='parallel', algorithm='bpj')

print('Refocusing with SIRT without PSF...')
imgs_sirt = pleno.tomo.compute_refocus_iterative(lfv, z0s, beam_geometry='parallel', iterations=3, algorithm='sirt')

print('Refocusing with CP-LS-TV with PSF...')
algo = pleno.solvers.CP_tv(data_term='l2', lambda_tv=1e-1, axes=(-2, -1))
imgs_cplstv_p = pleno.tomo.compute_refocus_iterative(lfv, z0s, beam_geometry='parallel', iterations=50, algorithm=algo, psf=psf_ml)

(f, axs) = plt.subplots(1, 3, sharex=True, sharey=True)
axs[0].set_title('Back-projection')
axs[1].set_title('SIRT w/o PSF')
axs[2].set_title('CP-LS-TV w/ PSF')

axs[0].imshow(imgs_bpj[0, ...])
axs[1].imshow(imgs_sirt[0, ...])
axs[2].imshow(imgs_cplstv_p[0, ...])

plt.tight_layout()
plt.show()

