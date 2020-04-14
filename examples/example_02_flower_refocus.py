#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 13:26:03 2019

@author: vigano
"""

import numpy as np
import matplotlib.pyplot as plt

import os

import plenoptomos as pleno

data_dir = 'examples/data/'
dpath = os.path.join(data_dir, 'flowers_plants_30_eslf.png')
jpath = os.path.join(data_dir, 'flowers_plants_30.json')

error_msg_lf_files = \
    'Please download the following files from the Stanford light-field archive and put them in %s:\n - %s\n - %s'

if not os.path.exists(dpath):
    print('Example files: "%s" and "%s" do not exist.' % (dpath, jpath))
    dpath_url = 'http://lightfields.stanford.edu/images/flowers_plants/raw/flowers_plants_30_eslf.png'
    jpath_url = 'http://lightfields.stanford.edu/images/flowers_plants/metadata/flowers_plants_30.json'
    try:
        import urllib.request as ur

        print('They will be now downloaded:\n - %s' % dpath_url)
        ur.urlretrieve(dpath_url, dpath)
        print(' - %s' % jpath_url)
        ur.urlretrieve(jpath_url, jpath)
    except ImportError:
        raise ValueError(error_msg_lf_files % (data_dir, dpath_url, jpath_url))

print('Importing the light-field from the Lytro eslf format..')
(lf_r, lf_g, lf_b) = pleno.import_lf.from_lytro(dpath, jpath, source='eslf', mode='rgb')

print('Creating the theoretical PSFs for the different color channels..')
psf_ml_r = pleno.psf.PSF.create_theo_psf(lf_r.camera, coordinates='vu', airy_rings=2)
psf_ml_r = pleno.psf.PSFApply2D(psf_d=psf_ml_r)

psf_ml_g = pleno.psf.PSF.create_theo_psf(lf_g.camera, coordinates='vu', airy_rings=2)
psf_ml_g = pleno.psf.PSFApply2D(psf_d=psf_ml_g)

psf_ml_b = pleno.psf.PSF.create_theo_psf(lf_b.camera, coordinates='vu', airy_rings=2)
psf_ml_b = pleno.psf.PSFApply2D(psf_d=psf_ml_b)

print('Computing refocusing distances..')
z0 = lf_r.camera.get_focused_distance()
alphas_con = np.linspace(0.5, 3.0, 46)
alphas_par = lf_r.camera.get_alphas(alphas_con, beam_geometry_in='cone', beam_geometry_out='parallel')
z0s = z0 * alphas_par

# we choose only the 3 most interesting ones
dists = [6, 10, 21]
z0s_sel = z0s[np.r_[dists]]


# Convenience function for hangling RGB images
def refocus_rgb(refocus_func, renorm=False):
    imgs_r = refocus_func(lf_r, psf_ml_r)
    imgs_g = refocus_func(lf_g, psf_ml_g)
    imgs_b = refocus_func(lf_b, psf_ml_b)
    if renorm:
        lf_ones = lf_r.clone()
        lf_ones.data = np.ones_like(lf_ones.data)
        imgs_ones = refocus_func(lf_ones, psf_ml_r)
        imgs_r /= imgs_ones
        imgs_g /= imgs_ones
        imgs_b /= imgs_ones
    return pleno.colors.merge_rgb_images(imgs_r, imgs_g, imgs_b)


print('Refocusing with Integration...')
refocus_int = lambda x, _: pleno.refocus.compute_refocus_integration(
    x, z0s_sel, beam_geometry='parallel')
imgs_int = refocus_rgb(refocus_int)

print('Refocusing with Back-projection...')
refocus_bpj = lambda x, _: pleno.tomo.compute_refocus_iterative(
    x, z0s_sel, beam_geometry='parallel', algorithm='bpj')
imgs_bpj = refocus_rgb(refocus_bpj)

print('Refocusing with SIRT without PSF...')
refocus_sirt = lambda x, _: pleno.tomo.compute_refocus_iterative(
    x, z0s_sel, beam_geometry='parallel', iterations=3, algorithm='sirt')
imgs_sirt = refocus_rgb(refocus_sirt)

print('Refocusing with CP-LS-TV with PSF...')
algo = pleno.solvers.CP_tv(data_term='l2', lambda_tv=1e-1, axes=(-2, -1))
refocus_cplstv_p = lambda x, p: pleno.tomo.compute_refocus_iterative(
    x, z0s_sel, beam_geometry='parallel', iterations=50, algorithm=algo, psf=p)
imgs_cplstv_p = refocus_rgb(refocus_cplstv_p)

(f, axs) = plt.subplots(len(dists), 4, sharex=True, sharey=True, squeeze=False)
axs[0, 0].set_title('Integration')
axs[0, 1].set_title('Back-projection')
axs[0, 2].set_title('SIRT w/o PSF')
axs[0, 3].set_title('CP-LS-TV w/ PSF')
for ii, d in enumerate(dists):
    axs[ii, 0].imshow(imgs_int[ii, ...])
    axs[ii, 1].imshow(imgs_bpj[ii, ...])
    axs[ii, 2].imshow(imgs_sirt[ii, ...])
    axs[ii, 3].imshow(imgs_cplstv_p[ii, ...])

plt.tight_layout()
plt.show()
