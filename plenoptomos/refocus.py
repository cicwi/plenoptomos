#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains the implementation of the traditional reference refocusing algorithms.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France

Created on Thu Mar  2 18:48:28 2017
"""

import numpy as np
import scipy as sp

import numpy.fft as fft

import time as tm
from . import lightfield


def compute_refocus_integration(
        lf: lightfield.Lightfield, zs, beam_geometry='parallel', domain='object',
        up_sampling=1, border=4, border_padding='edge'):
    """Perform the refocusing of the input light field image using the integration method.

    This method was presented in:
    [1] R. Ng, et al., “Light Field Photography with a Hand-held Plenoptic Camera,”
    Stanford Univ. Tech. Rep. CSTR 2005-02, 2005.

    :param lf: The light-field object (lightfield.Lightfield)
    :param zs: Refocusing distances (numpy.array_like)
    :param up_sampling: Integer greater than 1 for up-sampling of the final images (int, default: 1)
    :param border: Number of pixels to extend the border and reduce darkening of edges (int, default: 4)
    :param border_padding: Border padding method (string, default: 'edge')
    :param beam_geometry: Beam geometry. Possible options: 'parallel' | 'cone' (string, default: 'parallel')
    :param domain: Refocusing domain. Possible options: 'object' | 'image' (string, default: 'object')

    :returns: Stack of 2D refocused images.
    :rtype: numpy.array_like
    """
    zs = np.array(zs)
    if len(zs.shape) == 0:
        zs = np.expand_dims(zs, 0)
    if domain.lower() == 'object':
        ref_z = lf.camera.get_focused_distance()
    elif domain.lower() == 'image':
        ref_z = lf.camera.z1
    else:
        raise ValueError('No known domain "%s"' % domain)
    alphas = zs / ref_z
    num_alphas = len(alphas)

    c_in = tm.time()
    m = lf.camera.get_focused_distance() / lf.camera.z1
    print('Refocusing through Integration %d alphas:\n- Domain = "%s", Beam geometry = "%s", Magnification = %g: ' % (
        num_alphas, domain, beam_geometry, m), end='', flush=True)

    # Refocusing operate on sub-aperture images:
    lf_sa = lf.clone()
    lf_sa.set_mode_subaperture()

    if lf_sa.mask is not None:
        lf_sa.data *= lf_sa.mask
        if lf_sa.flat is not None:
            lf_sa.flat *= lf_sa.mask

    if lf_sa.flat is None:
        renorm_sa_images = np.prod(lf_sa.camera.data_size_vu)
    else:
        lf_sa.data *= lf_sa.flat
        renorm_sa_images = lf_sa.get_photograph(image='flat') * np.prod(lf_sa.camera.data_size_vu)

    camera_sheared = lf_sa.camera.clone()
    if up_sampling > 1:
        camera_sheared.regrid(regrid_size=(1, 1, up_sampling, up_sampling))
        if lf_sa.flat is not None:
            tt = np.linspace(0.5, lf_sa.camera.data_size_ts[0]-0.5, lf_sa.camera.data_size_ts[0])
            ss = np.linspace(0.5, lf_sa.camera.data_size_ts[1]-0.5, lf_sa.camera.data_size_ts[1])

            interp_renorm = sp.interpolate.interp2d(
                tt, ss, renorm_sa_images, bounds_error=False, fill_value=np.mean(renorm_sa_images))

            tt = np.linspace(
                0.5 / up_sampling, lf_sa.camera.data_size_ts[0] - 0.5 / up_sampling,
                lf_sa.camera.data_size_ts[0] * up_sampling)
            ss = np.linspace(
                0.5 / up_sampling, lf_sa.camera.data_size_ts[1] - 0.5 / up_sampling,
                lf_sa.camera.data_size_ts[1] * up_sampling)

            renorm_sa_images = interp_renorm(tt, ss)

    imgs_size = (num_alphas, camera_sheared.data_size_ts[0], camera_sheared.data_size_ts[1])

    # Pad image:
    lf_sa.pad((0, 0, border * up_sampling, border * up_sampling), method=border_padding)
    lf_sa.pad(1)

    imgs = np.empty(imgs_size, lf_sa.data.dtype)

    (samp_v, samp_u, samp_t, samp_s) = lf_sa.camera.get_grid_points(space='direct', domain=domain)
    interp_lf4D = sp.interpolate.RegularGridInterpolator(
        (samp_v, samp_u, samp_t, samp_s), lf_sa.data, bounds_error=False, fill_value=0)

    # Computing the changes of base, and integrating (for each alpha):
    for ii in range(num_alphas):
        prnt_str = "%d/%d (alpha = %g, z = %g)" % (ii+1, num_alphas, alphas[ii], zs[ii])
        print(prnt_str, end='', flush=True)
        # Change of base:
        sheared_coords = camera_sheared.get_sheared_coords(
            alphas[ii], space='direct', beam_geometry=beam_geometry, domain=domain)
        sheared_lf_data = interp_lf4D(sheared_coords)

        # Integrate:
        imgs[ii, ...] = np.sum(sheared_lf_data, axis=(0, 1))

        print(('\b') * len(prnt_str), end='', flush=True)

    # Renormalizing:
    imgs /= renorm_sa_images

    c_out = tm.time()
    print("Done in %g seconds." % (c_out - c_in))

    return imgs


def compute_refocus_fourier(
        lf: lightfield.Lightfield, zs, method='slice', beam_geometry='parallel', domain='object',
        up_sampling=1, border=4, border_padding='edge', padding_factor=1.1, oversampling=2):
    """Perform the refocusing of the input light field image using Fourier slice theorem method.

    This method was presented in:
    [1] R. Ng, “Fourier slice photography,” ACM Trans. Graph., vol. 24, no. 3, p. 735, 2005.

    :param lf: The light-field object (lightfield.Lightfield)
    :param zs: Refocusing distances (numpy.array_like)
    :param up_sampling: Integer greater than 1 for up-sampling of the final images (int, default: 1)
    :param border: Number of pixels to extend the border and reduce darkening of edges (int, default: 4)
    :param border_padding: Border padding method (string, default: 'edge')
    :param beam_geometry: Beam geometry. Possible options: 'parallel' | 'cone' (string, default: 'parallel')
    :param domain: Refocusing domain. Possible options: 'object' | 'image' (string, default: 'object')
    :param method: Fourier space operation method. Options: 'slice' | 'full', 'hyper_cone' (string, default: 'slice')
    :param padding: Real-space padding factor (float, default: 1.1)
    :param oversampling: Fourier space oversampling factor (float, default: 2)

    :returns: Stack of 2D refocused images.
    :rtype: numpy.array_like
    """

    zs = np.array(zs)
    if len(zs.shape) == 0:
        zs = np.expand_dims(zs, 0)
    if domain.lower() == 'object':
        ref_z = lf.camera.get_focused_distance()
    elif domain.lower() == 'image':
        ref_z = lf.camera.z1
    else:
        raise ValueError('No known domain "%s"' % domain)
    alphas = zs / ref_z
    num_alphas = len(alphas)

    m = lf.camera.get_focused_distance() / lf.camera.z1
    print('Refocusing through Fourier %d alphas:\n- Domain = "%s", Beam geometry = "%s", Magnification = %g, Method = %s:' % (
        num_alphas, domain, beam_geometry, m, method))
    c_in = tm.time()

    lf_sa = lf.clone()
    lf_sa.set_mode_subaperture()

    if lf_sa.mask is not None:
        lf_sa.data *= lf_sa.mask
        if lf_sa.flat is not None:
            lf_sa.flat *= lf_sa.mask

    if lf.flat is None:
        renorm_sa_images = np.prod(lf.camera.data_size_vu)
    else:
        lf_sa.data *= lf_sa.flat
        renorm_sa_images = lf.get_photograph(image='flat') * np.prod(lf.camera.data_size_vu)

    paddings_ts = (np.ceil(np.array(lf_sa.camera.data_size_ts) * (padding_factor - 1) / 4) * 2).astype(np.int32)
    paddings_vu = np.ceil(np.array(lf_sa.camera.data_size_vu) / 2).astype(np.int32)
    print("- Padding, border = %d, (v, u, t, s) = (%d, %d, %d, %d).." % (
        border, paddings_vu[0], paddings_vu[1], paddings_ts[0], paddings_ts[1]),
        end='', flush=True)
    # Pad image:
    lf_sa.pad((0, 0, border, border), method=border_padding)
    lf_sa.pad((paddings_vu[0], paddings_vu[1], paddings_ts[0], paddings_ts[1]))

    data_fft4 = lf_sa.data

    c_pad = tm.time()
    print("\b\b: Done in %g seconds.\n- Performing 4D FFT, dimensions: " % (c_pad - c_in), end='', flush=True)
    # Computing the 4D Fourier transform:
    data_fft4 = fft.ifftshift(data_fft4)
    for ii in range(4):
        prnt_str = "%d/%d" % (ii+1, 4)
        print(prnt_str, end='', flush=True)
        data_fft4 = fft.fftn(data_fft4, axes=(ii, ))
        print(('\b') * len(prnt_str), end='', flush=True)
    data_fft4 = fft.fftshift(data_fft4)

    c_fft = tm.time()
    print("\b\b: Done in %g seconds.\n- Creating interpolation grid.." % (c_fft - c_pad), end='', flush=True)

    base_grid = lf_sa.camera.get_grid_points(space='fourier', domain=domain)
    interp_lf4D = sp.interpolate.RegularGridInterpolator(base_grid, data_fft4, bounds_error=False, fill_value=0)

    camera_sheared = lf_sa.camera.clone()
    if up_sampling > 1:
        camera_sheared.regrid(regrid_size=(1, 1, up_sampling, up_sampling))

    imgs_size = (num_alphas, camera_sheared.data_size_ts[0] * oversampling, camera_sheared.data_size_ts[1] * oversampling)

    imgs = np.empty(imgs_size, data_fft4.dtype)
    corr = np.empty(imgs_size, data_fft4.dtype)

    c_cre = tm.time()
    print("\b\b: Done in %g seconds.\n- Performing 2D interpolation: " % (c_cre - c_fft), end='', flush=True)

    (scale_t, scale_s, scale_v, scale_u) = np.array(lf_sa.camera.get_scales(space='fourier', domain=domain))
    scales = np.array((scale_v, scale_u, scale_t, scale_s))[None, None, :]

    # Computing the changes of base, and integrating (for each alpha):
    if method.lower() == 'slice':
        for ii in range(num_alphas):
            prnt_str = "%d/%d (alpha = %g, z = %g)" % (ii+1, num_alphas, alphas[ii], zs[ii])
            print(prnt_str, end='', flush=True)

            # Change of base:
            sheared_coords = camera_sheared.get_sheared_coords(
                alphas[ii], space='fourier_slice', beam_geometry=beam_geometry, domain=domain, oversampling=oversampling)
            imgs[ii, ...] = interp_lf4D(sheared_coords)

            dists = 1 - np.abs(sheared_coords) / scales
            corr[ii, ...] = np.prod(dists, axis=2) * np.all(dists > 0, axis=2)

            print(('\b') * len(prnt_str), end='', flush=True)

    elif method.lower() in ('full', 'hyper_cone'):
        slice_00 = camera_sheared.get_sheared_coords(
            np.array((1, )), space='fourier_slice', beam_geometry=beam_geometry, domain=domain, oversampling=oversampling)
        for ii in range(num_alphas):
            prnt_str = "%d/%d (alpha = %g, z = %g)" % (ii+1, num_alphas, alphas[ii], zs[ii])
            print(prnt_str, end='', flush=True)

            # Change of base:
            if method.lower() == 'full':
                sheared_coords = camera_sheared.get_sheared_coords(
                    alphas[ii], space='fourier', beam_geometry=beam_geometry, domain=domain)
                sheared_lf = interp_lf4D(sheared_coords)
            elif method.lower() == 'hyper_cone':
                sheared_lf = data_fft4 * camera_sheared.get_filter(alphas[ii], beam_geometry=beam_geometry, domain=domain)
                sheared_lf = fft.ifftshift(sheared_lf, axes=(0, 1))
                sheared_lf = fft.ifftn(sheared_lf, axes=(0, 1))

            interp_obj = sp.interpolate.RegularGridInterpolator(
                base_grid, sheared_lf, bounds_error=False, fill_value=0)
            imgs[ii, ...] = interp_obj(slice_00)

            print(('\b') * len(prnt_str), end='', flush=True)

    c_int = tm.time()
    print("\b\b: Done in %g seconds.\n- Performing 2D FFT.." % (c_int - c_cre), end='', flush=True)
    imgs = fft.ifftshift(imgs, axes=(1, 2))
    imgs = fft.ifftn(imgs, axes=(1, 2))
    imgs = np.real(imgs).astype(lf_sa.data.dtype)
    imgs = fft.fftshift(imgs, axes=(1, 2))

    # Correcting roll-off
    corr = fft.ifftshift(corr, axes=(1, 2))
    corr = fft.ifftn(corr, axes=(1, 2))
    corr = np.real(corr).astype(lf_sa.data.dtype)
    corr = fft.fftshift(corr, axes=(1, 2))
    corr /= corr.max()

    imgs /= corr + (corr == 0)

    if oversampling > 1:
        borders_os = (np.array(imgs.shape[1:]) - camera_sheared.data_size_ts) / 2
        borders_os = np.concatenate((np.ceil(borders_os), np.floor(borders_os))).astype(np.int32)
        imgs = imgs[:, borders_os[0]:-borders_os[2], borders_os[1]:-borders_os[3]]

    # Crop output images:
    tot_paddings_ts = (paddings_ts + border) * up_sampling
    imgs = imgs[:, tot_paddings_ts[0]:-tot_paddings_ts[0], tot_paddings_ts[1]:-tot_paddings_ts[1]]
    # Renormalizing:
    if method.lower() in ('slice', 'full'):
        imgs *= up_sampling ** 2 / renorm_sa_images

    c_out = tm.time()
    print("\b\b: Done in %g seconds.\nDone in %g seconds." % (c_out - c_int, c_out - c_in))

    return imgs
