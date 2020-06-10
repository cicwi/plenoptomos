#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module implements the depth estimation routines, that allow to compute
depth-maps from light-fields.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France

Created on Wed Mar 29 12:34:36 2017
"""

import numpy as np
import scipy as sp
import scipy.ndimage as spimg
import scipy.signal as sps

from . import lightfield
from . import solvers
from . import utils_proc as proc

from .tomo import Projector

import time as tm

import warnings

try:
    import pywt
    has_pywt = True
    use_swtn = pywt.version.version >= '1.0.2'
except ImportError:
    has_pywt = False
    use_swtn = False
    print('WARNING - pywt was not found')


def _apply_smoothing_filter(arr, window_filter, mask=None, mask_renorm=1):
    if mask is not None:
        return sps.convolve(np.squeeze(arr) * mask, window_filter, mode='same') / mask_renorm
    else:
        return sps.convolve(np.squeeze(arr), window_filter, mode='same')


def _compute_depth_and_confidence(depth_cue, confidence_method, peak_range=2, med_filt_size=3):
    cue_map_size = depth_cue.shape[1:]
    num_zs = depth_cue.shape[0]

    response_funcs = np.reshape(depth_cue, (num_zs, -1))

    num_pixels = response_funcs.shape[1]

    peaks_val = np.max(response_funcs, axis=0)
    peaks_pos = np.argmax(response_funcs, axis=0)

    if confidence_method.lower() == 'integral':
        bckground = np.min(response_funcs, axis=0)
        peaks_val -= bckground
        response_funcs -= bckground

        integral_conf = np.sum(response_funcs, axis=0) - peaks_val

        invalid_vals = peaks_val == 0
        confidence = integral_conf / (peaks_val * (num_zs-1) + invalid_vals)
        confidence = np.fmax(1 - confidence, 0) * (1 - invalid_vals)

    elif confidence_method.lower() == '2nd_peak':
        peaks_val_second = np.zeros((num_pixels, ), dtype=response_funcs.dtype)

        for ii in range(num_pixels):
            peaks = sps.find_peaks(response_funcs[:, ii], peaks_val[ii] / 10)
            if len(peaks[0]) > 1:
                peaks_val_second[ii] = response_funcs[peaks[0][1], ii]
            elif len(peaks[0]) == 1:
                peaks_val_second[ii] = 0
            elif len(peaks[0]) == 0:
                peaks_val_second[ii] = peaks_val[ii]

        confidence = 1 - peaks_val_second / peaks_val

    elif confidence_method.lower() == 'neighbor_stddev':
        confidence = np.empty(num_pixels, dtype=depth_cue.dtype)

        peak_ranges_min = np.fmax(peaks_pos - peak_range, 0).astype(np.intp)
        peak_ranges_max = np.fmin(peaks_pos + peak_range, num_zs-1).astype(np.intp)

        for ii, pmin, pmax in zip(range(num_pixels), peak_ranges_min, peak_ranges_max):
            pnt_pos = np.concatenate((np.arange(pmin, peaks_pos[ii]), np.arange(peaks_pos[ii]+1, pmax+1)))
            diffs = np.abs(response_funcs[pnt_pos, ii] - peaks_val[ii])
            dists = pnt_pos - peaks_pos[ii]
            coeffs = diffs / np.abs(dists)
            confidence[ii] = np.sqrt(np.sum(coeffs ** 2) / len(coeffs))
    else:
        raise ValueError('Unknown confidence method: %s' % confidence_method)

    confidence = np.reshape(confidence, cue_map_size)
    depth = np.reshape(peaks_pos, cue_map_size)
    if med_filt_size is not None:
        depth = sps.medfilt(depth, med_filt_size)

    return (depth, confidence)


def compute_depth_cues(
        lf: lightfield.Lightfield, zs, compute_defocus=True,
        compute_correspondence=True, compute_emergence=False,
        beam_geometry='parallel', domain='object', psf=None,
        up_sampling=1, super_sampling=1, algorithm='bpj', iterations=5,
        confidence_method='integral',
        window_size=(9, 9), window_shape='gauss', mask=None, plot_filter=False):
    """Computes depth cues, needed to create a depth map.

    These depth cues are created following the procedure from:
    [1] M. W. Tao, et al., “Depth from combining defocus and correspondence using light-field cameras,”
    in Proceedings of the IEEE International Conference on Computer Vision, 2013, pp. 673–680.

    :param lf: The light-field object (lightfield.Lightfield)
    :param zs: Refocusing distances (numpy.array_like)
    :param compute_defocus: Switch for defocus cues (Boolean, default: True)
    :param compute_correspondence: Switch for corresponence cues (Boolean, default: True)
    :param window_size: Filtering window size (tuple, default: (9, 9))
    :param window_shape: Filtering window shape. Options: 'tri' | 'circ' | 'gauss' | 'rect' (string, default: 'circ')
    :param up_sampling: Integer greater than 1 for up-sampling of the final images (int, default: 1)
    :param border: Number of pixels to extend the border and reduce darkening of edges (int, default: 4)
    :param border_padding: Border padding method (string, default: 'edge')
    :param beam_geometry: Beam geometry. Possible options: 'parallel' | 'cone' (string, default: 'parallel')
    :param domain: Refocusing domain. Possible options: 'object' | 'image' (string, default: 'object')

    :returns: Depth cues and their confidences.
    :rtype: dict
    """

    lf_sa = lf.clone()
    lf_sa.set_mode_subaperture()

    num_zs = zs.size

    window_size = np.array(window_size) * up_sampling

    window_filter = proc.get_smoothing_filter(
            window_size=window_size, window_shape=window_shape,
            plot_filter=plot_filter)

    depth_cues = {
        'defocus': np.array(()),
        'correspondence': np.array(()),
        'emergence': np.array(()),
        'depth_defocus': np.array(()),
        'depth_correspondence': np.array(()),
        'depth_emergence': np.array(()),
        'confidence_defocus': np.array(()),
        'confidence_correspondence': np.array(()),
        'confidence_emergence': np.array(())
        }

    data_type = lf.data.dtype
    final_image_shape = lf_sa.camera.data_size_ts * up_sampling
    responses_shape = np.concatenate(((num_zs, ), final_image_shape))
    if compute_defocus:
        depth_cues['defocus'] = np.empty(responses_shape, dtype=data_type)
    if compute_correspondence:
        depth_cues['correspondence'] = np.empty(responses_shape, dtype=data_type)
    if compute_emergence:
        depth_cues['emergence'] = np.empty(responses_shape, dtype=data_type)

    paddings_ts = ((np.fmax(window_size, window_size) - 1) / 2 + 5).astype(np.intp)
    final_size_ts = lf_sa.camera.data_size_ts + 2 * paddings_ts
    additional_padding_ts = np.floor(((4 - (final_size_ts % 4)) % 4) / 2).astype(np.intp)
    paddings_ts += additional_padding_ts
    lf_sa.pad([0, 0, paddings_ts[0], paddings_ts[1]], method='edge')
    paddings_ts *= up_sampling

    if mask is not None:
        mask = np.pad(mask, ((paddings_ts[0], ), (paddings_ts[1], )), mode='edge')
        mask_renorm = spimg.convolve(mask, window_filter, mode='constant', cval=0.0)
        mask_renorm[mask_renorm == 0] = 1
    else:
        mask_renorm = 1

    if algorithm.lower() == 'sirt':
        algo = solvers.Sirt()
    elif algorithm.lower() == 'cp_ls':
        algo = solvers.CP_uc()
    elif algorithm.lower() == 'cp_tv':
        algo = solvers.CP_tv(axes=(-2, -1), lambda_tv=0.1)
    elif algorithm.lower() == 'cp_wl':
        algo = solvers.CP_wl(axes=(-2, -1), wl_type='sym4', decomp_lvl=2, lambda_wl=1e-1)
    elif algorithm.lower() == 'bpj':
        algo = solvers.BPJ()
    else:
        raise ValueError('Unrecognized algorithm: %s' % algorithm.lower())

    b = lf_sa.data[np.newaxis, ...]
    if compute_emergence:
        highpass_filter = proc.get_highpass_filter(b.shape, 8, 1)
        b_hp = proc.apply_bandpass_filter(b, highpass_filter)

    print('Computing responses for each alpha value: ', end='', flush=True)
    c = tm.time()

    for ii_a, z0 in enumerate(zs):
        prnt_str = '%03d/%03d' % (ii_a, num_zs)
        print(prnt_str, end='', flush=True)

        with Projector(
                lf_sa.camera, np.array((z0, )), mask=lf_sa.mask, mode='independent',
                up_sampling=up_sampling, beam_geometry=beam_geometry,
                domain=domain, super_sampling=super_sampling, psf_d=psf) as p:

            l_alpha_intuv = algo(p.FP, b, num_iter=iterations, At=p.BP, lower_limit=0)[0]

            if compute_defocus:
                l_alpha_grad = _gradient2(np.squeeze(l_alpha_intuv))
                l_alpha_grad = np.sqrt(np.abs(l_alpha_grad[0]) ** 2 + np.abs(l_alpha_grad[1]) ** 2)

                depth_defocus = _apply_smoothing_filter(l_alpha_grad, window_filter, mask, mask_renorm)
                depth_defocus = np.fmax(depth_defocus, 1e-5)

                depth_cues['defocus'][ii_a, :, :] = depth_defocus[
                    paddings_ts[0]:-paddings_ts[0], paddings_ts[1]:-paddings_ts[1]]

            if compute_emergence:
                # Needs a high pass filter to the data!
                l_alpha_intuv_hp = algo(p.FP, b_hp, num_iter=iterations, At=p.BP, lower_limit=0)[0]
                l_alpha_intuv_hp = np.abs(l_alpha_intuv_hp)

                depth_emergence = _apply_smoothing_filter(l_alpha_intuv_hp, window_filter, mask, mask_renorm)
                depth_emergence = np.fmax(depth_emergence, 1e-5)

                depth_cues['emergence'][ii_a, :, :] = depth_emergence[
                    paddings_ts[0]:-paddings_ts[0], paddings_ts[1]:-paddings_ts[1]]

            if compute_correspondence:
                reprojected_l_alpha_intuv = p.FP(l_alpha_intuv)
                variances = np.abs(reprojected_l_alpha_intuv - b) ** 2

                with Projector(
                        lf_sa.camera, np.array((z0, )), mask=lf_sa.mask, mode='independent',
                        up_sampling=up_sampling, beam_geometry=beam_geometry,
                        domain=domain, super_sampling=super_sampling) as p:
                    bpj_variances = solvers.BPJ()(p.FP, variances, At=p.BP, lower_limit=0)[0]

                std_devs = np.sqrt(bpj_variances)
                inv_std_devs = 1 / np.fmax(std_devs, 1e-5)
                depth_correspondence = _apply_smoothing_filter(inv_std_devs, window_filter, mask, mask_renorm)
                depth_correspondence = np.fmax(depth_correspondence, 1e-5)

                depth_cues['correspondence'][ii_a, :, :] = depth_correspondence[
                    paddings_ts[0]:-paddings_ts[0], paddings_ts[1]:-paddings_ts[1]]

        print(('\b') * len(prnt_str), end='', flush=True)

    print('Done (%d) in %g seconds.' % (num_zs, tm.time() - c))

    if compute_defocus:
        print('Computing depth estimations for defocus:\n - Preparing response..', end='', flush=True)
        c = tm.time()
        depth_cues['depth_defocus'], depth_cues['confidence_defocus'] = _compute_depth_and_confidence(
            depth_cues['defocus'], confidence_method=confidence_method)
        print('\b\b: Done (%d) in %g seconds.' % (depth_cues['depth_defocus'].size, tm.time() - c))

    if compute_emergence:
        print('Computing depth estimations for emergence:\n - Preparing response..', end='', flush=True)
        c = tm.time()
        depth_cues['depth_emergence'], depth_cues['confidence_emergence'] = _compute_depth_and_confidence(
            depth_cues['emergence'], confidence_method=confidence_method)
        print('\b\b: Done (%d) in %g seconds.' % (depth_cues['depth_emergence'].size, tm.time() - c))

    if compute_correspondence:
        print('Computing depth estimations for correspondence:\n - Preparing response..', end='', flush=True)
        c = tm.time()
        depth_cues['depth_correspondence'], depth_cues['confidence_correspondence'] = _compute_depth_and_confidence(
            depth_cues['correspondence'], confidence_method=confidence_method)
        print('\b\b: Done (%d) in %g seconds.' % (depth_cues['depth_correspondence'].size, tm.time() - c))

    return depth_cues


def compute_depth_map(
        depth_cues, iterations=500, lambda_tv=2.0, lambda_d2=0.05,
        lambda_wl=None, use_defocus=1.0, use_correspondence=1.0, use_emergence=0.0):
    """Computes a depth map from the given depth cues.

    This depth map is based on the procedure from:
    [1] M. W. Tao, et al., “Depth from combining defocus and correspondence using light-field cameras,”
    in Proceedings of the IEEE International Conference on Computer Vision, 2013, pp. 673–680.

    :param depth_cues: The depth cues
    :type depth_cues: dict
    :param iterations: Number of iterations, defaults to 500
    :type iterations: int, optional
    :param lambda_tv: Lambda value of the TV term, defaults to 2.0
    :type lambda_tv: float, optional
    :param lambda_d2: Lambda value of the smoothing term, defaults to 0.05
    :type lambda_d2: float, optional
    :param lambda_wl: Lambda value of the wavelet term, defaults to None
    :type lambda_wl: float, optional
    :param use_defocus: Weight of defocus cues, defaults to 1.0
    :type use_defocus: float, optional
    :param use_correspondence: Weight of corresponence cues, defaults to 1.0
    :type use_correspondence: float, optional
    :param use_emergence: Weight of the emergence cues, defaults to 0.0
    :type use_emergence: float, optional

    :returns: The depth map
    :rtype: `numpy.array_like`
    """

    use_defocus = np.fmax(use_defocus, 0.0)
    use_defocus = np.fmin(use_defocus, 1.0)
    use_correspondence = np.fmax(use_correspondence, 0.0)
    use_correspondence = np.fmin(use_correspondence, 1.0)
    use_emergence = np.fmax(use_emergence, 0.0)
    use_emergence = np.fmin(use_emergence, 1.0)

    W_d = depth_cues['confidence_defocus']
    a_d = depth_cues['depth_defocus']

    W_c = depth_cues['confidence_correspondence']
    a_c = depth_cues['depth_correspondence']

    W_e = depth_cues['confidence_emergence']
    a_e = depth_cues['depth_emergence']

    if use_defocus > 0 and (W_d.size == 0 or a_d.size == 0):
        use_defocus = 0
        warnings.warn('Defocusing parameters were not passed, disabling their use')

    if use_correspondence > 0 and (W_c.size == 0 or a_c.size == 0):
        use_correspondence = 0
        warnings.warn('Correspondence parameters were not passed, disabling their use')

    if use_emergence > 0 and (W_e.size == 0 or a_e.size == 0):
        use_emergence = 0
        warnings.warn('Emergence parameters were not passed, disabling their use')

    if use_defocus:
        img_size = a_d.shape
        data_type = a_d.dtype
    elif use_correspondence:
        img_size = a_c.shape
        data_type = a_c.dtype
    elif use_emergence:
        img_size = a_e.shape
        data_type = a_e.dtype
    else:
        raise ValueError('Cannot proceed if at least one of Defocus, Correspondence, and Emergence cues can be used')

    if lambda_wl is not None and has_pywt is False:
        lambda_wl = None
        print('WARNING - wavelets selected but not available')

    depth = np.zeros(img_size, dtype=data_type)
    depth_it = depth

    q_g = np.zeros(np.concatenate(((2, ), img_size)), dtype=data_type)
    tau = 4 * lambda_tv
    if lambda_d2 is not None:
        q_l = np.zeros(img_size, dtype=data_type)
        tau += 8 * lambda_d2
    if use_defocus > 0:
        q_d = np.zeros(img_size, dtype=data_type)
        tau += W_d
    if use_correspondence > 0:
        q_c = np.zeros(img_size, dtype=data_type)
        tau += W_c
    if use_emergence > 0:
        q_e = np.zeros(img_size, dtype=data_type)
        tau += W_e
    if lambda_wl is not None:
        wl_type = 'sym4'
        wl_lvl = np.fmin(pywt.dwtn_max_level(img_size, wl_type), 2)
        print('Wavelets selected! Wl type: %s, Wl lvl %d' % (wl_type, wl_lvl))
        q_wl = pywt.swtn(depth, wl_type, wl_lvl)
        tau += lambda_wl * (2 ** wl_lvl)
        sigma_wl = 1 / (2 ** np.arange(wl_lvl, 0, -1))
    tau = 1 / tau

    for ii in range(iterations):
        (d0, d1) = _gradient2(depth_it)
        d_2 = np.stack((d0, d1)) / 2
        q_g += d_2
        grad_l2_norm = np.fmax(1, np.sqrt(np.sum(q_g ** 2, axis=0)))
        q_g /= grad_l2_norm

        update = - lambda_tv * _divergence2(q_g[0, :, :], q_g[1, :, :])
        if lambda_d2 is not None:
            l_dep = _laplacian2(depth_it)
            q_l += l_dep / 8
            q_l /= np.fmax(1, np.abs(q_l))

            update += lambda_d2 * _laplacian2(q_l)

        if use_defocus > 0:
            q_d += (depth_it - a_d)
            q_d /= np.fmax(1, np.abs(q_d))

            update += use_defocus * W_d * q_d

        if use_correspondence > 0:
            q_c += (depth_it - a_c)
            q_c /= np.fmax(1, np.abs(q_c))

            update += use_correspondence * W_c * q_c

        if use_emergence > 0:
            q_e += (depth_it - a_c)
            q_e /= np.fmax(1, np.abs(q_e))

            update += use_emergence * W_e * q_e

        if lambda_wl is not None:
            d = pywt.swtn(depth_it, wl_type, wl_lvl)
            for ii_l in range(wl_lvl):
                for k in q_wl[ii_l].keys():
                    q_wl[ii_l][k] += d[ii_l][k] * sigma_wl[ii_l]
                    q_wl[ii_l][k] /= np.fmax(1, np.abs(q_wl[ii_l][k]))
            update += lambda_wl * pywt.iswtn(q_wl, wl_type)

        depth_new = depth - update * tau
        depth_it = depth_new + (depth_new - depth)
        depth = depth_new

    return depth


def get_distances(dm, zs):
    """Convert depth map indices into distances

    :param dm: the depth map (numpy.array_like)
    :param zs: the corresponding distances in the depth map (numpy.array_like)

    :returns: the depth-map containing the real distances
    :rtype: numpy.array_like
    """
    dm = np.fmin(dm, len(zs)-1)
    dm = np.fmax(dm, 0)
    interp_dists = sp.interpolate.interp1d(np.arange(zs.size), zs)
    return np.reshape(interp_dists(dm.flatten()), dm.shape)


def _laplacian2(x):
    l0 = np.pad(x, ((1, 1), (0, 0)), mode='edge')
    l1 = np.pad(x, ((0, 0), (1, 1)), mode='edge')
    return np.diff(l0, n=2, axis=0) + np.diff(l1, n=2, axis=1)


def _gradient2(x):
    d0 = np.pad(x, ((0, 1), (0, 0)), mode='constant')
    d0 = np.diff(d0, n=1, axis=0)
    d1 = np.pad(x, ((0, 0), (0, 1)), mode='constant')
    d1 = np.diff(d1, n=1, axis=1)
    return (d0, d1)


def _divergence2(x0, x1):
    d0 = np.pad(x0, ((1, 0), (0, 0)), mode='constant')
    d1 = np.pad(x1, ((0, 0), (1, 0)), mode='constant')
    return np.diff(d0, n=1, axis=0) + np.diff(d1, n=1, axis=1)
