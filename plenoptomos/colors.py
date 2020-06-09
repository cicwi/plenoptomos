#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convenience functions for handling multi-channel datasets.
They mainly focus on visible light (RGB channels).

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France

Created on Tue Apr  4 16:01:41 2017
"""

import numpy as np


def merge_rgb_images(imgs_r, imgs_g, imgs_b, vmin=0.0, vmax=1.0):
    imgs_t = np.array((imgs_r, imgs_g, imgs_b))
    imgs_t = np.fmin(imgs_t, vmax)
    imgs_t = np.fmax(imgs_t, vmin)
    return np.transpose(imgs_t, (1, 2, 3, 0))


def from_rgb_to_grayscale(img, mode='luma'):
    if mode.lower() == 'luma':
        coeffs = np.array((0.3, 0.59, 0.11))
    elif mode.lower() == 'flat':
        coeffs = 1.0/3.0
    else:
        raise ValueError("unrecognized RGB -> GrayScale conversion mode: %s" % mode)

    return np.sum(img * coeffs, axis=-1)


def deal_with_channels(img, mode='grayscale', rgb2gs_mode='luma'):
    num_channels = img.shape[-1]

    if num_channels == 1:
        return (img, np.array([]))
    elif num_channels == 2:
        return (img[..., 0], img[..., 1] > 0)
    elif num_channels == 3:
        if mode.lower() == 'grayscale':
            return (from_rgb_to_grayscale(img, mode=rgb2gs_mode), np.array([]))
        else:
            return ((img[..., 0], img[..., 1], img[..., 2]), np.array([]))
    elif num_channels == 4:
        if mode.lower() == 'grayscale':
            return (from_rgb_to_grayscale(img[..., 0:3], mode=rgb2gs_mode), img[..., 3] > 0)
        else:
            return ((img[..., 0], img[..., 1], img[..., 2]), img[..., 3] > 0)
    else:
        raise ValueError('Incoming image should either be mono-chromatic or an RGB image, with optional alpha-channel')


def get_rgb_wavelengths():
    return (np.array((0.62, 0.74)), np.array((0.495, 0.570)), np.array((0.45, 0.495)))


def merge_rgb_lightfields(lf_r, lf_g, lf_b, mode='luma'):
    lf = lf_r.clone()
    lf.camera.wavelength_range = [0.45, 0.74]

    lf_g.set_mode(lf.mode)
    lf_b.set_mode(lf.mode)

    lf.data = np.stack((lf_r.data, lf_g.data, lf_b.data))
    lf.data = from_rgb_to_grayscale(np.transpose(lf.data, (1, 2, 3, 4, 0)), mode=mode)

    return lf


def detect_color(camera, tol=1e-2):
    def get_overlap(c):
        color_range = c[1] - c[0]
        lambda_range = camera.wavelength_range[1] - camera.wavelength_range[0]
        side_1_diff = c[1] - camera.wavelength_range[0]
        side_2_diff = camera.wavelength_range[1] - c[0]
        ranges = np.array((side_1_diff, side_2_diff, color_range, lambda_range))
        return np.min(ranges) / color_range

    (red, green, blue) = get_rgb_wavelengths()
    if len(camera.wavelength_range) == 1:
        if red[0] <= camera.wavelength_range <= red[1]:
            return 'red'
        elif green[0] <= camera.wavelength_range <= green[1]:
            return 'green'
        elif blue[0] <= camera.wavelength_range <= blue[1]:
            return 'blue'
        else:
            return 'unknown'
    elif len(camera.wavelength_range) == 2:
        overlap_r = get_overlap(red)
        overlap_g = get_overlap(green)
        overlap_b = get_overlap(blue)

        if np.all(np.array((overlap_r, overlap_g, overlap_b)) > 0):
            return 'white'
        elif overlap_r > tol and overlap_g < tol and overlap_b < tol:
            return 'red'
        elif overlap_r < tol and overlap_g > tol and overlap_b < tol:
            return 'green'
        elif overlap_r < tol and overlap_g < tol and overlap_b > tol:
            return 'blue'
        else:
            return 'unknown'
    else:
        return 'unknown'


def convert_energy_to_wavelength(energy, unit_energy='eV', unit_lambda='um'):
    # Here we think in um
    if unit_lambda.lower() == 'mm':
        l_scale = 1e+3
    elif unit_lambda.lower() == 'um':
        l_scale = 1
    elif unit_lambda.lower() == 'nm':
        l_scale = 1e-3
    elif unit_lambda.lower() == 'pm':
        l_scale = 1e-6
    else:
        raise ValueError("Unknown wavelength unit: %s" % unit_lambda)
    # Here we think in eV
    if unit_energy.lower() == 'ev':
        e_scale = 1
    elif unit_energy.lower() == 'kev':
        e_scale = 1e+3
    elif unit_energy.lower() == 'mev':
        e_scale = 1e+6
    elif unit_energy.lower() == 'gev':
        e_scale = 1e+9
    else:
        raise ValueError("Unknown wavelength unit: %s" % unit_energy)

    return 1.2398 / (energy * e_scale * l_scale)
