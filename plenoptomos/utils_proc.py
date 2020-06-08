# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 17:00:50 2019

@author: VIGANO
"""

import numpy as np
import scipy.ndimage as spimg
import scipy.signal as sps

import matplotlib.pyplot as plt
# Do not remove the following import: it is used somehow by the plotting
# functionality in the PSF creation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def get_smoothing_filter(
        window_size=(9, 9), window_shape='gauss', plot_filter=False):
    """
    """
    window_size = np.array(window_size)

    if window_shape.lower() == 'tri':
        window_filter = sps.triang(window_size[0]) * np.reshape(sps.triang(window_size[1]), [1, -1])
    elif window_shape.lower() == 'circ':
        tt = np.linspace(-(window_size[0] - 1) / 2, (window_size[0] - 1) / 2, window_size[0])
        ss = np.linspace(-(window_size[1] - 1) / 2, (window_size[1] - 1) / 2, window_size[1])
        [tt, ss] = np.meshgrid(tt, ss, indexing='ij')
        window_filter = np.sqrt(tt ** 2 + ss ** 2) <= (window_size[1] - 1) / 2
    elif window_shape.lower() == 'gauss':
        tt = np.linspace(-(window_size[0] - 1) / 2, (window_size[0] - 1) / 2, window_size[0])
        ss = np.linspace(-(window_size[1] - 1) / 2, (window_size[1] - 1) / 2, window_size[1])
        [tt, ss] = np.meshgrid(tt, ss, indexing='ij')
        window_filter = np.exp(- ((2 * tt) ** 2 / window_size[0] + (2 * ss) ** 2 / window_size[1]))
    elif window_shape.lower() == 'rect':
        window_filter = np.ones(window_size)
    else:
        raise ValueError('Unknown filter: %s' % window_shape)

    if plot_filter:
        tt = np.linspace(-(window_size[0] - 1) / 2, (window_size[0] - 1) / 2, window_size[0])
        ss = np.linspace(-(window_size[1] - 1) / 2, (window_size[1] - 1) / 2, window_size[1])
        [tt, ss] = np.meshgrid(tt, ss, indexing='ij')

        f = plt.figure()
        ax = f.add_subplot(1, 1, 1, projection='3d')
        ax.plot_surface(tt, ss, window_filter)
        ax.view_init(12, -7.5)
        plt.show()

    return window_filter / np.sum(window_filter)


def remove_background(
        data, blur_size, blur_func='circ', axes=(0, 1), do_reverse=False, non_negative=False):
    """
    """
    blur_size = np.array(blur_size, dtype=np.int)
    if blur_size.size == 1:
        blur_size = np.array([blur_size] * 2, dtype=np.int)
    blur_win = get_smoothing_filter(window_size=blur_size, window_shape=blur_func)

    window_shape = np.ones(len(data.shape), dtype=np.int)
    window_shape[axes[0]], window_shape[axes[1]] = blur_size[0], blur_size[1]
    blur_win = np.reshape(blur_win, window_shape)

    data -= spimg.convolve(data, blur_win, mode='nearest')
    if do_reverse:
        data = -data
    if non_negative:
        data[data < 0] = 0
    return data
