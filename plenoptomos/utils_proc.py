# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 17:00:50 2019

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

import numpy as np
import scipy.ndimage as spimg
import scipy.signal as spsig
import scipy.special as spspe

import matplotlib.pyplot as plt
# Do not remove the following import: it is used somehow by the plotting
# functionality in the PSF creation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def get_smoothing_filter(
        window_size=(9, 9), window_shape='gauss', plot_filter=False):
    """Computes and returns a smoothing filter.

    :param window_size: Filter window size, defaults to (9, 9)
    :type window_size: tuple(int, int), optional
    :param window_shape: Filter type, defaults to 'gauss'
    :type window_shape: str, optional. Options: {'gauss'} | 'tri' | 'circ' | 'rect'.
    :param plot_filter: Whether to plot the filter or not, defaults to False
    :type plot_filter: boolean, optional

    :raises ValueError: In case of wrong filter name

    :return: The filter
    :rtype: `numpy.array_like`
    """
    window_size = np.array(window_size)

    if window_shape.lower() == 'tri':
        window_filter = spsig.triang(window_size[0]) * np.reshape(spsig.triang(window_size[1]), [1, -1])
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
    """Removes the background.

    :param data: Input data.
    :type data: `numpy.array_like`
    :param blur_size: Size of the blur filter
    :type blur_size: tuple(int, int)
    :param blur_func: Smoothing blur type, defaults to 'circ'
    :type blur_func: str, optional. Options are the ones of `get_smoothing_filter`
    :param axes: Axes where to remove the background, defaults to (0, 1)
    :type axes: tuple, optional
    :param do_reverse: Computes the opposite of the input data (minus), defaults to False
    :type do_reverse: boolean, optional
    :param non_negative: Truncates the values below zero, defaults to False
    :type non_negative: boolean, optional

    :return: Background removed data
    :rtype: `numpy.array_like`
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


def get_lowpass_filter(img_shape, cutoff_pix, trans_pix):
    """Computes a low pass filter with the erfc.

    :param img_shape: Shape of the image
    :type img_shape: tuple
    :param cutoff_pix: Position of the cutoff in k-vector, in the fft domain
    :type cutoff_pix: int or tuple of ints
    :param trans_pix: Size of the cutoff transition in k-vector, in the fft domain
    :type trans_pix: int

    :return: The computes filter
    :rtype: `numpy.array_like`
    """
    coords = [np.fft.fftfreq(s, 1/s) for s in img_shape]
    coords = np.meshgrid(*coords, indexing='ij')

    cutoff_pix = np.reshape(cutoff_pix, [-1] + [1] * len(img_shape))
    rescale_co_pix = cutoff_pix / cutoff_pix[0]
    r = np.sqrt(np.sum((coords / rescale_co_pix) ** 2, axis=0))
    return spspe.erfc((r - cutoff_pix[0]) / trans_pix) / 2


def get_highpass_filter(img_shape, cutoff_pix, trans_pix):
    """Computes a high pass filter with the erf.

    :param img_shape: Shape of the image
    :type img_shape: tuple
    :param cutoff_pix: Position of the cutoff in k-vector, in the fft domain
    :type cutoff_pix: int or tuple of ints
    :param trans_pix: Size of the cutoff transition in k-vector, in the fft domain
    :type trans_pix: int

    :return: The computes filter
    :rtype: `numpy.array_like`
    """
    return 1 - get_lowpass_filter(img_shape, cutoff_pix, trans_pix)


def get_bandpass_filter(img_shape, cutoff_pix, trans_pix):
    """Computes a band pass filter with the erf.

    :param img_shape: Shape of the image
    :type img_shape: tuple
    :param cutoff_pix: Position of the cutoffs in k-vector, in the fft domain
    :type cutoff_pix: tuple(int, int)
    :param trans_pix: Size of the cutoffs transition in k-vector, in the fft domain
    :type trans_pix: tuple(int, int)

    :return: The computes filter
    :rtype: `numpy.array_like`
    """
    return (
        get_lowpass_filter(img_shape, cutoff_pix[0], trans_pix[0])
        * get_highpass_filter(img_shape, cutoff_pix[0], trans_pix[0]))


def apply_bandpass_filter(img, filt_fft):
    """Applies a 2D band pass filter to an image or collection of images.

    :param img: Image to be filtered
    :type img: `numpy.array_like`
    :param filt_fft: 2D filter in Fourier domain
    :type filt_fft: `numpy.array_like`

    :return: The filtered image
    :rtype: `numpy.array_like`
    """
    img_f = np.fft.fft2(img)
    img_f *= filt_fft
    return np.real(np.fft.ifft2(img_f))
