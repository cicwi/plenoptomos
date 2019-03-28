#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 18:31:14 2017

@author: vigano

This module implements the point spread function (PSF) handling routines.
"""

import copy

import numpy as np
import numpy.fft

import scipy.ndimage as spimg
import scipy.special as spspecial
import scipy.signal as spsig
from scipy import fftpack

import matplotlib.pyplot as plt
# Do not remove the following import: it is used somehow by the plotting
# functionality in the PSF creation
from mpl_toolkits.mplot3d import Axes3D # noqa: F401 unused import

from collections import namedtuple

from . import solvers

import time as tm

class PSF(object):
    """Data class that allows to store n-dimensional PSFs, and their
    accompanying information.
    """

    def __init__(self, coordinates, data=None, conf=None, data_format=None):
        self.coordinates = (coordinates[0], coordinates[1])
        self.data = data
        self.conf = conf
        self.data_format = data_format

    def clone(self):
        return copy.deepcopy(self)

    @staticmethod
    def create_theo_psf(camera, coordinates, wavelength_steps=10, \
                        wavelength_intensity=1, airy_rings=2, \
                        refocus_distance=None, up_sampling=1, \
                        beam_coherence='incoherent', \
                        over_sampling=25, data_type=np.float32, plot=False):
        """Compute the theoretical PSF for the given coordinates in the given
        camera setup.

        :param camera: The camera setup object (lightfield.Camera)
        :param coordinates: The coordinates where to compute the PSF. Options: 'vu' | 'ts' (string)
        :param airy_rings: Orders of the Airy function to consider (int, default:2)
        :param beam_coherence: coherence of the light source. Options: 'incoherent' | coherent (string, default: 'incoherent')
        :param refocus_distance: Distance of refocusing, for fine tuning the disk of confusion (float, default: None)

        :returns: The computed PSF.
        :rtype: PSF
        """
        # We think in mm
        if camera.wavelength_unit.lower() == 'mm':
            ls_unit = 1
        elif camera.wavelength_unit.lower() == 'um':
            ls_unit = 1e-3
        elif camera.wavelength_unit.lower() == 'nm':
            ls_unit = 1e-6
        elif camera.wavelength_unit.lower() == 'pm':
            ls_unit = 1e-9
        else:
            raise ValueError("Unknown wavelength unit: %s" % camera.wavelength_unit)

        if len(camera.wavelength_range) == 2:
            ls = np.linspace(camera.wavelength_range[0], camera.wavelength_range[1], wavelength_steps) * ls_unit
        else:
            ls = camera.wavelength_range * ls_unit

        z0 = camera.get_focused_distance()
        if coordinates.lower() in ('st', 'ts') and refocus_distance is not None:
            blur_size = np.abs(refocus_distance - z0) / z0 * camera.pixel_size_vu
            M = z0 / camera.z1
            defocus_size = np.mean(blur_size / (camera.pixel_size_ts * M)) / 2
#        elif coordinates.lower() in ('uv', 'vu') and camera.is_focused():
#            blur_size = np.abs(camera.b) / camera.f2
#            defocus_size = np.mean(blur_size) / 2
        else:
            defocus_size = 0

        Conf = namedtuple('Conf', ('airy_rings', 'over_sampling', \
                                   'beam_coherence',
                                   'wavelength_intensity', 'pixels_defocus', \
                                   'guarantee_pixel_multiple', 'data_type'))
        conf = Conf(airy_rings, over_sampling, beam_coherence, \
                    wavelength_intensity, defocus_size, 1, data_type)

        print("- Creating Theoretical PSFs for (%s, %s) coordinates (defocus size: %g).." \
              % (coordinates[0], coordinates[1], defocus_size), end='', flush=True)
        c_in = tm.time()

        if coordinates.lower() in ('uv', 'vu'):
            # Micro lenses PSF
            d2 = camera.f2 / camera.aperture_f2
            if camera.is_focused():
                psf_data = PSF.compute_fraunhofer_psf(conf, d2, camera.b, camera.pixel_size_yx, ls)
            else:
                psf_data = PSF.compute_fraunhofer_psf(conf, d2, camera.f2, camera.pixel_size_yx, ls)
            data_format = 'raw'
        elif coordinates.lower() in ('st', 'ts'):
            # Main lens PSF
            d1 = camera.f1 / camera.aperture_f1
            if up_sampling > 1:
                data_format = 'subpixel'
            else:
                data_format = None
            if camera.is_focused():
                psf_data = PSF.compute_fraunhofer_psf(conf, d1, camera.z1 + camera.a, camera.pixel_size_ts / up_sampling, ls)
            else:
                psf_data = PSF.compute_fraunhofer_psf(conf, d1, camera.z1, camera.pixel_size_ts / up_sampling, ls)

        c_out = tm.time()
        print("\b\b: Done in %g seconds." % (c_out - c_in))

        psf = PSF(coordinates, data=psf_data, conf=conf, data_format=data_format)

        if plot:
            pixels_distance = (psf_data.shape[0] - 1) / 2
            grid_p = np.linspace(-pixels_distance, pixels_distance, 2 * pixels_distance + 1)
            [grid_p1, grid_p2] = np.meshgrid(grid_p, grid_p, indexing='ij')

            f = plt.figure()
            ax = f.add_subplot(1, 1, 1, projection='3d')
            ax.plot_surface(grid_p1, grid_p2, psf_data)
            ax.view_init(12, -7.5)
            plt.show()

        return psf

    @staticmethod
    def compute_fraunhofer_psf(conf, d, z, pixel_size, ls):
        #computing real pixel_distance of first zero
        disk_d = np.mean(z * np.arcsin(1.22 * ls / d)) / np.mean(pixel_size)
        pixels_distance = np.ceil(disk_d * conf.airy_rings).astype(np.int)
        base_block_size = 2 * pixels_distance + 1
        sampled_pixels = base_block_size * conf.over_sampling

        samp_1 = np.linspace(-pixels_distance, pixels_distance, sampled_pixels)
        samp_2 = np.linspace(-pixels_distance, pixels_distance, sampled_pixels)

        samp_1 = samp_1 * pixel_size[0]
        samp_2 = samp_2 * pixel_size[1]

        [grid_1, grid_2] = np.meshgrid(samp_1, samp_2, indexing='ij')

        data_center = pixels_distance * conf.over_sampling + np.floor(conf.over_sampling / 2)
        data_center = data_center.astype(np.int32)

        # Airy function
        r = np.sqrt(grid_1 ** 2 + grid_2 ** 2)
        r = np.reshape(r, np.concatenate(((1, ), r.shape)))
        ls = np.reshape(ls, (-1, 1, 1))
        h = np.pi * d * r / (ls * z)
        J10 = spspecial.jv(1, h)
        h[:, data_center, data_center] = 1 # avoid warning abou NaN
        h = 2 * J10 / h
        # Setting central pixel to 1 (otherwise it would be NaN or 0)
        h[:, data_center, data_center] = 1

        if conf.beam_coherence.lower() == 'coherent':
            int_exp = 2
        else:
            int_exp = 4
        # The abs is not really needed, since the Airy function is a real
        # valued function
        h = h ** int_exp

        # Summing the contribution from all the wavelengths, and renormalizing
        h *= np.reshape(conf.wavelength_intensity, (-1, 1, 1))
        h = np.sum(h, axis=0)
        h /= np.sum(h)

        if conf.pixels_defocus > 0:
            hd = PSF.compute_defocus_psf(conf, keep_oversampling=True)
            h = spsig.convolve2d(h, hd, 'same')

        # Producing the final impulse response h, at the given resolution
        h = np.reshape(h, (base_block_size, conf.over_sampling, base_block_size, conf.over_sampling))
        h = np.sum(h, axis=(1, 3))
        h = h.astype(conf.data_type)

        return h

    @staticmethod
    def compute_defocus_psf(conf, norm=2, keep_oversampling=False):
        sampling_distance = np.ceil(conf.pixels_defocus).astype(np.intp)
        h_size = 2 * sampling_distance + 1
        pixel_target = conf.guarantee_pixel_multiple
        h_size = h_size + (pixel_target - (h_size % pixel_target)) % pixel_target
        h_size = (h_size + pixel_target * ((h_size / pixel_target - 1) % 2)).astype(np.intp)

        sampling_distance = (h_size - 1) / 2
        sampled_pixels = h_size * conf.over_sampling

        samp_1 = np.linspace(-sampling_distance, sampling_distance, sampled_pixels)
        samp_2 = np.linspace(-sampling_distance, sampling_distance, sampled_pixels)

        [grid_1, grid_2] = np.meshgrid(samp_1, samp_2, indexing='ij')

        if (isinstance(norm, str) and norm.lower() == 'inf') or norm == np.inf:
            h = (np.fmax(np.abs(grid_1), np.abs(grid_2)) <= conf.pixels_defocus).astype(conf.data_type)
        else:
            h = ((grid_1 ** norm + grid_2 ** norm) <= (conf.pixels_defocus ** norm)).astype(conf.data_type)

        # Summing the contribution from all the wavelengths, and renormalizing
        h /= np.sum(h)

        if keep_oversampling is False:
            # Producing the final impulse response h, at the given resolution
            h = np.reshape(h, (h_size, conf.over_sampling, h_size, conf.over_sampling))
            h = np.sum(h, axis=(1, 3))
            h = h.astype(conf.data_type)

        return h

class PSFApply(object):
    """Class PSFApply handles all PSF/OTF applications
    """

    def __init__(self, psf_d=None, img_size=None, use_otf=True, data_format=None, use_fftconv=True):
        print("- Initializing PSF application class..", end='', flush=True)
        c_in = tm.time()

        self._reset()

        self.data_format = data_format
        self.use_otf = use_otf
        self.use_fftconv = use_fftconv

        if psf_d is not None:
            self.set_psf_direct(psf_d, img_size=img_size)

        c_out = tm.time()
        print("\b\b: Done in %g seconds." % (c_out - c_in))

    def print(self):
        print("PSF object's properties:")
        for k, v in self.__dict__.items():
            if v is not None and k.lower() in ('psf_direct', 'psf_adjoint', 'otf_direct', 'otf_adjoint'):
                print(' %18s : %s, %s' % (k, str(v.shape), str(v.dtype)))
            else:
                print(' %18s : %s' % (k, str(v)))

    def _reset(self):
        self.data_type = np.float32

        self.psf_direct = None
        self.psf_adjoint = None
        self.otf_direct = None
        self.otf_adjoint = None

        self.is_symmetric = True
        self.is_projection_invariant = True
        self.image_size = None

        self.use_otf = True
        self.use_fftconv = False
        self.psf_edge = np.zeros((0, ), dtype=self.data_type)
        self.extra_pad = np.zeros((0, ), dtype=self.data_type)
        self.data_format = None

    def set_psf_direct(self, psf_d, img_size=None):
        if self.psf_direct is not None:
            self._reset()

        # let's be sure that it will be in ASTRA's projection convention
        self._check_incoming_psf(psf_d)

        self.is_projection_invariant = len(psf_d.shape) == len(self.otf_axes)

        # let's renormalize the PSFs
        psf_norms = np.sum(np.abs(psf_d), axis=self.otf_axes)
        if self.is_projection_invariant:
            psf_d = psf_d / psf_norms
        else:
            repl_ones = (1, ) * len(self.otf_axes)
            psf_d = psf_d / np.reshape(psf_norms, np.concatenate(((-1, ), repl_ones)))

        psf_edge = (np.array(psf_d.shape[-len(self.otf_axes):]) - 1) / 2
        self.psf_edge = psf_edge.astype(np.intp)

        self.psf_direct = psf_d.astype(self.data_type)

        # Let's find out whether they are symmetric or not
        psf_t = psf_d
        for otf_axis in self.otf_axes:
            psf_t = np.flip(psf_t, axis=otf_axis)

        self.is_symmetric = np.all(np.abs(psf_d.flatten() - psf_t.flatten()) < np.finfo(np.float32).eps)
        if self.is_symmetric is not True:
            self.psf_adjoint = psf_t

        if img_size is not None:
            self.image_size = np.array(img_size)
        # if we know the images sizes, we can already compute the OTF
        if self.use_otf is True and self.image_size is not None:
            self._init_otfs()

    def apply_psf_direct(self, imgs):
        """Applies the PSF to the given images

        :param imgs: The incoming images (numpy.array_like)

        :returns: The convolution between the images and the PSF
        :rtype: numpy.array_like
        """
        self._check_incoming_images(imgs)

        if self.use_otf is True:
            return self._apply_otf(imgs, True)
        else:
            return self._apply_psf(imgs, True)

    def apply_psf_adjoint(self, imgs):
        """Applies the adjoint of the PSF to the given images

        :param imgs: The incoming images (numpy.array_like)

        :returns: The convolution between the images and the adjoint of the PSF
        :rtype: numpy.array_like
        """
        self._check_incoming_images(imgs);

        if self.use_otf is True:
            return self._apply_otf(imgs, False)
        else:
            return self._apply_psf(imgs, False)

    def deconvolve(self, imgs, iterations=100, data_term='l2', lambda_wl=None, lower_limit=None, upper_limit=None, verbose=False):
        """Uses iterative algorithms to deconvolve the PSF from the given images

        :param imgs: The incoming images (numpy.array_like)
        :param iterations: The number of reconstruciton iterations (int, default: 100)
        :param lambda_wl: Weight factor for the wavelet deconvolution. If None is passed, the SIRT algorithm will be used instead (float, default: None)
        :param data_term: Data consistency term used by the wl recosntruction. Options: 'l2' | 'kl' (string, default: 'l2')
        :param lower_limit: Lower clipping value (float, default: None)
        :param upper_limit: Upper clipping value (float, default: None)

        :returns: The deconvolution of the images
        :rtype: numpy.array_like
        """
        if lambda_wl is not None:
            sol = solvers.CP_wl(data_term=data_term, lambda_wl=lambda_wl, wl_type='db1', verbose=verbose)
        else:
            sol = solvers.Sirt(verbose=verbose)
        return sol(self.apply_psf_direct, imgs, iterations, At=self.apply_psf_adjoint, lower_limit=lower_limit, upper_limit=upper_limit)

    def _get_psf_datashape(self):
        psf_shape = np.zeros_like(self.image_size)
        psf_shape[np.r_[self.otf_axes]] = self.psf_edge * 2
        return psf_shape + 1

    def _init_otfs(self):
        full_conv_shape = self.image_size + self._get_psf_datashape() - 1
        for ii in self.otf_axes:
            full_conv_shape[ii] = fftpack.helper.next_fast_len(full_conv_shape[ii])
        self.extra_pad = full_conv_shape - self.image_size

        self.otf_direct = self._init_single_otf(self.psf_direct)

        if self.is_symmetric is not True:
            self.otf_adjoint = self._init_single_otf(self.psf_adjoint)

    def _init_single_otf(self, psf):
        imgs_shape = self.image_size + self.extra_pad
        psf = np.reshape(psf, self._get_psf_datashape())
        fft_shape = imgs_shape[np.r_[self.otf_axes]]
        return np.fft.rfftn(psf, fft_shape, axes=self.otf_axes)

    def _check_incoming_psf(self, psf_d):
        raise NotImplementedError()

    def _check_incoming_images(self, img):
        img_size = np.array(img.shape)

        if self.image_size is None:
            self.otf_direct = None
            self.otf_adjoint = None
            self.image_size = img_size
        elif not np.all(self.image_size == img_size):
            print("WARNING: OTFs computed for the wrong image size ([%s] instead of [%s]). Recomputing them..." \
                  % (", ".join((str(x) for x in self.image_size)), ", ".join((str(x) for x in img_size))))
            self.otf_direct = None
            self.otf_adjoint = None
            self.image_size = img_size

        if self.use_otf is True and self.otf_direct is None:
            self._init_otfs()

    def _apply_otf(self, imgs, is_direct):
        imgs_shape = self.image_size + self.extra_pad
        fft_shape = imgs_shape[np.r_[self.otf_axes]]
        imgs = np.fft.rfftn(imgs, fft_shape, axes=self.otf_axes)

        if is_direct or self.is_symmetric:
            imgs *= self.otf_direct
        else:
            imgs *= self.otf_adjoint

        imgs = np.fft.irfftn(imgs, fft_shape, axes=self.otf_axes)
        imgs = np.real(imgs)

        # slicing images to remove padding used during convolution
        psf_edge_shape = ((self._get_psf_datashape() - 1) / 2).astype(np.int)
        fslice = [slice(None), ] * len(self.image_size)
        for ii in self.otf_axes:
            fslice[ii] = slice(psf_edge_shape[ii], self.image_size[ii] + psf_edge_shape[ii])
        return imgs[tuple(fslice)]

    def _apply_psf(self, imgs, is_direct):
        if is_direct or self.is_symmetric:
            psf = self.psf_direct
        else:
            psf = self.psf_adjoint

        psf_dims = len(psf.shape)
        num_dims_imgs = len(imgs.shape) - psf_dims

        pre_ones = (1, ) * num_dims_imgs
        psf = np.reshape(psf, np.concatenate((pre_ones, psf.shape)).astype(np.intp))
        if self.use_fftconv:
            return spsig.fftconvolve(imgs, psf, mode='same')
        else:
            return spimg.convolve(imgs, psf, mode='reflect')


class PSFApply2D(PSFApply):
    """Class PSFApply2D handles all PSF applications and
    """

    def __init__(self, psf_d, img_size=None, use_otf=True, data_format=None, use_fftconv=True):
        self.otf_axes = (-2, -1)
        if isinstance(psf_d, PSF):
            psf_inst = psf_d.clone()
            psf = np.squeeze(psf_inst.data)
            if data_format is None:
                data_format = psf_inst.data_format
        else:
            psf = np.squeeze(psf_d)
        PSFApply.__init__(self, psf, img_size=img_size, use_otf=use_otf, data_format=data_format, use_fftconv=use_fftconv)

    def _check_incoming_psf(self, psf_d):
        if len(psf_d) == 0 or not len(psf_d.shape) in (2, 3):
            raise ValueError('PSFs should be in the form of 2D images  with dimensions [0] == [1], with odd edges')
        elif not psf_d.shape[-2] == psf_d.shape[-1]:
            raise ValueError('PSFs should be in the form of 2D images  with _dimensions [0] == [1]_, with odd edges')
        elif np.mod(psf_d.shape[-2], 2) == 0 or np.mod(psf_d.shape[-1], 2) == 0:
            raise ValueError('PSFs should be in the form of 2D images  with dimensions [0] == [1], with _odd edges_')


class PSFApply4D(PSFApply):
    """Class PSFApply4D handles all PSF applications and
    """

    def __init__(self, psf_d, img_size=None, use_otf=True, data_format=None, use_fftconv=True):
        self.otf_axes = (-4, -3, -2, -1)
        if isinstance(psf_d, PSF):
            psf = psf_d.data
            if data_format is None:
                data_format = psf_d.data_format
        else:
            psf = np.squeeze(psf_d)
        PSFApply.__init__(self, psf, img_size=img_size, use_otf=use_otf, data_format=data_format, use_fftconv=use_fftconv)

    def _check_incoming_psf(self, psf_d):
        if len(psf_d) == 0 or not len(psf_d.shape) == 4:
            raise ValueError('PSFs should be in the form of 4D images  with dimensions [0] == [1] and [2] == [3], with odd edges')
        elif not psf_d.shape[0] == psf_d.shape[1] or not psf_d.shape[2] == psf_d.shape[3]:
            raise ValueError('PSFs should be in the form of 4D images  with _dimensions [0] == [1] and [2] == [3]_, with odd edges')
        elif np.mod(psf_d.shape[0], 2) == 0 or np.mod(psf_d.shape[2], 2) == 0:
            raise ValueError('PSFs should be in the form of 4D images  with dimensions [0] == [1] and [2] == [3], with _odd edges_')
