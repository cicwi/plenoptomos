#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 16:08:17 2017

@author: vigano

This module implements two important data containers: the data container
'Lightfield' and the metadata container 'Camera'.
They are used throughout the entire package for interpreting and performing low
level manipulation of the raw light-field data.
"""

import copy
import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

def get_camera(model_name, down_sampling_st=1, down_sampling_uv=1):
    camera = Camera(model=model_name)
    if model_name.lower() == "lytro_illum":
        camera.data_size_vu = np.array((14, 14), dtype=np.intp)
        camera.pixel_size_ts = np.array((20e-3, 20e-3), dtype=np.float32)
        camera.pixel_size_yx = camera.pixel_size_ts / camera.data_size_vu
    elif model_name.lower() == "stanford_archive":
        camera.data_size_vu = np.array((17, 17), dtype=np.intp)
        camera.pixel_size_yx = np.array((5.7e-3, 5.7e-3), dtype=np.float32)
        camera.z1 = 66.0
    elif model_name.lower() == "flexray":
        camera.data_size_vu = (np.array((17, 17), dtype=np.intp) / down_sampling_uv).astype(np.intp)
        camera.data_size_ts = (np.array((1536, 1944), dtype=np.intp) / down_sampling_st).astype(np.intp)
        camera.pixel_size_ts = np.array((74.9e-3, 74.9e-3), dtype=np.float32) * down_sampling_st
        camera.pixel_size_yx = camera.pixel_size_ts / camera.data_size_vu
        camera.aperture_f1 = 1.0
        camera.aperture_f2 = 1.0
    elif model_name.lower() == "synthetic":
        camera.data_size_vu = (np.array((16, 16), dtype=np.intp) / down_sampling_uv).astype(np.intp)
        camera.data_size_ts = (np.array((256, 512), dtype=np.intp) / down_sampling_st).astype(np.intp)
        camera.pixel_size_ts = np.array((16e-3, 16e-3), dtype=np.float32) * down_sampling_st
        camera.pixel_size_yx = camera.pixel_size_ts / camera.data_size_vu
        camera.f2 = 0.05
        camera.f1 = 20.0
        camera.z1 = 25.0
        camera.pixel_size_vu = camera.z1 * camera.pixel_size_yx / camera.f2
        camera.aperture_f1 = 2.0
        camera.aperture_f2 = 2.0
    else:
        raise ValueError("Camera '%s' is not supported!" % model_name)
    return camera

class Camera(object):
    """Class that holds the metadata needed to interpret light-field data"""

    def __init__(self, model="unknown"):
        self.model = model
        self.pixel_size_yx = np.zeros(2, dtype=np.float32)
        self.pixel_size_ts = np.zeros(2, dtype=np.float32)
        self.pixel_size_vu = np.zeros(2, dtype=np.float32)
        self.data_size_vu = np.zeros(2, dtype=np.intp)
        self.data_size_ts = np.zeros(2, dtype=np.intp)
        self.f1 = 0.0
        self.z1 = 0.0
        self.f2 = 0.0
        self.aperture_f1 = 0.0
        self.aperture_f2 = 0.0
        self.a = 0.0
        self.b = 0.0
        self.wavelength_range = np.array((0.4, 0.7)) # For polychromatic visible light
        self.wavelength_unit = 'um'

    def clone(self):
        return copy.deepcopy(self)

    def is_focused(self):
        return not (self.a == 0.0 or self.b == 0.0)

    def get_focused_distance(self):
        return np.abs(self.z1 * self.f1 / (self.z1 - self.f1))

    def get_distance_domain(self, d):
        return d * self.f1 / (d - self.f1)

    def print(self):
        for k, v in self.__dict__.items():
            print(' %18s : %s' % (k, str(v)))
        print(' %18s : %s' % ('z0', self.get_focused_distance()))

    def get_alphas(self, alphas, beam_geometry_in='cone', beam_geometry_out='parallel'):
        """Converts the refocusing alphas between beam geometries

        :param alphas: Sequence of alpha values as numpy array (numpy.array_like)
        :param beam_geometry_in: Beam shape of the input alphas. Options: 'parallel' | 'cone' (string)
        :param beam_geometry_out: Beam shape of the output alphas. Options: 'parallel' | 'cone' (string)

        :returns: The converted alphas
        :rtype: (numpy.array_like)
        """
        if beam_geometry_in.lower() == 'cone' and beam_geometry_out.lower() == 'parallel':
            return 2 - 1 / alphas
        elif beam_geometry_in.lower() == 'parallel' and beam_geometry_out.lower() == 'cone':
            return 1 / (2 - alphas)
        else:
            return alphas

    def get_f1(self, z0):
        return self.z1 * z0 / (self.z1 + z0)

    def get_number_of_subaperture_images(self):
        return np.prod(np.array(self.data_size_vu))

    def get_raw_detector_size(self):
        return self.data_size_ts * self.data_size_vu

    def get_scales(self, space='direct', domain='object'):
        if domain.lower() == 'object':
            M = self.get_focused_distance() / self.z1
        elif domain.lower() == 'image':
            M = 1
        else:
            raise ValueError("No known domain '%s'" % domain)

        if space.lower() == 'direct':
            scale_t = self.pixel_size_ts[0] * M
            scale_s = self.pixel_size_ts[1] * M
            scale_v = self.pixel_size_vu[0]
            scale_u = self.pixel_size_vu[1]
        elif space.lower() in ('fourier', 'fourier_slice'):
            scale_t = 1 / (self.pixel_size_ts[0] * self.data_size_ts[0] * M)
            scale_s = 1 / (self.pixel_size_ts[1] * self.data_size_ts[1] * M)
            scale_v = 1 / (self.pixel_size_vu[0] * self.data_size_vu[0])
            scale_u = 1 / (self.pixel_size_vu[1] * self.data_size_vu[1])
        else:
            raise ValueError("No known space '%s'" % space)
        return (scale_t, scale_s, scale_v, scale_u)

    def get_sigmatau_scales(self, space='direct', domain='object'):
        if domain.lower() == 'object':
            M = self.get_focused_distance() / self.z1
        elif domain.lower() == 'image':
            M = 1
        else:
            raise ValueError("No known domain '%s'" % domain)

        if space.lower() == 'direct':
            scale_tau = self.pixel_size_yx[0] * M
            scale_sigma = self.pixel_size_yx[1] * M
        elif space.lower() in ('fourier', 'fourier_slice'):
            scale_tau = 1 / (self.pixel_size_yx[0] * self.data_size_vu[0] * M)
            scale_sigma = 1 / (self.pixel_size_yx[1] * self.data_size_vu[1] * M)
        else:
            raise ValueError("No known space '%s'" % space)
        return (scale_tau, scale_sigma)

    def get_grid_points(self, space='direct', domain='object', oversampling=1):
        (scale_t, scale_s, scale_v, scale_u) = self.get_scales(space='direct', domain=domain)

        data_size = np.concatenate((np.array(self.data_size_ts) * oversampling, self.data_size_vu))
        data_span = (data_size - 1) / 2
        if space.lower() == 'direct':
            samp_t = np.linspace(-data_span[0], data_span[0], data_size[0]) * scale_t / oversampling
            samp_s = np.linspace(-data_span[1], data_span[1], data_size[1]) * scale_s / oversampling
            samp_v = np.linspace(-data_span[2], data_span[2], data_size[2]) * scale_v
            samp_u = np.linspace(-data_span[3], data_span[3], data_size[3]) * scale_u
        elif space.lower() in ('fourier', 'fourier_slice'):
            samp_t = np.fft.fftshift(np.fft.fftfreq(data_size[0])) / scale_t
            samp_s = np.fft.fftshift(np.fft.fftfreq(data_size[1])) / scale_s
            samp_v = np.fft.fftshift(np.fft.fftfreq(data_size[2])) / scale_v
            samp_u = np.fft.fftshift(np.fft.fftfreq(data_size[3])) / scale_u
        else:
            raise ValueError("No known space '%s'" % space)

        return (samp_v, samp_u, samp_t, samp_s)

    def get_sigmatau_grid_points(self, space='direct', domain='object'):
        data_size = np.array(self.data_size_vu)
        data_span = (data_size - 1) / 2

        (scale_tau, scale_sigma) = self.get_sigmatau_scales(space=space, domain=domain)

        samp_tau = np.linspace(-data_span[0], data_span[0], data_size[0]) * scale_tau
        samp_sigma = np.linspace(-data_span[1], data_span[1], data_size[1]) * scale_sigma

        return (samp_tau, samp_sigma)

    def get_sheared_coords(self, alpha, space='direct', beam_geometry='parallel', domain='object', transformation='shear', oversampling=1):
        (samp_v, samp_u, samp_t, samp_s) = self.get_grid_points(space=space, domain=domain, oversampling=oversampling)

        # We are multiplying the equations by alpha because we don't need
        # to rescale the images in lightfield refocusing
        if space.lower() == 'direct':
            (out_samp_v, out_samp_u, out_samp_t, out_samp_s) = np.meshgrid(samp_v, samp_u, samp_t, samp_s, indexing='ij')

            if beam_geometry.lower() == 'parallel':
                out_samp_t = out_samp_t + out_samp_v * (1 - 1 / alpha)
                out_samp_s = out_samp_s + out_samp_u * (1 - 1 / alpha)
            elif beam_geometry.lower() == 'cone':
                out_samp_t = out_samp_t / alpha + out_samp_v * (1 - 1 / alpha)
                out_samp_s = out_samp_s / alpha + out_samp_u * (1 - 1 / alpha)
            else:
                raise ValueError("Unknown beam_geometry '%s'" % beam_geometry)

            out_grid = np.array((out_samp_v, out_samp_u, out_samp_t, out_samp_s))
            return np.transpose(out_grid, axes=(1, 2, 3, 4, 0))

        elif space.lower() == 'fourier':
            (out_samp_v, out_samp_u, out_samp_t, out_samp_s) = np.meshgrid(samp_v, samp_u, samp_t, samp_s, indexing='ij')

            if transformation.lower() == 'shear':
                out_samp_v = out_samp_t * (1 - alpha)
                out_samp_u = out_samp_s * (1 - alpha)
                if beam_geometry.lower() == 'parallel':
                    pass
                elif beam_geometry.lower() == 'cone':
                    out_samp_t = out_samp_t * alpha
                    out_samp_s = out_samp_s * alpha
                else:
                    raise ValueError("No known beam_geometry '%s'" % beam_geometry)
            elif transformation.lower() == 'rotate':
                angle = np.arctan2(1 - alpha, alpha)
                cos_angle = np.cos(angle)
                sin_angle = np.sin(angle)

                out_samp_v = out_samp_t * sin_angle + out_samp_v * cos_angle
                out_samp_u = out_samp_s * sin_angle + out_samp_u * cos_angle
                out_samp_t = out_samp_t * cos_angle - out_samp_v * sin_angle
                out_samp_s = out_samp_s * cos_angle - out_samp_u * sin_angle
                if beam_geometry.lower() == 'parallel':
                    pass
                elif beam_geometry.lower() == 'cone':
                    out_samp_t = out_samp_t * alpha
                    out_samp_s = out_samp_s * alpha
                else:
                    raise ValueError("Unknown beam_geometry '%s'" % beam_geometry)
            elif transformation.lower() == 'filter':
                if beam_geometry.lower() == 'parallel':
                    out_samp_t = out_samp_t + out_samp_v * (1 - 1 / alpha)
                    out_samp_s = out_samp_s + out_samp_u * (1 - 1 / alpha)
                elif beam_geometry.lower() == 'cone':
                    out_samp_t = out_samp_t / alpha + out_samp_v * (1 - 1 / alpha)
                    out_samp_s = out_samp_s / alpha + out_samp_u * (1 - 1 / alpha)
                else:
                    raise ValueError("Unknown beam_geometry '%s'" % beam_geometry)
            else:
                raise ValueError("Unknown transformation '%s'" % transformation)

            out_grid = np.array((out_samp_v, out_samp_u, out_samp_t, out_samp_s))
            return np.transpose(out_grid, axes=(1, 2, 3, 4, 0))

        elif space.lower() == 'fourier_slice':
            (out_samp_t, out_samp_s) = np.meshgrid(samp_t, samp_s, indexing='ij')

            out_samp_v = out_samp_t * (1 - alpha)
            out_samp_u = out_samp_s * (1 - alpha)
            if beam_geometry.lower() == 'parallel':
                pass
            elif beam_geometry.lower() == 'cone':
                out_samp_t = out_samp_t * alpha
                out_samp_s = out_samp_s * alpha
            else:
                raise ValueError("Unknown beam_geometry '%s'" % beam_geometry)

            out_grid = np.array((out_samp_v, out_samp_u, out_samp_t, out_samp_s))
            return np.transpose(out_grid, axes=(1, 2, 0))

        else:
            raise ValueError("No known space '%s'" % space)

    def get_sampling_renorm_factor(self, alpha, space='direct', beam_geometry='parallel', domain='object'):
        (scale_t, scale_s, scale_v, scale_u) = self.get_scales(space=space, domain=domain)
        if beam_geometry.lower() == 'parallel':
            angle_t = np.arctan2((1 - alpha) * scale_v, scale_t)
            angle_s = np.arctan2((1 - alpha) * scale_u, scale_s)
        elif beam_geometry.lower() == 'cone':
            angle_t = np.arctan2((1 - alpha) * scale_v, alpha * scale_t)
            angle_s = np.arctan2((1 - alpha) * scale_u, alpha * scale_s)
        else:
            raise ValueError("Unknown beam_geometry '%s'" % beam_geometry)

        return 1 / np.abs(np.cos(angle_t) * np.cos(angle_s))

    def get_filter(self, alpha, bandwidth=0.04, beam_geometry='parallel', domain='object', oversampling=1):
        (scale_t, scale_s, scale_v, scale_u) = self.get_scales(space='direct', domain=domain)
        shear_grid = self.get_sheared_coords(alpha, space='fourier', \
                                             beam_geometry=beam_geometry, domain=domain, \
                                             transformation='filter', oversampling=oversampling)
        base_grid = self.get_sheared_coords(np.array((1, )), space='fourier', \
                                            beam_geometry=beam_geometry, domain=domain, \
                                            transformation='filter', oversampling=oversampling)
        (vv_s, uu_s, tt_s, ss_s) = (shear_grid[..., 0], shear_grid[..., 1], shear_grid[..., 2], shear_grid[..., 3])
        (vv_b, uu_b, tt_b, ss_b) = (base_grid[..., 0], base_grid[..., 1], base_grid[..., 2], base_grid[..., 3])
        d = (tt_s - tt_b) ** 2 + (ss_s - ss_b) ** 2
        bandwidth = bandwidth ** 2 / np.log(np.sqrt(2))
        return np.exp(-d / bandwidth)

    def get_theo_flatfield_raw(self, data_type=np.float32, over_sampling=5, mode='micro-image'):
        main_lens_radius = self.f1 / (2 * self.aperture_f1)
        lenslets_radius = self.f2 / (2 * self.aperture_f2)

        scale_x = self.pixel_size_yx[1] / over_sampling
        scale_y = self.pixel_size_yx[0] / over_sampling

        data_size = self.data_size_vu * over_sampling
        data_span = (data_size - 1) / 2
        samp_x = np.linspace(-data_span[1], data_span[1], data_size[1]) * scale_x
        samp_y = np.linspace(-data_span[0], data_span[0], data_size[0]) * scale_y

        [samp_y, samp_x] = np.meshgrid(samp_y, samp_x, indexing='ij')
        phi = np.arctan2(np.sqrt(samp_y ** 2 + samp_x ** 2), self.f2)

        center_distances = self.z1 * np.sin(phi)

        pixels_out = center_distances > (main_lens_radius + lenslets_radius)
        pixels_in = center_distances < (main_lens_radius - lenslets_radius)
        pixels_border = ~(pixels_out | pixels_in)

        mr = main_lens_radius
        mr2 = main_lens_radius ** 2
        lr = lenslets_radius
        lr2 = lenslets_radius ** 2
        d = center_distances[pixels_border]
        d2 = d ** 2
        # Intesection area formula from:
        # http://mathworld.wolfram.com/Circle-CircleIntersection.html
        overlap_area = lr2 * np.arccos((d2 + lr2 - mr2) / (2 * d * lr)) \
            + mr2 * np.arccos((d2 + mr2 - lr2) / (2 * d * mr)) \
            - np.sqrt((-d + lr + mr) * (d - lr + mr) * (d + lr - mr) * (d + lr + mr)) / 2

        bf = pixels_in.astype(data_type)
        bf[pixels_border] = overlap_area / (np.pi * lr2)
        bf = (np.cos(phi) ** 4) * bf

        bf = np.reshape(bf, (self.data_size_vu[0], over_sampling, self.data_size_vu[1], over_sampling))
        bf = np.sum(bf, axis=(1, 3))
        bf = np.reshape(bf, np.concatenate((self.data_size_vu, (1, 1)))) / (over_sampling ** 2)

        bf = bf.astype(data_type)
        if mode.lower() == 'micro-image':
            bf = np.reshape(bf, (1, 1, self.data_size_vu[0], self.data_size_vu[1]))
            bf = np.tile(bf, np.concatenate((self.data_size_ts, (1, 1))))
        elif mode.lower() == 'sub-aperture':
            bf = np.reshape(bf, (self.data_size_vu[0], self.data_size_vu[1], 1, 1))
            bf = np.tile(bf, np.concatenate(((1, 1), self.data_size_ts)))
        return bf

    def regrid(self, regrid_size, regrid_mode='interleave'):
        regrid_size = np.array(regrid_size, dtype=np.intp)
        if regrid_mode.lower() == 'interleave':

            self.data_size_vu = (self.data_size_vu * regrid_size[0:2]).astype(np.intp)
            self.data_size_ts = (self.data_size_ts * regrid_size[2:4]).astype(np.intp)

            self.pixel_size_vu = self.pixel_size_vu / regrid_size[0:2].astype(np.float32)
            self.pixel_size_ts = self.pixel_size_ts / regrid_size[2:4].astype(np.float32)

        elif regrid_mode.lower() == 'bin':
            if np.any(np.mod(self.data.shape, regrid_size) > 0):
                raise ValueError("When rebinning, the bins size should be a divisor of the image size. Size was: [%s], data size: [%s]" \
                                 % (", ".join(("%d" % x for x in regrid_size)), ", ".join(("%d" % x for x in self.data.shape))))

            self.data_size_vu = (self.data_size_vu / regrid_size[0:2]).astype(np.intp)
            self.data_size_ts = (self.data_size_ts / regrid_size[2:4]).astype(np.intp)

            self.pixel_size_vu = self.pixel_size_vu * regrid_size[0:2].astype(np.float32)
            self.pixel_size_ts = self.pixel_size_ts * regrid_size[2:4].astype(np.float32)

    def crop(self, crop_size_ts=None, crop_size_vu=None):
        if crop_size_vu is not None:
            self.data_size_vu = crop_size_vu.astype(np.intp)
        if crop_size_ts is not None:
            self.data_size_ts = crop_size_ts.astype(np.intp)

    def get_focused_patch_size(self):
        return self.b / self.a * self.f2 / self.aperture_f2 / self.pixel_size_yx

    def plot_phase_space_diagram(self, coordinates='su', show_central_images=None):
        (samp_v, samp_u, samp_t, samp_s) = self.get_grid_points(space='direct')
        if coordinates.lower() in ('su', 'us'):
            samp_abscissa = samp_s
            axis_name_abscissa = 's'
            samp_ordinate = samp_u
            axis_name_ordinate = 'u'
            det_psize_abscissa = self.pixel_size_yx[1]
        elif coordinates.lower() in ('tv', 'vt'):
            samp_abscissa = samp_t
            axis_name_abscissa = 't'
            samp_ordinate = samp_v
            axis_name_ordinate = 'v'
            det_psize_abscissa = self.pixel_size_yx[0]

        (grid_abs, grid_ord) = np.meshgrid(samp_abscissa, samp_ordinate, indexing='ij')
        print(self.a, self.b)
        if self.is_focused():
            samp_delta_det = np.linspace(-grid_abs.shape[1]/2, grid_abs.shape[1]/2, grid_abs.shape[1])
            grid_abs += samp_delta_det * self.a / self.b * det_psize_abscissa

        cm2inch = lambda x : np.array(x) / 2.54
        f_size = cm2inch([24, 18])
        f = plt.figure(None, figsize=f_size)

        upper_margin = np.array([0.1, 0.1])
        lower_margin = np.array([0.75, 0.75])
        ax_size = (f_size - (upper_margin + lower_margin)) / f_size
        rect = np.concatenate((lower_margin / f_size, ax_size))
        ax = f.add_axes(rect, label='image')

        ax.scatter(grid_abs, grid_ord)
        labels_fontsize = 22
        ax.set_xlabel('Coordinate: %s (mm)' % axis_name_abscissa, fontsize=labels_fontsize)
        ax.set_ylabel('Coordinate: %s (mm)' % axis_name_ordinate, fontsize=labels_fontsize)

        print('VU size:', self.data_size_vu)
        print('TS size:', self.data_size_ts)
        if show_central_images is not None:
            if coordinates.lower() in ('su', 'us'):
                size_abscissa = self.data_size_ts[1]
                size_ordinate = self.data_size_vu[1]
#                step_abscissa = self.pixel_size_ts[1]
#                step_ordinate = self.pixel_size_vu[1]
            elif coordinates.lower() in ('tv', 'vt'):
                size_abscissa = self.data_size_ts[0]
                size_ordinate = self.data_size_vu[0]
#                step_abscissa = self.pixel_size_ts[0]
#                step_ordinate = self.pixel_size_vu[0]
            center_abscissa = (np.floor(size_abscissa / 2)).astype(np.intp)
            center_ordinate = (np.floor(size_ordinate / 2)).astype(np.intp)

            x_micro = grid_abs[center_abscissa, :]
            y_micro = grid_ord[center_abscissa, :]
            x_sub_aperture = grid_abs[:, center_ordinate]
            y_sub_aperture = grid_ord[:, center_ordinate]
            if show_central_images.lower() == 'line':
                ax.plot(x_micro, y_micro, color=[1, 0, 0], linewidth=2)
                ax.plot(x_sub_aperture, y_sub_aperture, color=[0, 1, 0], linewidth=2)
            elif show_central_images.lower() == 'box':
                pass

        plt.show(block=False)
        return f, ax


class Lightfield(object):
    """Container class for the light-fields"""

    available_modes = ('micro-image', 'sub-aperture', 'epipolar_s', 'epipolar_t')

    def __init__(
            self, camera_type : Camera, data=None, flat=None, mask=None,
            mode='micro-image', dtype=np.float32, shifts_vu=(None, None)):
        """Initializes the Lightfield class

        :param camera_type: The Camera class that stores the metadata about the light-field (Camera)
        :param data: The actual light-field data in 4D format (numpy.array_like, default: None)
        :param flat: The flat field, usually encoding vignetting (numpy.array_like, default: None)
        :param mask: A mask indicating the pixels to use (numpy.array_like, default: None)
        :param mode: Mode of the data (string, default: 'micro-image')
        :param dtype: Data type of the light-field data (numpy.dtype, default: np.float32)
        """
        self.camera = camera_type
        if mode.lower() not in self.available_modes:
            raise ValueError('Not recognized mode: "%s"' % mode.lower())
        self.mode = mode.lower()
        self.data = data
        self.flat = flat
        self.mask = mask
        self.pixel_effective_size = camera_type.pixel_size_yx
        self.shifts_vu = shifts_vu

        if self.data is None:
            if self.mode.lower() == 'micro-image':
                data_size = np.concatenate((self.camera.data_size_ts, self.camera.data_size_vu))
            elif self.mode.lower() == 'sub-aperture':
                data_size = np.concatenate((self.camera.data_size_vu, self.camera.data_size_ts))
            elif self.mode.lower() == 'epipolar_s':
                data_size = np.array((self.camera.data_size_ts[1], self.camera.data_size_vu[0], self.camera.data_size_vu[1], self.camera.data_size_ts[0]))
            elif self.mode.lower() == 'epipolar_t':
                data_size = np.array((self.camera.data_size_vu[1], self.camera.data_size_ts[0], self.camera.data_size_ts[1], self.camera.data_size_vu[0]))
            else:
                raise ValueError("No light-field mode called: '%s'" % self.mode)
            self.data = np.zeros(data_size, dtype)
        else:
            self.data = self.data.astype(dtype)

    def clone(self):
        return copy.deepcopy(self)

    def set_mode(self, new_mode):
        """Set the required view for the given light-field object.

        :param new_mode: One of the following: {'micro-image', 'sub-aperture', 'epipolar_s', 'epipolar_t'} (string)
        """
        if new_mode.lower() == 'micro-image':
            self.set_mode_microimage()
        elif new_mode.lower() == 'sub-aperture':
            self.set_mode_subaperture()
        elif new_mode.lower() == 'epipolar_s':
            self.set_mode_epipolar_s()
        elif new_mode.lower() == 'epipolar_t':
            self.set_mode_epipolar_t
        else:
            raise ValueError("No light-field mode called: '%s'" % new_mode)

    def set_mode_epipolar_s(self):
        """Set the 'epipolar_s' mode"""
        if self.mode.lower() == 'micro-image':
            perm_op = (0, 3, 2, 1)
        elif self.mode.lower() == 'sub-aperture':
            perm_op = (2, 1, 0, 3)
        elif self.mode.lower() == 'epipolar_s':
            perm_op = (0, 1, 2, 3)
        elif self.mode.lower() == 'epipolar_t':
            perm_op = (2, 3, 0, 1)

        self.mode = 'epipolar_s'

        self.data = np.transpose(self.data, perm_op)
        if self.flat is not None:
            self.flat = np.transpose(self.flat, perm_op)
        if self.mask is not None:
            self.mask = np.transpose(self.mask, perm_op)

    def set_mode_epipolar_t(self):
        """Set the 'epipolar_t' mode"""
        if self.mode.lower() == 'micro-image':
            perm_op = (2, 1, 0, 3)
        elif self.mode.lower() == 'sub-aperture':
            perm_op = (0, 3, 2, 1)
        elif self.mode.lower() == 'epipolar_s':
            perm_op = (2, 3, 0, 1)
        elif self.mode.lower() == 'epipolar_t':
            perm_op = (0, 1, 2, 3)

        self.mode = 'epipolar_t'

        self.data = np.transpose(self.data, perm_op)
        if self.flat is not None:
            self.flat = np.transpose(self.flat, perm_op)
        if self.mask is not None:
            self.mask = np.transpose(self.mask, perm_op)

    def set_mode_microimage(self):
        """Set the 'micro-image' mode"""
        if self.mode.lower() == 'micro-image':
            perm_op = (0, 1, 2, 3)
        elif self.mode.lower() == 'sub-aperture':
            perm_op = (2, 3, 0, 1)
        elif self.mode.lower() == 'epipolar_s':
            perm_op = (0, 3, 2, 1)
        elif self.mode.lower() == 'epipolar_t':
            perm_op = (2, 1, 0, 3)

        self.mode = 'micro-image'

        self.data = np.transpose(self.data, perm_op)
        if self.flat is not None:
            self.flat = np.transpose(self.flat, perm_op)
        if self.mask is not None:
            self.mask = np.transpose(self.mask, perm_op)

    def set_mode_subaperture(self):
        """Set the 'sub-aperture' image mode"""
        if self.mode.lower() == 'micro-image':
            perm_op = (2, 3, 0, 1)
        elif self.mode.lower() == 'sub-aperture':
            perm_op = (0, 1, 2, 3)
        elif self.mode.lower() == 'epipolar_s':
            perm_op = (2, 1, 0, 3)
        elif self.mode.lower() == 'epipolar_t':
            perm_op = (0, 3, 2, 1)

        self.mode = 'sub-aperture'

        self.data = np.transpose(self.data, perm_op)
        if self.flat is not None:
            self.flat = np.transpose(self.flat, perm_op)
        if self.mask is not None:
            self.mask = np.transpose(self.mask, perm_op)

    def get_raw_detector_picture(self, image='data'):
        """Returns the detector data, or the flat image in the raw detector format

        :param image: Selects whether we want the detector data, or the flat image (string)
        :returns: The requested image
        :rtype: numpy.array_like
        """
        if self.mode.lower() == 'micro-image':
            perm_op = (0, 2, 1, 3)
        elif self.mode.lower() == 'sub-aperture':
            perm_op = (2, 0, 3, 1)
        elif self.mode.lower() == 'epipolar_s':
            perm_op = (3, 0, 2, 1)
        elif self.mode.lower() == 'epipolar_t':
            perm_op = (0, 1, 2, 3)

        if image.lower() == 'data':
            data_raw = np.transpose(self.data, perm_op)
        elif image.lower() == 'flat':
            data_raw = np.transpose(self.flat, perm_op)
        elif image.lower() == 'mask':
            data_raw = np.transpose(self.mask, perm_op)
        else:
            raise ValueError('Unknown image type: %s' % image)
        return np.reshape(data_raw, self.camera.get_raw_detector_size())

    def set_raw_detector_picture(self, data_raw, image='data'):
        """Sets the detector data, or the flat image from an image in raw
        detector format

        :param data_raw: The new detector image (numpy.array_like)
        :param image: Selects whether we want the detector data, or the flat image (string)
        """
        in_size = (self.camera.data_size_ts[0], self.camera.data_size_vu[0], \
                   self.camera.data_size_ts[1], self.camera.data_size_vu[1])
        data_raw = np.reshape(data_raw, in_size)

        if self.mode.lower() == 'micro-image':
            perm_op = (0, 2, 1, 3)
        elif self.mode.lower() == 'sub-aperture':
            perm_op = (1, 3, 0, 2)
        elif self.mode.lower() == 'epipolar_s':
            perm_op = (0, 3, 1, 2)
        elif self.mode.lower() == 'epipolar_t':
            perm_op = (1, 2, 0, 3)

        if image.lower() == 'data':
            self.data = np.transpose(data_raw, perm_op)
        elif image.lower() == 'flat':
            self.flat = np.transpose(data_raw, perm_op)
        elif image.lower() == 'mask':
            self.mask = np.transpose(data_raw, perm_op)
        else:
            raise ValueError('Unknown image type: %s' % image)

    def get_sub_aperture_image(self, v, u, image='data'):
        """Returns a chosen sub-aperture image from either detector data, or
        the flat field

        :param u: U coordinate of the image (int)
        :param v: V coordinate of the image (int)
        :param image: Selects whether we want the detector data, or the flat image (string)
        :returns: The requested image
        :rtype: numpy.array_like
        """
        if v < 0 or v > self.camera.data_size_vu[0]:
            raise ValueError('V coordinate %d is outside the range: [%d %d]' % (v, 0, self.camera.data_size_vu[0]))
        if u < 0 or u > self.camera.data_size_vu[1]:
            raise ValueError('U coordinate %d is outside the range: [%d %d]' % (u, 0, self.camera.data_size_vu[1]))

        if self.mode.lower() == 'micro-image':
            slice_op = (slice(None), slice(None), v, u)
        elif self.mode.lower() == 'sub-aperture':
            slice_op = (v, u, slice(None), slice(None))
        elif self.mode.lower() == 'epipolar_s':
            slice_op = (slice(None), u, v, slice(None))
        elif self.mode.lower() == 'epipolar_t':
            slice_op = (v, slice(None), slice(None), u)

        if image.lower() == 'data':
            return self.data[slice_op]
        elif image.lower() == 'flat':
            return self.flat[slice_op]
        elif image.lower() == 'mask':
            return self.mask[slice_op]
        else:
            raise ValueError('Unknown image type: %s' % image)

    def get_sub_aperture_images(self, image='data'):
        """Returns all the sub-aperture images from either detector data, or
        the flat field

        :param image: Selects whether we want the detector data, or the flat image (string)
        :returns: The requested image
        :rtype: numpy.array_like
        """
        if self.mode.lower() == 'micro-image':
            perm_op = (2, 0, 3, 1)
        elif self.mode.lower() == 'sub-aperture':
            perm_op = (0, 2, 1, 3)
        elif self.mode.lower() == 'epipolar_s':
            perm_op = (0, 1, 2, 3)
        elif self.mode.lower() == 'epipolar_t':
            perm_op = (1, 0, 3, 2)

        if image.lower() == 'data':
            data_raw = np.transpose(self.data, perm_op)
        elif image.lower() == 'flat':
            data_raw = np.transpose(self.flat, perm_op)
        elif image.lower() == 'mask':
            data_raw = np.transpose(self.mask, perm_op)
        else:
            raise ValueError('Unknown image type: %s' % image)
        photo_size_2D = np.array([
                np.array(self.camera.data_size_vu[0]) * np.array(self.camera.data_size_ts[0]),
                np.array(self.camera.data_size_vu[1]) * np.array(self.camera.data_size_ts[1]) ])
        return np.reshape(data_raw, photo_size_2D)

    def get_photograph(self, image='data'):
        """Computes the refocused photograph at z0

        :param image: Selects whether we want the detector data, or the flat image (string)
        :returns: The refocused photograph at z0
        :rtype: numpy.array_like
        """

        if not self.mode == 'sub-aperture':
            lf_sa = self.clone()
            lf_sa.set_mode_subaperture()
        else:
            lf_sa = self

        if image.lower() == 'data':
            photo = np.sum(lf_sa.data, axis=(0, 1))
        elif image.lower() == 'flat':
            photo = np.sum(lf_sa.flat, axis=(0, 1))
        elif image.lower() == 'mask':
            photo = np.sum(lf_sa.mask, axis=(0, 1))
        else:
            raise ValueError('Unknown image type: %s' % image)
        return photo / np.prod(self.camera.data_size_vu)

    def pad(self, paddings, method='constant', pad_value=(0,)):
        """Pad a light-field

        :param paddings: Padding to add (<4x1> numpy.array_like)
        :param method: Padding method. Possible values: 'constant' | ‘edge’ | ‘linear_ramp’
            | ‘maximum’ | ‘mean’ | ‘median’ | ‘minimum’ | ‘reflect’ | ‘symmetric’ | ‘wrap’ (string)
        """
        old_mode = self.mode
        self.set_mode_subaperture()

        paddings = np.array(paddings)
        if len(paddings.shape) > 0 and paddings.shape[0] > 1 and (len(paddings.shape) == 1 or paddings.shape[1] == 1):
            paddings = np.tile(paddings, (2, 1))
            paddings = paddings.transpose((1, 0))
        if not method == 'constant':
            self.data = np.pad(self.data, pad_width=paddings, mode=method)
            if self.flat is not None:
                self.flat = np.pad(self.flat, pad_width=paddings, mode=method)
            if self.mask is not None:
                self.mask = np.pad(self.mask, pad_width=paddings, mode=method)
        else:
            self.data = np.pad(self.data, pad_width=paddings, mode=method, constant_values=pad_value)
            if self.flat is not None:
                self.flat = np.pad(self.flat, pad_width=paddings, mode=method, constant_values=pad_value)
            if self.mask is not None:
                self.mask = np.pad(self.mask, pad_width=paddings, mode=method, constant_values=pad_value)

        self.camera.data_size_ts = np.array(self.data.shape[2:4]).astype(np.int)
        self.camera.data_size_vu = np.array(self.data.shape[0:2]).astype(np.int)

        self.set_mode(old_mode)

    def regrid(self, regrid_size, regrid_mode='interleave'):
        regrid_size = np.array(regrid_size, dtype=np.intp)
        if regrid_mode.lower() == 'interleave':
            old_mode = self.mode
            self.set_mode_subaperture()

            base_grid = self.camera.get_grid_points(space='direct')

            self.camera.data_size_vu = ((np.array(self.camera.data_size_vu) - 1) * regrid_size[0:2]).astype(np.intp) + 1
            self.camera.data_size_ts = ((np.array(self.camera.data_size_ts) - 1) * regrid_size[2:4]).astype(np.intp) + 1

            self.camera.pixel_size_vu = np.array(self.camera.pixel_size_vu) / regrid_size[0:2].astype(np.float32)
            self.camera.pixel_size_ts = np.array(self.camera.pixel_size_ts) / regrid_size[2:4].astype(np.float32)

            new_grid = self.camera.get_grid_points(space='direct')
            new_grid = np.meshgrid(*new_grid, indexing='ij')
            new_grid = np.array(new_grid)
            new_grid = np.transpose(new_grid, axes=(1, 2, 3, 4, 0))

            interp_data = sp.interpolate.RegularGridInterpolator(base_grid, self.data, bounds_error=False, fill_value=0)
            self.data = interp_data(new_grid)
            if self.flat is not None:
                interp_flat = sp.interpolate.RegularGridInterpolator(base_grid, self.flat, bounds_error=False, fill_value=0)
                self.flat = interp_flat(new_grid)
            if self.mask is not None:
                interp_mask = sp.interpolate.RegularGridInterpolator(base_grid, self.mask, bounds_error=False, fill_value=0)
                self.mask = interp_mask(new_grid)

            self.set_mode(old_mode)
        elif regrid_mode.lower() == 'copy':
            old_mode = self.mode
            self.set_mode_subaperture()

            temp_shape = [
                    self.camera.data_size_vu[0], 1,
                    self.camera.data_size_vu[1], 1,
                    self.camera.data_size_ts[0], 1,
                    self.camera.data_size_ts[1], 1]
            tile_size = [
                    1, regrid_size[0], 1, regrid_size[1],
                    1, regrid_size[2], 1, regrid_size[3]]
            copied_data = np.reshape(self.data, temp_shape)
            copied_data = np.tile(copied_data, tile_size)
            if self.flat is not None:
                copied_flat = np.reshape(self.flat, temp_shape)
                copied_flat = np.tile(copied_flat, tile_size)
            if self.mask is not None:
                copied_mask = np.reshape(self.mask, temp_shape)
                copied_mask = np.tile(copied_mask, tile_size)

            self.camera.data_size_vu = (np.array(self.camera.data_size_vu) * regrid_size[0:2]).astype(np.intp)
            self.camera.data_size_ts = (np.array(self.camera.data_size_ts) * regrid_size[2:4]).astype(np.intp)

            self.camera.pixel_size_vu = np.array(self.camera.pixel_size_vu) / regrid_size[0:2].astype(np.float32)
            self.camera.pixel_size_ts = np.array(self.camera.pixel_size_ts) / regrid_size[2:4].astype(np.float32)

            self.data = np.reshape(copied_data, np.concatenate((self.camera.data_size_vu, self.camera.data_size_ts)))
            if self.flat is not None:
                self.flat = np.reshape(copied_flat, np.concatenate((self.camera.data_size_vu, self.camera.data_size_ts)))
            if self.mask is not None:
                self.mask = np.reshape(copied_mask, np.concatenate((self.camera.data_size_vu, self.camera.data_size_ts)))

            self.set_mode(old_mode)
        elif regrid_mode.lower() == 'bin':
            if np.any(np.mod(self.data.shape, regrid_size) > 0):
                raise ValueError("When rebinning, the bins size should be a divisor of the image size. Size was: [%s], data size: [%s]" \
                                 % (", ".join(("%d" % x for x in regrid_size)), ", ".join(("%d" % x for x in self.data.shape))))

            old_mode = self.mode
            self.set_mode_subaperture()

            self.camera.data_size_vu = (self.camera.data_size_vu / regrid_size[0:2]).astype(np.intp)
            self.camera.data_size_ts = (self.camera.data_size_ts / regrid_size[2:4]).astype(np.intp)

            self.camera.pixel_size_vu = self.camera.pixel_size_vu * regrid_size[0:2].astype(np.float32)
            self.camera.pixel_size_ts = self.camera.pixel_size_ts * regrid_size[2:4].astype(np.float32)

            new_data_size = np.array((self.camera.data_size_vu[0], regrid_size[0], \
                                      self.camera.data_size_vu[1], regrid_size[1], \
                                      self.camera.data_size_ts[0], regrid_size[2], \
                                      self.camera.data_size_ts[1], regrid_size[3]), dtype= np.intp)

            self.data = np.reshape(self.data, new_data_size)
            self.data = np.sum(self.data, axis=(1, 3, 5, 7)) / np.prod(regrid_size)
            if self.flat is not None:
                self.flat = np.reshape(self.flat, new_data_size)
                self.flat = np.sum(self.flat, axis=(1, 3, 5, 7)) / np.prod(regrid_size)
            if self.mask is not None:
                self.mask = np.reshape(self.mask, new_data_size)
                self.mask = np.sum(self.mask, axis=(1, 3, 5, 7)) / np.prod(regrid_size)

            self.set_mode(old_mode)

    def crop(self, crop_size_ts=None, crop_size_vu=None):
        """Crop a light-field

        :param crop_size_ts: Either new size in the (t, s) coordinates or a ROI (<2x1> or <4x1> numpy.array_like)
        :param crop_size_vu: Either new size in the (v, u) coordinates or a ROI (<2x1> or <4x1> numpy.array_like)
        """

        old_mode = self.mode
        self.set_mode_subaperture()

        if crop_size_ts is not None:
            crop_size_ts = np.array(crop_size_ts)
            if len(crop_size_ts) == 2:
                # Centered
                center_data_ts = (self.camera.data_size_ts - 1) / 2
                center_roi_ts = (crop_size_ts - 1) / 2
                start_ts = np.floor(center_data_ts - center_roi_ts).astype(np.int)
                end_ts = (start_ts + crop_size_ts).astype(np.int)
                crop_roi_ts = np.concatenate((start_ts, end_ts))
            else:
                crop_roi_ts = crop_size_ts
                crop_size_ts = crop_roi_ts[2:] - crop_roi_ts[:2]

            self.data = self.data[..., crop_roi_ts[0]:crop_roi_ts[2], crop_roi_ts[1]:crop_roi_ts[3]]
            if self.flat is not None:
                self.flat = self.flat[..., crop_roi_ts[0]:crop_roi_ts[2], crop_roi_ts[1]:crop_roi_ts[3]]
            if self.mask is not None:
                self.mask = self.mask[..., crop_roi_ts[0]:crop_roi_ts[2], crop_roi_ts[1]:crop_roi_ts[3]]

        if crop_size_vu is not None:
            crop_size_vu = np.array(crop_size_vu)
            if len(crop_size_vu) == 2:
                # Centered
                center_data_vu = (self.camera.data_size_vu - 1) / 2
                center_roi_vu = (crop_size_vu - 1) / 2
                start_vu = np.floor(center_data_vu - center_roi_vu).astype(np.int)
                end_vu = (start_vu + crop_size_vu).astype(np.int)
                crop_roi_vu = np.concatenate((start_vu, end_vu))
            else:
                crop_roi_vu = crop_size_vu
                crop_size_vu = crop_roi_vu[2:] - crop_roi_vu[:2]

            self.data = self.data[crop_roi_vu[0]:crop_roi_vu[2], crop_roi_vu[1]:crop_roi_vu[3], ...]
            if self.flat is not None:
                self.flat = self.flat[crop_roi_vu[0]:crop_roi_vu[2], crop_roi_vu[1]:crop_roi_vu[3], ...]
            if self.mask is not None:
                self.mask = self.mask[crop_roi_vu[0]:crop_roi_vu[2], crop_roi_vu[1]:crop_roi_vu[3], ...]

        self.camera.crop(crop_size_ts=crop_size_ts, crop_size_vu=crop_size_vu)

        self.set_mode(old_mode)

    def get_central_subaperture(self, origin_vu=None):
        center_vu = (np.array(self.camera.data_size_vu, dtype=np.float) - 1) / 2
        if origin_vu is None:
            origin_vu = np.array((0., 0.))
        if np.any(np.abs(origin_vu) > center_vu):
            raise ValueError('Origin VU (%f, %f) outside of bounds' % (origin_vu[0], origin_vu[1]))

        origin_vu = center_vu + np.array(origin_vu)
        lower_ind = np.floor(origin_vu).astype(np.int)
        upper_ind = lower_ind + 1
        lower_c = upper_ind - origin_vu
        upper_c = 1 - lower_c
        out_img = np.zeros(self.camera.data_size_ts)
        eps = np.finfo(np.float32).eps
        if lower_c[0] > eps and lower_c[1] > eps:
            out_img += lower_c[0] * lower_c[1] * self.get_sub_aperture_image(lower_ind[0], lower_ind[1], image='data')
        if upper_c[0] > eps and lower_c[1] > eps:
            out_img += upper_c[0] * lower_c[1] * self.get_sub_aperture_image(upper_ind[0], lower_ind[1], image='data')
        if lower_c[0] > eps and upper_c[1] > eps:
            out_img += lower_c[0] * upper_c[1] * self.get_sub_aperture_image(lower_ind[0], upper_ind[1], image='data')
        if upper_c[0] > eps and upper_c[1] > eps:
            out_img += upper_c[0] * upper_c[1] * self.get_sub_aperture_image(upper_ind[0], upper_ind[1], image='data')
        return out_img

    def get_sub_lightfields(self, sub_aperture_size=5):
        v_lower_range = range(self.camera.data_size_vu[0] - sub_aperture_size + 1)
        v_upper_range = range(sub_aperture_size, self.camera.data_size_vu[0] + 1)

        u_lower_range = range(self.camera.data_size_vu[1] - sub_aperture_size + 1)
        u_upper_range = range(sub_aperture_size, self.camera.data_size_vu[1] + 1)

        current_mode = self.mode
        self.set_mode_subaperture()

        lfs = []
        empty_lf = Lightfield(camera_type=self.camera.clone(), mode='sub-aperture', dtype=self.data.dtype)
        empty_lf.camera.data_size_vu = np.array([sub_aperture_size, sub_aperture_size])

        for v_lower, v_upper in zip(v_lower_range, v_upper_range):
            u_lfs = []
            for u_lower, u_upper in zip(u_lower_range, u_upper_range):
                lf = empty_lf.clone()
                lf.data = self.data[v_lower:v_upper, u_lower:u_upper, ...]
                if self.flat is not None:
                    lf.flat = self.flat[v_lower:v_upper, u_lower:u_upper, ...]
                if self.mask is not None:
                    lf.mask = self.mask[v_lower:v_upper, u_lower:u_upper, ...]
                u_lfs.append(lf)
            lfs.append(u_lfs)

        self.set_mode(current_mode)
        return lfs

    def get_sub_lightfield(self, center_u=None, center_v=None, sub_aperture_size=5):
        current_mode = self.mode
        self.set_mode_subaperture()

        if center_v is None:
            center_v = int((self.camera.data_size_vu[0] - 1) / 2)
        v_lower, v_upper = center_v - sub_aperture_size, center_v + sub_aperture_size + 1
        if center_u is None:
            center_u = int((self.camera.data_size_vu[1] - 1) / 2)
        u_lower, u_upper = center_u - sub_aperture_size, center_u + sub_aperture_size + 1

        data_size_vu = np.array((sub_aperture_size, ) * 2) * 2 + 1
        lf = Lightfield(camera_type=self.camera.clone(), mode='sub-aperture', dtype=self.data.dtype)
        lf.camera.data_size_vu = data_size_vu

        data_size = np.concatenate((data_size_vu, self.camera.data_size_ts))
        lf.data = self.data[v_lower:v_upper, u_lower:u_upper, ...]
        lf.data = np.reshape(lf.data, data_size)
        if self.flat is not None:
            lf.flat = self.flat[v_lower:v_upper, u_lower:u_upper, ...]
            lf.flat = np.reshape(lf.flat, data_size)
        if self.mask is not None:
            lf.mask = self.mask[v_lower:v_upper, u_lower:u_upper, ...]
            lf.mask = np.reshape(lf.mask, data_size)

        self.set_mode(current_mode)
        return lf

