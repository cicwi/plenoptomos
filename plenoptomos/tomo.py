#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 11:30:00 2017

@author: vigano

This module contains the tomographic refocusing routines.
"""

import numpy as np

import time as tm

import astra

from . import lightfield
from . import solvers


class Projector(object):
    """Projector class: it allows to forward-project and back-project light-fields.

    This class should not be used directly, but rather through the functions
    provided in the containing module.
    """

    def __init__(
            self, camera : lightfield.Camera, zs, mask=None, vignetting_int=None,
            beam_geometry='cone', domain='object', psf_d=None,
            up_sampling=1, super_sampling=1,
            mode='independent', shifts_vu=(None, None), gpu_index=-1):
        self.mode = mode # can be: {'independent' | 'simultaneous' | 'range'}
        self.up_sampling = up_sampling
        self.super_sampling = super_sampling
        self.gpu_index = gpu_index
        self.beam_geometry = beam_geometry
        self.domain = domain

        self.projectors = []
        self.Ws = []

        self.camera = camera
        self.img_size = np.concatenate((camera.data_size_vu, camera.data_size_ts))
        self.img_size_us = np.concatenate((camera.data_size_vu, np.array(camera.data_size_ts) * self.up_sampling))
        self.photo_size_2D = np.array(camera.data_size_vu) * np.array(camera.data_size_ts)

        self.mask = mask
        if self.mask is not None:
            self.mask = np.reshape(mask, np.concatenate(((-1, ), self.img_size)))
        if vignetting_int is not None:
            self.vignetting_int = np.reshape(vignetting_int, np.concatenate(((-1, ), self.img_size)))
            if self.mask is not None:
                self.vignetting_int *= self.mask
        else:
            self.vignetting_int = self.mask

        self.shifts_vu = shifts_vu

        if psf_d is None:
            psf_d = []
        self.psf = psf_d

        self._initialize_geometry(zs)
        self._init_psf_mask_correction()

    def __enter__(self):
        self._initialize_projectors()
        return self

    def __exit__(self, *args):
        self.reset()

    def _initialize_geometry(self, zs):
        """The this function produces the ASTRA geometry needed to refocus or
        project light-fields.

        This function is based on the geometry described in the following articles:
        [1] N. Viganò, et al., “Tomographic approach for the quantitative scene
        reconstruction from light field images,” Opt. Express, vol. 26, no. 18,
        p. 22574, Sep. 2018.
        """
        (samp_v, samp_u, _, _) = self.camera.get_grid_points(space='direct')

        (virt_pos_v, virt_pos_u) = np.meshgrid(samp_v, samp_u, indexing='ij')

        (_, _, scale_v, scale_u) = self.camera.get_scales(space='direct', domain=self.domain)
        if self.shifts_vu[0] is not None:
            virt_pos_v += self.shifts_vu[0] * scale_v
        if self.shifts_vu[1] is not None:
            virt_pos_u += self.shifts_vu[1] * scale_u

        if self.domain.lower() == 'object':
            ref_z = self.camera.get_focused_distance()
        elif self.domain.lower() == 'image':
            ref_z = self.camera.z1
        else:
            raise ValueError('No known domain "%s"' % self.domain)

        M = ref_z / self.camera.z1
        camera_pixel_size_ts_us = np.array(self.camera.pixel_size_ts) / self.up_sampling

        renorm_resolution_factor = np.mean(camera_pixel_size_ts_us) * M
        zs_n = zs / renorm_resolution_factor

        voxel_size_obj_space_ts = camera_pixel_size_ts_us * M

        num_imgs = self.camera.get_number_of_subaperture_images()
        imgs_pixel_size_s = np.zeros( (num_imgs, 3) )
        imgs_pixel_size_t = np.zeros( (num_imgs, 3) )
        imgs_pixel_size_s[:, 0] = voxel_size_obj_space_ts[1] / renorm_resolution_factor
        imgs_pixel_size_t[:, 1] = voxel_size_obj_space_ts[0] / renorm_resolution_factor

        if self.camera.is_focused():
            (samp_tau, samp_sigma) = self.camera.get_sigmatau_grid_points(space='direct', domain=self.domain)
            (samp_tau, samp_sigma) = np.meshgrid(samp_tau, samp_sigma, indexing='ij')
            m = - self.camera.a / self.camera.b
            acq_sa_imgs_ts = np.empty((num_imgs, 3))
            acq_sa_imgs_ts[:, 0] = samp_sigma.flatten() * m * M
            acq_sa_imgs_ts[:, 1] = samp_tau.flatten() * m * M
        else:
            acq_sa_imgs_ts = np.zeros((num_imgs, 3))
        acq_sa_imgs_ts[:, 2] = ref_z
        acq_sa_imgs_ts /= renorm_resolution_factor

        ph_imgs_vu = np.empty((num_imgs, 3))
        ph_imgs_vu[:, 0] = virt_pos_u.flatten() / renorm_resolution_factor
        ph_imgs_vu[:, 1] = virt_pos_v.flatten() / renorm_resolution_factor
        ph_imgs_vu[:, 2] = 0

        up_sampled_array_size = np.array(self.camera.data_size_ts) * self.up_sampling
        lims_s = (np.array([-1., 1.]) * up_sampled_array_size[1] / 2) * voxel_size_obj_space_ts[1] / renorm_resolution_factor
        lims_t = (np.array([-1., 1.]) * up_sampled_array_size[0] / 2) * voxel_size_obj_space_ts[0] / renorm_resolution_factor

        num_dists = len(zs_n)
        if self.mode in ('independent', 'simultaneous'):
            self.vol_size = np.concatenate((up_sampled_array_size, (1, )))
        elif self.mode == 'range':
            self.vol_size = np.concatenate((up_sampled_array_size, (num_dists, )))
        else:
            raise ValueError("Mode: '%s' not allowed! Possible choices are: 'independent' | 'simultaneous' | 'range'" % self.mode)

        if self.mode in ('independent', 'simultaneous'):
            self.vol_geom = []
            for d in zs_n:
                lims_z = d + np.array([-1., 1.]) / 2
                self.vol_geom.append(astra.create_vol_geom(self.vol_size[0], self.vol_size[1], self.vol_size[2], lims_s[0], lims_s[1], lims_t[0], lims_t[1], lims_z[0], lims_z[1]))
        elif self.mode == 'range':
            if num_dists > 1:
                lims_z = (zs_n[0], zs_n[-1])
                delta = (lims_z[1] - lims_z[0]) / (num_dists - 1)
                lims_z = lims_z + np.array([-1., 1.]) * delta / 2
            else:
                lims_z = zs_n + np.array([-1., 1.]) / 2
            self.vol_geom = [astra.create_vol_geom(self.vol_size[0], self.vol_size[1], self.vol_size[2], lims_s[0], lims_s[1], lims_t[0], lims_t[1], lims_z[0], lims_z[1])]

        if self.beam_geometry.lower() == 'cone':
            det_geometry = np.hstack([ph_imgs_vu, acq_sa_imgs_ts, imgs_pixel_size_s, imgs_pixel_size_t])
            self.proj_geom = astra.create_proj_geom('cone_vec', self.img_size_us[2], self.img_size_us[3], det_geometry)
        elif self.beam_geometry.lower() == 'parallel':
            proj_dir = acq_sa_imgs_ts - ph_imgs_vu
            img_dist = np.sqrt(np.sum(proj_dir ** 2, axis=1))
            img_dist = np.expand_dims(img_dist, axis=1)
            proj_dir /= img_dist

            det_geometry = np.hstack([proj_dir, acq_sa_imgs_ts, imgs_pixel_size_s, imgs_pixel_size_t])
            self.proj_geom = astra.create_proj_geom('parallel3d_vec', self.img_size_us[2], self.img_size_us[3], det_geometry)
        else:
            raise ValueError("Beam shape: '%s' not allowed! Possible choices are: 'parallel' | 'cone'" % self.beam_geometry)

    def _init_psf_mask_correction(self):
        self.mask_psf_renorm = [None] * len(self.psf)
        if self.mask is not None:
            for ii, p in enumerate(self.psf):
                psf_on_subpixel = p.data_format is not None and p.data_format.lower() == 'subpixel'
                if not psf_on_subpixel:
                    psf_on_raw = p.data_format is not None and p.data_format.lower() == 'raw'
                    if psf_on_raw:
                        # handle 2D flattening
                        y = np.transpose(self.mask, (0, 3, 1, 4, 2))
                        img_size_det = y.shape
                        y = np.reshape(y, np.concatenate(((-1, ), self.photo_size_2D)))
                    else:
                        y = self.mask

                    y = p.apply_psf_direct(y)

                    if psf_on_raw:
                        y = np.reshape(y, img_size_det)
                        y = np.transpose(y, (0, 2, 4, 1, 3))

                    self.mask_psf_renorm[ii] = 1 / (y + (np.abs(y) < 1e-5))

    def reset(self):
        for p in self.projectors:
            astra.projector.delete(p)
        self.projectors = []

    def _initialize_projectors(self):
        # Volume downscaling option and similar:
        opts = {
            'VoxelSuperSampling': self.super_sampling,
            'DetectorSuperSampling': self.super_sampling,
            'GPUindex': self.gpu_index
            }
        for vg in self.vol_geom:
            proj_id = astra.create_projector('cuda3d', self.proj_geom, vg, opts)
            self.projectors.append(proj_id)
            self.Ws.append(astra.OpTomo(proj_id))

    def FP(self, x):
        """Forward-projection function

        :param x: The volume to project (numpy.array_like)

        :returns: The projected light-field
        :rtype: numpy.array_like
        """
        proj_stack_shape = (
                len(self.Ws), self.img_size_us[-2],
                self.camera.get_number_of_subaperture_images(),
                self.img_size_us[-1])

        if self.mode == 'range':
            y = self.Ws[0].FP(np.squeeze(x))
        elif self.mode in ('simultaneous', 'independent'):
            y = np.empty(proj_stack_shape, x.dtype)
            for ii, W in enumerate(self.Ws):
                temp = x[ii, :, :]
                temp = W.FP(temp[np.newaxis, ...])
                y[ii, :, :, :] = temp
        y = np.transpose(y, (0, 2, 1, 3))

        for p in self.psf:
            psf_on_subpixel = p.data_format is not None and p.data_format.lower() == 'subpixel'
            if psf_on_subpixel:
                y = p.apply_psf_direct(y)

        if self.up_sampling > 1:
            y = np.reshape(y, (-1, self.camera.get_number_of_subaperture_images(), self.img_size[-2], self.up_sampling, self.img_size[-1], self.up_sampling))
            y = np.sum(y, axis=(-3, -1)) / (self.up_sampling ** 2)

        y = np.reshape(y, np.concatenate(((-1, ), self.img_size)))

        if self.vignetting_int is not None:
            y *= self.vignetting_int

        if self.mode == 'simultaneous':
            y = np.sum(y, axis=0, keepdims=True)

        for ii, p in enumerate(self.psf):
            psf_on_subpixel = p.data_format is not None and p.data_format.lower() == 'subpixel'
            if not psf_on_subpixel:
                psf_on_raw = p.data_format is not None and p.data_format.lower() == 'raw'
                if psf_on_raw:
                    # handle 2D flattening
                    y = np.transpose(y, (0, 3, 1, 4, 2))
                    img_size_det = y.shape
                    y = np.reshape(y, np.concatenate(((-1, ), self.photo_size_2D)))

                y = p.apply_psf_direct(y)

                if psf_on_raw:
                    y = np.reshape(y, img_size_det)
                    y = np.transpose(y, (0, 2, 4, 1, 3))

                if self.mask_psf_renorm[ii] is not None:
                    y *= self.mask_psf_renorm[ii]

        return y

    def BP(self, y):
        """Back-projection function

        :param x: Light-field to back-project (numpy.array_like)

        :returns: The back-projected volume
        :rtype: numpy.array_like
        """
        y = np.reshape(y, np.concatenate(((-1, ), self.img_size)))

        for ii, p in enumerate(self.psf):
            psf_on_subpixel = p.data_format is not None and p.data_format.lower() == 'subpixel'
            if not psf_on_subpixel:
                psf_on_raw = p.data_format is not None and p.data_format.lower() == 'raw'
                if psf_on_raw:
                    # handle 2D flattening
                    y = np.transpose(y, (0, 3, 1, 4, 2))
                    img_size_det = y.shape
                    y = np.reshape(y, np.concatenate(((-1, ), self.photo_size_2D)))

                y = p.apply_psf_adjoint(y)

                if psf_on_raw:
                    y = np.reshape(y, img_size_det)
                    y = np.transpose(y, (0, 2, 4, 1, 3))

                if self.mask_psf_renorm[ii] is not None:
                    y *= self.mask_psf_renorm[ii]

        if self.vignetting_int is not None:
            y *= self.vignetting_int

        proj_stack_shape = np.concatenate(((-1, self.camera.get_number_of_subaperture_images()), self.camera.data_size_ts))
        y = np.reshape(y, proj_stack_shape)
        y = np.transpose(y, (0, 2, 1, 3))

        if self.up_sampling > 1:
            y = np.reshape(y, (-1, self.img_size[-2], 1, self.camera.get_number_of_subaperture_images(), self.img_size[-1], 1) )
            y = np.tile(y, [1, 1, self.up_sampling, 1, 1, self.up_sampling])
            y = np.reshape(y, (-1, self.img_size_us[-2], self.camera.get_number_of_subaperture_images(), self.img_size_us[-1]))

        for p in self.psf:
            psf_on_subpixel = p.data_format is not None and p.data_format.lower() == 'subpixel'
            if psf_on_subpixel:
                y = p.apply_psf_direct(y)

        vol_geom_size = (len(self.Ws), self.vol_size[0], self.vol_size[1])

        if self.mode == 'range':
            x = self.Ws[0].BP(np.squeeze(y))
        elif self.mode == 'simultaneous':
            x = np.empty(vol_geom_size, y.dtype)
            for ii, W in enumerate(self.Ws):
                x[ii, :, :] = W.BP(np.squeeze(y))
        else:
            x = np.empty(vol_geom_size, y.dtype)
            for ii, W in enumerate(self.Ws):
                x[ii, :, :] = W.BP(y[ii, :, :, :])

        return x


def compute_forwardprojection(
        camera : lightfield.Camera, zs, vols, masks, reflective_geom=True,
        up_sampling=1, super_sampling=1, border=5, border_padding='edge',
        gpu_index=-1):

    print("Creating projected lightfield..", end='', flush=True)
    c_in = tm.time()

    lf = lightfield.Lightfield(camera, mode='sub-aperture')

    if reflective_geom:
        for ii, z in enumerate(zs):
            dist = np.array((z, ))
            temp_vol = np.expand_dims(vols[ii, :, :], 0)
            temp_mask = np.expand_dims(masks[ii, :, :], 0)
            with Projector(camera, dist, mode='simultaneous', \
                              up_sampling=up_sampling, \
                              super_sampling=super_sampling, \
                              gpu_index=gpu_index) as p:
                pvol = p.FP(temp_vol)
                pmask = p.FP(temp_mask)
                lf.data *= np.squeeze(1 - pmask)
                lf.data += np.squeeze(pvol)
    else:
        with Projector(camera, zs, mode='simultaneous', \
                          up_sampling=up_sampling, \
                          super_sampling=super_sampling, \
                          gpu_index=gpu_index) as p:
            lf.data = p.FP(vols)

    c_out = tm.time()
    print("\b\b: Done in %g seconds." % (c_out - c_in))

    # Return the stack of refocused images:
    return lf

def compute_refocus_backprojection(
        lf : lightfield.Lightfield, zs, border=4, border_padding='edge',
        up_sampling=1, super_sampling=1,
        beam_geometry='cone', domain='object', gpu_index=-1):
    """Compute refocusing of the input lightfield image at the input distances by
    applying the backprojection method.

    :param lf: The light-field object (lightfield.Lightfield)
    :param zs: Refocusing distances (numpy.array_like)
    :param up_sampling: Integer greater than 1 for up-sampling of the final images (int, default: 1)
    :param super_sampling: Super-sampling of the back-projection operator
        (it will not increase refocused image size/resolution) (int, default: 1)
    :param border: Number of pixels to extend the border and reduce darkening of edges (int, default: 4)
    :param border_padding: Border padding method (string, default: 'edge')
    :param beam_geometry: Beam geometry. Possible options: 'parallel' | 'cone' (string, default: 'parallel')
    :param domain: Refocusing domain. Possible options: 'object' | 'image' (string, default: 'object')

    :returns: Stack of 2D refocused images.
    :rtype: numpy.array_like
    """

    print("Refocusing through Backprojection..", end='', flush=True)
    c_in = tm.time()

    lf_sa = lf.clone()
    lf_sa.set_mode_subaperture()

    if lf_sa.mask is not None:
        lf_sa.data *= lf_sa.mask
        if lf_sa.flat is not None:
            lf_sa.flat *= lf_sa.mask

    paddings_ts = np.array((border, border))
    lf_sa.pad((0, 0, paddings_ts[0], paddings_ts[1]), method=border_padding)

    with Projector(
            lf_sa.camera, zs, mask=lf_sa.mask, mode='range',
            beam_geometry=beam_geometry, domain=domain,
            up_sampling=up_sampling,
            super_sampling=super_sampling, shifts_vu=lf_sa.shifts_vu,
            gpu_index=gpu_index) as p:
        imgs = p.BP(lf_sa.data)
        ones = p.BP(np.ones_like(lf_sa.data))
        imgs /= ones

    # Crop the refocused images:
    paddings_ts = paddings_ts * up_sampling
    imgs = imgs[:, paddings_ts[0]:-paddings_ts[0], paddings_ts[1]:-paddings_ts[1]]

    c_out = tm.time()
    print("\b\b: Done in %g seconds." % (c_out - c_in))

    # Return the stack of refocused images:
    return imgs


def _get_paddings(data_size_ts, border, up_sampling, algorithm):
    paddings_ts_upper = np.array((border, border))
    paddings_ts_lower = np.array((border, border))
    final_size_ts = (data_size_ts + paddings_ts_lower + paddings_ts_upper) * up_sampling

    if isinstance(algorithm, str) and algorithm.lower() == 'cp_wl':
        padding_align = 8
    elif isinstance(algorithm, solvers.CP_wl):
        padding_align = 2 ** algorithm.decomp_lvl
    else:
        padding_align = 0

    if padding_align > 0:
        additional_padding_ts = ((padding_align - (final_size_ts % padding_align)) % padding_align) / up_sampling
        paddings_ts_lower += np.ceil(additional_padding_ts / 2).astype(np.int)
        paddings_ts_upper += np.floor(additional_padding_ts / 2).astype(np.int)

    return (paddings_ts_lower, paddings_ts_upper)


def compute_refocus_iterative(
        lf : lightfield.Lightfield, zs, iterations=10, algorithm='sirt',
        up_sampling=1, super_sampling=1,
        border_padding='edge', beam_geometry='cone', domain='object',
        psf=None, border=4, gpu_index=-1, verbose=False, chunks_zs=1):
    """Compute refocusing of the input lightfield image at the input distances by
    applying iterative methods.

    :param lf: The light-field object (lightfield.Lightfield)
    :param zs: Refocusing distances (numpy.array_like)
    :param algorithm: Algorithm to use (string, default: 'sirt')
    :param iterations: Number of iterations (int, default: 10)
    :param psf: Detector PSF (psf.PSFApply, default: None)
    :param up_sampling: Integer greater than 1 for up-sampling of the final images (int, default: 1)
    :param super_sampling: Super-sampling of the back-projection operator
        (it will not increase refocused image size/resolution) (int, default: 1)
    :param border: Number of pixels to extend the border and reduce darkening of edges (int, default: 4)
    :param border_padding: Border padding method (string, default: 'edge')
    :param beam_geometry: Beam geometry. Possible options: 'parallel' | 'cone' (string, default: 'parallel')
    :param domain: Refocusing domain. Possible options: 'object' | 'image' (string, default: 'object')

    :returns: Stack of 2D refocused images.
    :rtype: numpy.array_like
    """

    num_dists = len(zs)
    print("Refocusing through %s of %d distances:" % (algorithm.upper(), num_dists))
    c_in = tm.time()

    lf_sa = lf.clone()
    lf_sa.set_mode_subaperture()

    if lf_sa.mask is not None:
        lf_sa.data *= lf_sa.mask
        if lf_sa.flat is not None:
            lf_sa.flat *= lf_sa.mask

    (paddings_ts_lower, paddings_ts_upper) = _get_paddings(
            lf_sa.camera.data_size_ts, border, up_sampling, algorithm)
    padding_vuts = (
            (0, 0), (0, 0),
            (paddings_ts_lower[0], paddings_ts_upper[0]),
            (paddings_ts_lower[1], paddings_ts_upper[1]))
    lf_sa.pad(padding_vuts, method=border_padding)

    imgs_shape = (
            num_dists, lf_sa.camera.data_size_ts[0] * up_sampling,
            lf_sa.camera.data_size_ts[1] * up_sampling )
    imgs = np.empty(imgs_shape, dtype=lf_sa.data.dtype)

    c_init = tm.time()

    print(" * Init: %g seconds" % (c_init - c_in))
    for ii_z in range(0, num_dists, chunks_zs):
        print(" * Refocusing chunk of %03d-%03d (avg: %g seconds)" % (ii_z, ii_z+chunks_zs-1, (tm.time() - c_init) / np.fmax(ii_z, 1)))

        sel_zs = zs[ii_z:ii_z+chunks_zs]
        with Projector(
                lf_sa.camera, sel_zs, mask=lf_sa.mask, psf_d=psf,
                shifts_vu=lf_sa.shifts_vu, mode='independent',
                up_sampling=up_sampling,
                super_sampling=super_sampling, gpu_index=gpu_index,
                beam_geometry=beam_geometry, domain=domain) as p:
            A = lambda x: p.FP(x)
            At = lambda y: p.BP(y)
            b = np.tile(np.reshape(lf_sa.data, np.concatenate(((1, ), p.img_size))), (len(sel_zs), 1, 1, 1, 1))

            if isinstance(algorithm, solvers.Solver):
                algo = algorithm
            elif algorithm.lower() == 'cp_ls':
                algo = solvers.CP_uc(verbose=verbose)
            elif algorithm.lower() == 'cp_tv':
                algo = solvers.CP_tv(verbose=verbose, axes=(-1, -2), lambda_tv=1e-1)
            elif algorithm.lower() == 'cp_wl':
                algo = solvers.CP_wl(verbose=verbose, axes=(-1, -2), lambda_wl=1e-1, wl_type='db1', decomp_lvl=3)
            elif algorithm.lower() == 'bpj':
                algo = solvers.BPJ(verbose=verbose)
            elif algorithm.lower() == 'sirt':
                algo = solvers.Sirt(verbose=verbose)
            else:
                raise ValueError('Unknown algorithm: %s' % algorithm.lower())

            imgs[ii_z:ii_z+len(sel_zs), ...], rel_res_norms = algo(A, b, iterations, At=At, lower_limit=0)

    # Crop the refocused images:
    paddings_ts_lower = paddings_ts_lower * up_sampling
    paddings_ts_upper = paddings_ts_upper * up_sampling
    imgs = imgs[:, paddings_ts_lower[0]:-paddings_ts_upper[0], paddings_ts_lower[1]:-paddings_ts_upper[1]]

    c_out = tm.time()
    print(" * Done in %g seconds." % (c_out - c_in))

    # Return the stack of refocused images:
    return imgs

def compute_refocus_iterative_multiple(
        lf : lightfield.Lightfield, zs, iterations=10, algorithm='sirt',
        up_sampling=1, super_sampling=1, border=4, border_padding='edge',
        beam_geometry='cone', domain='object', psf=None, gpu_index=-1,
        verbose=False):
    """Compute refocusing of the input lightfield image simultaneously at the input
    distances by applying iterative methods.

    :param lf: The light-field object (lightfield.Lightfield)
    :param zs: Refocusing distances (numpy.array_like)
    :param algorithm: Algorithm to use (string, default: 'sirt')
    :param iterations: Number of iterations (int, default: 10)
    :param psf: Detector PSF (psf.PSFApply, default: None)
    :param up_sampling: Integer greater than 1 for up-sampling of the final images (int, default: 1)
    :param super_sampling: Super-sampling of the back-projection operator
        (it will not increase refocused image size/resolution) (int, default: 1)
    :param border: Number of pixels to extend the border and reduce darkening of edges (int, default: 4)
    :param border_padding: Border padding method (string, default: 'edge')
    :param beam_geometry: Beam geometry. Possible options: 'parallel' | 'cone' (string, default: 'parallel')
    :param domain: Refocusing domain. Possible options: 'object' | 'image' (string, default: 'object')

    :returns: Stack of 2D refocused images.
    :rtype: numpy.array_like
    """

    print("Simultaneous refocusing through %s:" % algorithm.upper())
    c_in = tm.time()

    lf_sa = lf.clone()
    lf_sa.set_mode_subaperture()

    if lf_sa.mask is not None:
        lf_sa.data *= lf_sa.mask
        if lf_sa.flat is not None:
            lf_sa.flat *= lf_sa.mask

    (paddings_ts_lower, paddings_ts_upper) = _get_paddings(
            lf_sa.camera.data_size_ts, border, up_sampling, algorithm)
    padding_vuts = (
            (0, 0), (0, 0),
            (paddings_ts_lower[0], paddings_ts_upper[0]),
            (paddings_ts_lower[1], paddings_ts_upper[1]))
    lf_sa.pad(padding_vuts, method=border_padding)

    with Projector(
            lf_sa.camera, zs, mask=lf_sa.mask, psf_d=psf,
            shifts_vu=lf_sa.shifts_vu, mode='simultaneous',
            up_sampling=up_sampling, super_sampling=super_sampling,
            gpu_index=gpu_index, beam_shape=beam_geometry, domain=domain) as p:
        A = lambda x: p.FP(x)
        At = lambda y: p.BP(y)
        b = np.reshape(lf_sa.data, np.concatenate(((1, ), p.img_size)))

        if isinstance(algorithm, solvers.Solver):
            algo = algorithm
        elif algorithm.lower() == 'cp_ls':
            algo = solvers.CP_uc(verbose=verbose)
        elif algorithm.lower() == 'cp_tv':
            algo = solvers.CP_tv(verbose=verbose, axes=(-1, -2), lambda_tv=1e-1)
        elif algorithm.lower() == 'cp_wl':
            algo = solvers.CP_wl(verbose=verbose, axes=(-1, -2), lambda_wl=1e-1, wl_type='db1', decomp_lvl=3)
        elif algorithm.lower() == 'bpj':
            algo = solvers.BPJ(verbose=verbose)
        elif algorithm.lower() == 'sirt':
            algo = solvers.Sirt(verbose=verbose)
        else:
            raise ValueError('Unknown algorithm: %s' % algorithm.lower())

        imgs, rel_res_norms = algo(A, b, iterations, At=At, lower_limit=0)

    # Crop the refocused images:
    paddings_ts_lower = paddings_ts_lower * up_sampling
    paddings_ts_upper = paddings_ts_upper * up_sampling
    imgs = imgs[:, paddings_ts_lower[0]:-paddings_ts_upper[0], paddings_ts_lower[1]:-paddings_ts_upper[1]]

    c_out = tm.time()
    print("Done in %g seconds." % (c_out - c_in))

    # Return the stack of refocused images:
    return imgs
