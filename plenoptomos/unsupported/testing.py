#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 16:51:41 2018

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import copy
import os

import time as tm

from plenoptomos import refocus, tomo, colors, solvers, utils_io, psf


class Tester(object):

    default_rec_params = dict(
        iterations=50,
        up_sampling_st=1,
        border=8,
        psf=None,
        fourier_method="slice",
        up_sampling_function="flat",
        data_term="l2",
    )
    default_test_params = dict(zs=[], domain=[], beam=[], algo=[], lambda_reg=[])
    default_conditions = dict(down_sampling_st=1, down_sampling_uv=1, extras="")

    def __init__(self, test_name, lfs, test_params={}, rec_params={}, test_conds={}, is_rgb=False, result_dir=None, save=True):
        self.test_name = test_name
        self.lfs = lfs
        self.is_rgb = is_rgb
        self.result_dir = result_dir
        if self.result_dir is None:
            self.result_dir = os.path.join(".", "results")
        self.save = save

        self.test_params = copy.deepcopy(Tester.default_test_params)
        self.test_params.update(test_params)
        self.rec_params = copy.deepcopy(Tester.default_rec_params)
        self.rec_params.update(rec_params)
        self.test_conditions = copy.deepcopy(Tester.default_conditions)
        self.test_conditions.update(test_conds)

        self.results = []

    def get_options_number(self):
        def get_num_option(self, option_name):
            option = self.test_params[option_name]
            if isinstance(option, (list, tuple)):
                return len(option)
            elif isinstance(option, (str, np.ndarray)):
                return 1

        option_nums = np.array([get_num_option(self, x) for x in self.test_params.keys()])
        more_than_ones = option_nums[option_nums > 1]
        if more_than_ones.size > 1:
            if np.all(more_than_ones[1:] == more_than_ones[0]):
                return more_than_ones[0]
            else:
                raise ValueError("Wrong set op options (Non matching numbers!)")
        elif more_than_ones.size == 1:
            return more_than_ones[0]
        else:
            return 1

    def get_test_option(self, option_name, ind, struct=None):
        if struct is None:
            struct = self.test_params
        test_option = struct[option_name]
        if isinstance(test_option, (list, tuple)):
            if len(test_option) == 0:
                return None
            elif len(test_option) == 1:
                return test_option[0]
            else:
                return test_option[ind]
        elif isinstance(test_option, (str, np.ndarray, int, float)) or test_option is None:
            return test_option
        else:
            raise ValueError(("What the **** is :", option_name))

    def launch_reconstruction(self, ii_t):
        td = self.get_test_option("domain", ii_t)
        tb = self.get_test_option("beam", ii_t)
        tz = self.get_test_option("zs", ii_t)
        ta = self.get_test_option("algo", ii_t)

        up_sampling_st = self.rec_params["up_sampling_st"]
        border = self.rec_params["border"]
        iterations = self.rec_params["iterations"]
        fourier_method = self.rec_params["fourier_method"]
        data_term = self.get_test_option("data_term", ii_t, struct=self.rec_params)
        up_sampling_function = self.get_test_option("up_sampling_function", ii_t, struct=self.rec_params)

        psf = self.rec_params["psf"]
        if psf is None or len(psf) == 0 and self.is_rgb:
            psf = (None, None, None)

        lambda_reg = self.get_test_option("lambda_reg", ii_t)

        if ta.lower() == "int":
            func = lambda x, _: refocus.compute_refocus_integration(
                x, tz, up_sampling=up_sampling_st, border=border, beam_geometry=tb, domain=td
            )
        elif ta.lower() == "fou":
            func = lambda x, _: refocus.compute_refocus_fourier(
                x,
                tz,
                method=fourier_method,
                padding_factor=1.125,
                up_sampling=up_sampling_st,
                border=border,
                beam_geometry=tb,
                domain=td,
            )
        elif ta.lower() == "bpj":
            func = lambda x, _: tomo.compute_refocus_backprojection(
                x, tz, up_sampling=up_sampling_st, border=border, beam_geometry=tb, domain=td
            )
        #            algo = solvers.BPJ(verbose=True)
        #            func = lambda x, _ : tomo.compute_refocus_iterative(
        #                    x, tz, algorithm=algo, up_sampling=up_sampling_st, border=border, beam_geometry=tb, domain=td)
        else:
            if ta[:3].lower() == "itr":
                if ta.lower() == "itr":
                    algo = solvers.CP_uc(verbose=True, data_term=data_term)
                elif ta.lower() == "itr_tv":
                    algo = solvers.CP_tv(verbose=True, lambda_tv=lambda_reg, axes=(-2, -1), data_term=data_term)
                elif ta.lower() == "itr_wl":
                    algo = solvers.CP_wl(
                        verbose=True, lambda_wl=lambda_reg, axes=(-2, -1), data_term=data_term, wl_type="db1", decomp_lvl=3
                    )
                elif ta.lower() == "itr_d2":
                    algo = solvers.CP_smooth(verbose=True, lambda_d2=lambda_reg, axes=(-2, -1), data_term=data_term)

                func = lambda x, _: tomo.compute_refocus_iterative(
                    x,
                    tz,
                    algorithm=algo,
                    iterations=iterations,
                    up_sampling=up_sampling_st,
                    border=border,
                    up_sampling_function=up_sampling_function,
                    beam_geometry=tb,
                    domain=td,
                )
            else:
                if ta.lower() == "itp":
                    algo = solvers.CP_uc(verbose=True, data_term=data_term)
                elif ta.lower() == "itp_tv":
                    algo = solvers.CP_tv(verbose=True, lambda_tv=lambda_reg, axes=(-2, -1), data_term=data_term)
                elif ta.lower() == "itp_wl":
                    algo = solvers.CP_wl(
                        verbose=True, lambda_wl=lambda_reg, axes=(-2, -1), data_term=data_term, wl_type="db1", decomp_lvl=3
                    )
                elif ta.lower() == "itp_d2":
                    algo = solvers.CP_smooth(verbose=True, lambda_d2=lambda_reg, axes=(-2, -1), data_term=data_term)
                func = lambda x, p: tomo.compute_refocus_iterative(
                    x,
                    tz,
                    algorithm=algo,
                    iterations=iterations,
                    up_sampling=up_sampling_st,
                    border=border,
                    up_sampling_function=up_sampling_function,
                    psf=p,
                    beam_geometry=tb,
                    domain=td,
                )

        if self.is_rgb:
            imgs_r = func(self.lfs[0], psf[0])
            imgs_g = func(self.lfs[1], psf[1])
            imgs_b = func(self.lfs[2], psf[2])
            if ta.lower() in ("int", "fou"):
                lf_ones = self.lfs[0].clone()
                lf_ones.data = np.ones_like(lf_ones.data)
                imgs_ones = func(lf_ones, psf[0])
                imgs_r /= imgs_ones
                imgs_g /= imgs_ones
                imgs_b /= imgs_ones

            imgs = colors.merge_rgb_images(imgs_r, imgs_g, imgs_b)
        else:
            imgs = func(self.lfs, psf)

        if self.save:
            if isinstance(self.test_conditions["extras"], (list, tuple)):
                extras_str = "_" + "_".join(self.test_conditions["extras"])
            elif self.test_conditions["extras"] is None or self.test_conditions["extras"] == "":
                extras_str = ""
            else:
                extras_str = "_" + self.test_conditions["extras"]
            if ta.lower() == "fou":
                extras_str = "_".join([extras_str, fourier_method])

            imgs_fname = "%s_algo-%s-%s_domain-%s_beam-%s_us-st%d_ds-st%d-uv%d%s.h5" % (
                self.test_name,
                ta,
                data_term,
                td,
                tb,
                up_sampling_st,
                self.test_conditions["down_sampling_st"],
                self.test_conditions["down_sampling_uv"],
                extras_str,
            )
            if not os.path.exists(self.result_dir):
                os.mkdir(self.result_dir)
            imgs_path = os.path.join(self.result_dir, imgs_fname)
            utils_io.save_refocused_stack(imgs, imgs_path, zs=tz, verbose=True)

        self.results.append(imgs)


class Blurring(object):
    def __init__(self, cameras, verbose=False):
        self.num_psfs_ml = len(cameras)
        self.cameras = cameras
        self.verbose = verbose

        self.psfs = {"vu": [], "ts": [], "defocus": []}
        self.p = {"vu": [], "ts": [], "defocus": []}

    def create_psf_lenses(
        self, coordinates="vu", airy_rings=3, refocus_distance=None, up_sampling=1, plot_psfs=False, use_otf=False
    ):
        make_psf = lambda x: psf.PSF.create_theo_psf(
            x, coordinates=coordinates, airy_rings=airy_rings, refocus_distance=refocus_distance, up_sampling=up_sampling
        )
        self.psfs[coordinates] = [make_psf(c) for c in self.cameras]

        if plot_psfs:
            f = plt.figure()
            for ii in range(self.num_psfs_ml):
                pixels_distance = (self.psfs[coordinates][ii].data.shape[0] - 1) / 2
                grid_p = np.linspace(-pixels_distance, pixels_distance, 2 * pixels_distance + 1)
                [grid_p1, grid_p2] = np.meshgrid(grid_p, grid_p, indexing="ij")

                ax = f.add_subplot(1, self.num_psfs_ml, ii + 1, projection="3d")
                ax.plot_surface(grid_p1, grid_p2, self.psfs[coordinates][ii].data)
                ax.view_init(12, -7.5)
            plt.show()

        if coordinates.lower() == "vu":
            data_format = "raw"
        else:
            data_format = None
        self.p[coordinates] = [
            psf.PSFApply2D(psf_d=p, use_otf=use_otf, data_format=data_format, use_fftconv=True) for p in self.psfs[coordinates]
        ]

    def blur(self, lfs, coordinates="vu"):
        if self.verbose:
            if coordinates.lower() == "vu":
                lens_type = "micro-lens"
            elif coordinates.lower() == "ts":
                lens_type = "main-lens"
        for ii in range(self.num_psfs_ml):
            if self.verbose:
                color = colors.detect_color(self.cameras[ii])
                print("Applying %s PSF for color: %s.." % (lens_type, color.upper()), end="", flush=False)
                c = tm.time()
            if coordinates.lower() == "vu":
                data_raw = lfs[ii].get_raw_detector_picture()
                data_raw = self.p[coordinates][ii].apply_psf_direct(data_raw)
                lfs[ii].set_raw_detector_picture(data_raw=data_raw)
            else:
                lfs[ii].data = self.p[coordinates][ii].apply_psf_direct(lfs[ii].data)
            if self.verbose:
                print("\b\b: Done in %g seconds." % (tm.time() - c))
