#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 12:20:48 2019

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

import numpy as np
import matplotlib.image as mim

from plenoptomos import solvers
from . import geometry

import astra

import os
import glob
import configparser


def load_images(dset_path, imprefix):
    files_path = os.path.join(dset_path, imprefix + "*")
    files = sorted(glob.glob(files_path))
    if len(files) == 0:
        raise ValueError('No files found for pattern "%s", in dir: %s' % (imprefix, dset_path))

    first_im = mim.imread(files[0])
    output = np.empty((len(files), *first_im.shape), dtype=first_im.dtype)
    output[0, ...] = first_im
    for ii_im in range(1, len(files)):
        output[ii_im, ...] = mim.imread(files[ii_im])
    return output


def correct_projs(projs, flats, darks, crop=None, pad=None):
    if crop is not None:
        projs = projs[..., crop[0] : crop[2], crop[1] : crop[3]]
        darks = darks[..., crop[0] : crop[2], crop[1] : crop[3]]
        flats = flats[..., crop[0] : crop[2], crop[1] : crop[3]]

    projs -= darks
    flats -= darks

    flats = np.mean(flats.astype(np.float32), axis=0)

    projs = projs.astype(np.float32) / flats
    projs = -np.log(projs)
    projs = np.fmax(projs, 0.0)
    if pad is not None:
        projs = np.pad(projs, ((0, 0), (pad[0], pad[2]), (pad[1], pad[3])), mode="constant")
    projs = np.ascontiguousarray(np.transpose(projs, [1, 0, 2]))

    return projs


def get_scanner_conf(dset_path):
    ini_file = os.path.join(dset_path, "data settings XRE.txt")
    config = configparser.ConfigParser()
    config.read(ini_file)
    return config


def get_geometry(dset_path, projs_shape, beam_geometry="cone", vol_shift_mm=0.0, vol_size_add=None, detector_tilt_deg=None):
    config = get_scanner_conf(dset_path)

    sdd = float(config["CT-parameters IN"]["SDD"][1:-1])
    sod = float(config["CT-parameters IN"]["SOD"][1:-1])
    first_angle = float(config["CT-parameters IN"]["Start angle"][1:-1])
    last_angle = float(config["CT-parameters IN"]["Last angle"][1:-1])
    voxsize = float(config["CT-parameters IN"]["Voxel size"][1:-1])
    pixsize = float(config["CT-parameters IN"]["Pixel size"][1:-1])

    angles = np.linspace(first_angle, last_angle, projs_shape[1])
    angles = angles / 180 * np.pi

    if beam_geometry.lower() == "cone":
        sdd = sdd / voxsize
        sod = sod / voxsize
        odd = sdd - sod
        rel_pixsize = pixsize / voxsize
        vol_shift = -vol_shift_mm / pixsize

        proj_geom = astra.create_proj_geom("cone", rel_pixsize, rel_pixsize, projs_shape[0], projs_shape[2], angles, sod, odd)
        proj_geom = astra.geom_2vec(proj_geom)
        V = proj_geom["Vectors"]
        V[:, 3:6] = V[:, 3:6] + vol_shift * V[:, 6:9]
        V[:, 0:3] = V[:, 0:3] + vol_shift * V[:, 6:9]
    elif beam_geometry.lower() == "parallel":
        vol_shift = -vol_shift_mm / voxsize

        proj_geom = astra.create_proj_geom("parallel3d", 1, 1, projs_shape[0], projs_shape[2], angles)
        proj_geom = astra.geom_postalignment(proj_geom, [vol_shift, 0])

    if detector_tilt_deg is not None:
        if proj_geom["type"] in ("cone", "parallel"):
            proj_geom = astra.geom_2vec(proj_geom)
        V = proj_geom["Vectors"]
        dir_proj = V[:, 3:6] - V[:, 0:3]
        dir_proj = dir_proj / np.sqrt(np.sum(dir_proj ** 2, axis=1))[..., np.newaxis]
        for ii in range(V.shape[0]):
            t = geometry.GeometryTransformation.get_rototranslation(dir_proj[ii, :], detector_tilt_deg)
            V[ii, 6:9] = np.transpose(t.apply_direction(np.transpose(V[ii, 6:9])))
            V[ii, 9:12] = np.transpose(t.apply_direction(np.transpose(V[ii, 9:12])))

    vol_size = np.array((projs_shape[2], projs_shape[2], projs_shape[0]), dtype=np.int)
    if vol_size_add is not None:
        vol_size = (vol_size + vol_size_add).astype(np.int)
    vol_geom = astra.create_vol_geom(vol_size)

    return (proj_geom, vol_geom, vol_shift)


class Reconstruct3D(object):
    def __init__(self, vol_geom, proj_geom, super_sampling=1, gpu_index=None):
        self.vol_geom = vol_geom
        self.proj_geom = proj_geom

        self.super_sampling = super_sampling
        self.gpu_index = gpu_index

        self.projector_id = None
        self.vol_id = None
        self.data_id = None
        self.algo_id = None

    def __enter__(self):
        self._initialize_projector()
        return self

    def __exit__(self, *args):
        self._reset()

    def _initialize_projector(self):
        if self.projector_id is None:
            opts = {"VoxelSuperSampling": self.super_sampling, "DetectorSuperSampling": self.super_sampling}
            self.projector_id = astra.create_projector("cuda3d", self.proj_geom, self.vol_geom, options=opts)

    def _reset(self):
        if self.projector_id is not None:
            astra.projector.delete(self.projector_id)
            self.projector_id = None
        if self.algo_id is not None:
            astra.algorithm.delete(self.algo_id)
            self.algo_id = None
        if self.vol_id is not None:
            astra.data3d.delete(self.vol_id)
            self.vol_id = None
        if self.data_id is not None:
            astra.data3d.delete(self.data_id)
            self.data_id = None

    def fp(self, vol):
        if self.vol_id is None:
            self.vol_id = astra.data3d.create("-vol", self.vol_geom, vol)
        else:
            astra.data3d.store(self.vol_id, vol)

        if self.data_id is None:
            self.data_id = astra.data3d.create("-sino", self.proj_geom, 0)
        else:
            astra.data3d.store(self.data_id, 0)

        cfg = astra.astra_dict("FP3D_CUDA")
        cfg["ProjectorId"] = self.projector_id
        if self.gpu_index is not None:
            cfg["option"] = {"GPUindex": self.gpu_index}
        cfg["ProjectionDataId"] = self.data_id
        cfg["VolumeDataId"] = self.vol_id

        self.algo_id = astra.algorithm.create(cfg)
        astra.algorithm.run(self.algo_id)
        astra.algorithm.delete(self.algo_id)
        self.algo_id = None

        return astra.data3d.get(self.data_id)

    def bp(self, data):
        if self.data_id is None:
            self.data_id = astra.data3d.create("-sino", self.proj_geom, data)
        else:
            astra.data3d.store(self.data_id, data)

        if self.vol_id is None:
            self.vol_id = astra.data3d.create("-vol", self.vol_geom, 0)
        else:
            astra.data3d.store(self.vol_id, 0)

        cfg = astra.astra_dict("BP3D_CUDA")
        cfg["ProjectorId"] = self.projector_id
        if self.gpu_index is not None:
            cfg["option"] = {"GPUindex": self.gpu_index}
        cfg["ProjectionDataId"] = self.data_id
        cfg["ReconstructionDataId"] = self.vol_id

        self.algo_id = astra.algorithm.create(cfg)
        astra.algorithm.run(self.algo_id)
        astra.algorithm.delete(self.algo_id)
        self.algo_id = None

        return astra.data3d.get(self.vol_id)

    def cp(self, data, iterations=100, pars={}):
        if "data_term" not in pars:
            pars["data_term"] = "l2"
        if "lower_limit" not in pars:
            pars["lower_limit"] = None
        if "upper_limit" not in pars:
            pars["upper_limit"] = None

        if "lambda_wl" in pars:
            sol = solvers.CP_wl(lambda_wl=pars["lambda_wl"], data_term=pars["data_term"], verbose=True)
        elif "lambda_tv" in pars:
            sol = solvers.CP_tv(lambda_tv=pars["lambda_tv"], data_term=pars["data_term"], verbose=True)
        else:
            sol = solvers.CP_uc(data_term=pars["data_term"], verbose=True)

        (x, _) = sol(self.fp, data, iterations, At=self.bp, lower_limit=pars["lower_limit"], upper_limit=pars["upper_limit"])
        return x

    def sirt(self, data, iterations=25, pars={}):
        if "lower_limit" not in pars:
            pars["lower_limit"] = None
        if "upper_limit" not in pars:
            pars["upper_limit"] = None

        sol = solvers.Sirt(verbose=True)

        (x, _) = sol(self.fp, data, iterations, At=self.bp, lower_limit=pars["lower_limit"], upper_limit=pars["upper_limit"])
        return x

    def reconstruct(self, data, algo="SIRT3D_CUDA", iterations=25, pars={}):
        if self.data_id is None:
            self.data_id = astra.data3d.create("-sino", self.proj_geom, data)
        else:
            astra.data3d.store(self.data_id, data)

        if self.vol_id is None:
            self.vol_id = astra.data3d.create("-vol", self.vol_geom, 0)
        else:
            astra.data3d.store(self.vol_id, 0)

        cfg = astra.astra_dict(algo)
        cfg["ProjectorId"] = self.projector_id
        cfg["ProjectionDataId"] = self.data_id
        cfg["ReconstructionDataId"] = self.vol_id
        if self.gpu_index is not None:
            cfg["options"] = {"GPUindex": self.gpu_index}
        else:
            cfg["options"] = {}
        if "lower_limit" in pars:
            cfg["options"]["UseMinConstraint"] = True
            cfg["options"]["MinConstraintValue"] = pars["lower_limit"]
        if "upper_limit" in pars:
            cfg["options"]["UseMaxConstraint"] = True
            cfg["options"]["MaxConstraintValue"] = pars["upper_limit"]

        self.algo_id = astra.algorithm.create(cfg)
        astra.algorithm.run(self.algo_id, iterations)
        astra.algorithm.delete(self.algo_id)
        self.algo_id = None

        return astra.data3d.get(self.vol_id)
