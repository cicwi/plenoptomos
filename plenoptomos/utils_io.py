#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convenience functions for handling IO with hdf5 files of frequently used data
structures.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France

Created on Wed Mar  1 16:09:53 2017
"""

import numpy as np

try:
    import imageio as iio
except ImportError as ex:
    print('WARNING: error importing Imageio, using matplotlib instead')
    print('Error message:\n', ex)
    import matplotlib.image as iio

from . import lightfield

import h5py


def save_refocused_image(img2d, filename, ind=None):
    if ind is not None:
        iio.imsave(filename, img2d[ind, ...], vmin=0.0, vmax=1.0)
    else:
        iio.imsave(filename, img2d, vmin=0.0, vmax=1.0)


def save_field_toh5(filename, dset_name, data, verbose=False, append=False, compression_lvl=7, to_uint8=False):
    if append:
        mode = 'a'
    else:
        mode = 'w'
    if to_uint8:
        data = ((data - np.min(data)) / (np.max(data) - np.min(data)) * 255).astype(np.uint8)
    with h5py.File(filename, mode) as f:
        if compression_lvl is not None:
            f.create_dataset(dset_name, data=data, compression='gzip', compression_opts=compression_lvl)
        else:
            f.create_dataset(dset_name, data=data)
    if verbose:
        print('Saved dataset "%s" to file: %s' % (dset_name, filename))


def load_field_fromh5(filename, dset, verbose=False):
    mode = 'r'
    with h5py.File(filename, mode) as f:
        data = f[dset][()]
    if verbose:
        print('Loaded dataset "%s" from file: %s' % (dset, filename))
    return data


def save_refocused_stack(refocus_stack, filename, verbose=False, zs=None):
    save_field_toh5(filename, 'refocus_stack', refocus_stack, verbose=verbose)
    if zs is not None:
        save_field_toh5(filename, 'zs', zs, append=True)


def save_lightfield(filename, lf: lightfield.Lightfield):
    with h5py.File(filename, 'w') as f:
        c = lf.camera.__dict__
        for k, v in c.items():
            f['camera/%s' % k] = v
        f['mode'] = lf.mode
        f.create_dataset('data', data=lf.data, compression="gzip")
        if lf.flat is not None:
            f.create_dataset('flat', data=lf.flat, compression="gzip")
        if lf.mask is not None:
            f.create_dataset('mask', data=lf.flat, compression="gzip")
        if lf.shifts_vu is not None and isinstance(lf.shifts_vu, (tuple, list)) \
                and lf.shifts_vu[0] is not None and lf.shifts_vu[1] is not None:
            f.create_dataset('shifts_vu', data=lf.shifts_vu, compression="gzip")


def load_lightfield(filename):
    with h5py.File(filename, 'r') as f:
        camera = lightfield.Camera()
        c = camera.__dict__
        for k in c:
            setattr(camera, k, f['camera/%s' % k][()])
        lf = lightfield.Lightfield(camera_type=camera)
        lf.mode = f['mode'][()]
        lf.data = f['data'][()]
        if '/flat' in f:
            lf.flat = f['flat'][()]
        if '/mask' in f:
            lf.mask = f['mask'][()]
        if '/shifts_vu' in f:
            lf.shifts_vu = f['shifts_vu'][()]
        return lf
