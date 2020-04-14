#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 16:08:43 2019

@author: vigano

This module contains the import functions for external light-field data formats.
"""

import numpy as np
import scipy.interpolate as spinterp

from . import lightfield
from . import colors
from . import data_format

import os
import glob
import json
import re
import configparser
import time as tm

import matplotlib.image as mim

import subprocess as sp


def lytro_create_warps(file_path):
    """Utility function to create warp images using Lytro's powertools and the
    corresponding json file

    :param file_path: The path to the .lfp file (string)
    """
    lytro_uv = lambda i: np.round((i % 14.0 + 0.5) / 14.0 - 0.5, 3)

    vv = lytro_uv(np.arange(14))
    uu = lytro_uv(np.arange(14))
    (vv, uu) = np.meshgrid(vv, uu, indexing='ij')
    vv = vv.flatten()
    uu = uu.flatten()

    cmd = ['lfptool', '--verbose', 'raw', '-i', file_path, '--imagerep', 'png', '-u', ]

    for u in uu:
        cmd.append(str(u))
    cmd.append('-v')
    for v in vv:
        cmd.append(str(v))

    print('Calling command:\n', cmd)
    sp.call(cmd)

    cmd = ['lfptool', 'info', '--json-out', '-i', file_path]
    print('Calling command:\n', cmd)
    sp.call(cmd)


def lytro_read_metadata(fname_json):
    """Reads the Lytro ILLUM metadata from the json file.

    :param fname_json: The path to the .json file (string)

    :returns: The camera object
    :rtype: lightfield.Camera
    """
    camera = lightfield.get_camera('lytro_illum')

    with open(fname_json) as f:
        pic_info = json.load(f)
    try:
        pic_info = pic_info[0]
    except KeyError:
        pic_info = pic_info['frames'][0]['frame']['metadata']

    camera.f2 = pic_info['devices']['mla']['sensorOffset']['z'] * 1e3
    camera.f1 = pic_info['devices']['lens']['focalLength'] * 1e3
    camera.pixel_size_yx = np.array((pic_info['devices']['sensor']['pixelPitch'] * 1e3, ) * 2)
    camera.pixel_size_ts = np.array((pic_info['devices']['mla']['lensPitch'] * 1e3, ) * 2)
    camera.z1 = pic_info['devices']['lens']['exitPupilOffset']['z'] * 1e3
    camera.aperture_f1 = pic_info['devices']['lens']['fNumber']
    camera.aperture_f2 = pic_info['devices']['lens']['fNumber']

    renorm_factor_lytro = (2 * np.abs(0.5 / camera.data_size_vu - 0.5))
    camera.pixel_size_vu = (camera.f1 / camera.aperture_f2) * renorm_factor_lytro / (camera.data_size_vu - 1)
    return camera


def _compute_shape_multiple(shape_in, power_of_2=3, binning=1):
    shape_align = binning * (2 ** power_of_2)
    border = np.mod(shape_align - np.mod(shape_in, shape_align), shape_align)
    extra_cols = np.array([np.floor(border[0] / 2), np.ceil(border[0] / 2)], dtype=np.int)
    extra_rows = np.array([np.floor(border[1] / 2), np.ceil(border[1] / 2)], dtype=np.int)
    shape_out = (np.array(shape_in + border) / binning).astype(np.int)
    shape_bin = np.array([shape_out[0], binning, shape_out[1], binning, -1], dtype=np.int)
    return (shape_out, shape_bin, extra_cols, extra_rows)


def from_lytro(data_path, fname_json, source='warp', mode='grayscale', rgb2gs_mode='luma', binning=1, data_type=np.float32):
    """Imports either the ESLF or warp image light-fields acquired with Lytro cameras.

    :param data_path: Either the path of the ESLF file or the directory of the warp images (string)
    :param fname_json: Path of the metadata public json file (string)
    :param source: Format of the lytro data. Options: {'warp'}, 'eslf' (string)
    :param mode: Switch between grayscale and RGB. Options: {'grayscale'}, 'rgb' (string)
    :param rgb2gs_mode: RGB -> grayscale conversion method. Options: {'luma'}, 'flat' (string)
    :param binning: Binning of the input images, used to reduce their size and resolution (int, default: 1)
    :param data_type: Datatype of the output light-field data (np.dtype)

    :returns: The imported light-field datastructure
    :rtype: lightfield.Lightfield
    """
    print('Initializing metadata..', end='', flush=True)
    c = tm.time()
    camera = lytro_read_metadata(fname_json)
    print('\b\b: Done in %g seconds.' % (tm.time()-c))

    if source.lower() == 'warp':
        camera.data_size_ts = np.array((1404, 2022), dtype=np.intp)

        base_name = glob.glob(os.path.join(data_path, '*.lfp'))
        base_name = os.path.basename(base_name[0])[:-4]

        (camera.data_size_ts, binning_shape, extra_cols, extra_rows) = _compute_shape_multiple(
            camera.data_size_ts, binning=binning)

        lf_img_size = np.concatenate((camera.data_size_vu, camera.data_size_ts, (4, )))
        lf_img = np.empty(lf_img_size, dtype=data_type)

        print('Loading single images: ', end='', flush=True)
        c = tm.time()
        for ii_v in range(camera.data_size_vu[0]):
            for ii_u in range(camera.data_size_vu[1]):
                prnt_str = 'v %02d/%02d, u %02d/%02d' % (ii_v, camera.data_size_vu[0], ii_u, camera.data_size_vu[1])
                print(prnt_str, end='', flush=True)

                filename = os.path.join(data_path, '%s_view000_warp_image%03d_*.png' % (base_name, (ii_v * 14) + ii_u))
                filenames = glob.glob(filename)
                filename = filenames[0]

                img_rgb = mim.imread(filename).astype(data_type)

                img_rgb = np.pad(img_rgb, ((extra_cols[0], extra_cols[1]), (extra_rows[0], extra_rows[1]), (0, 0)), mode='edge')
                img_rgb = np.reshape(img_rgb, binning_shape)
                img_rgb = np.sum(img_rgb, axis=(1, 3)) / (binning ** 2)

                lf_img[ii_v, ii_u, :, :, :] = img_rgb

                print(('\b') * len(prnt_str), end='', flush=True)
        print('Done (%d, %d) = %d in %g seconds.\n' % (
            camera.data_size_vu[0], camera.data_size_vu[1], np.prod(camera.data_size_vu), tm.time()-c))

        lf_img = np.transpose(lf_img, axes=(2, 3, 0, 1, 4))
        lf_mask = None

        print('Dealing with colors (mode=%s)..' % mode, end='', flush=True)
        c = tm.time()
        if mode.lower() == 'grayscale':
            (lf_img, _) = colors.deal_with_channels(lf_img, mode=mode, rgb2gs_mode=rgb2gs_mode)
        elif mode.lower() == 'rgb':
            ((lf_img_r, lf_img_g, lf_img_b), _) = colors.deal_with_channels(lf_img, mode=mode)
        print('\b\b: Done in %g seconds.' % (tm.time()-c))

    elif source.lower() == 'eslf':
        camera.data_size_ts = np.array((375, 541), dtype=np.intp)

        print('Loading ESLF image..', end='', flush=True)
        c = tm.time()
        raw_im2D = mim.imread(data_path)
        print('\b\b: Done in %g seconds.' % (tm.time()-c))

        lenslet_raw_size = np.array(camera.data_size_vu, dtype=np.int)
        array_offsets = np.array([0, 0], dtype=np.int)

        print('Dealing with colors (mode=%s)..' % mode, end='', flush=True)
        c = tm.time()
        (raw_im, raw_flat) = colors.deal_with_channels(raw_im2D, mode=mode, rgb2gs_mode=rgb2gs_mode)
        (lf_mask, lf_shape_tsvu) = data_format.raw_to_microimage_exact(raw_flat, array_offsets, lenslet_raw_size)
        lf_mask = data_format.transform_2D_to_4D(lf_mask, lf_shape_tsvu)
        print('\b\b: Done in %g seconds.' % (tm.time()-c))

        camera.data_size_ts = lf_shape_tsvu[0:2]

        print('Transforming from raw detector to 4D..', end='', flush=True)
        c = tm.time()
        if mode.lower() == 'grayscale':
            (lf_img, _) = data_format.raw_to_microimage_exact(raw_im, array_offsets, lenslet_raw_size)
            lf_img = data_format.transform_2D_to_4D(lf_img, lf_shape_tsvu)
        elif mode.lower() == 'rgb':
            (raw_im_r, raw_im_g, raw_im_b) = raw_im
            (lf_img_r, _) = data_format.raw_to_microimage_exact(raw_im_r, array_offsets, lenslet_raw_size)
            lf_img_r = data_format.transform_2D_to_4D(lf_img_r, lf_shape_tsvu)
            (lf_img_g, _) = data_format.raw_to_microimage_exact(raw_im_g, array_offsets, lenslet_raw_size)
            lf_img_g = data_format.transform_2D_to_4D(lf_img_g, lf_shape_tsvu)
            (lf_img_b, _) = data_format.raw_to_microimage_exact(raw_im_b, array_offsets, lenslet_raw_size)
            lf_img_b = data_format.transform_2D_to_4D(lf_img_b, lf_shape_tsvu)
        print('\b\b: Done in %g seconds.' % (tm.time()-c))

    if mode.lower() == 'grayscale':
        return lightfield.Lightfield(camera_type=camera, data=lf_img, mask=lf_mask)

    elif mode.lower() == 'rgb':
        lf_r = lightfield.Lightfield(camera_type=camera.clone(), data=lf_img_r, mask=lf_mask)
        lf_g = lightfield.Lightfield(camera_type=camera.clone(), data=lf_img_g, mask=lf_mask)
        lf_b = lightfield.Lightfield(camera_type=camera.clone(), data=lf_img_b, mask=lf_mask)

        (lf_r.camera.wavelength_range,
         lf_g.camera.wavelength_range,
         lf_b.camera.wavelength_range) = colors.get_rgb_wavelengths()

        return (lf_r, lf_g, lf_b)


def from_stanford_archive(dataset_path, mode='grayscale', rgb2gs_mode='luma', binning=1, data_type=np.float32):
    """Imports light-fields from the Stanford archive.

    :param dataset_path: Directory of the sub-aperture images (containing a json metadata file) (string)
    :param mode: Switch between grayscale and RGB. Options: {'grayscale'}, 'rgb' (string)
    :param rgb2gs_mode: RGB -> grayscale conversion method. Options: {'luma'}, 'flat' (string)
    :param binning: Binning of the input images, used to reduce their size and resolution (int, default: 1)
    :param data_type: Datatype of the output light-field data (np.dtype)

    :returns: The imported light-field datastructure
    :rtype: lightfield.Lightfield
    """
    print('Initializing metadata..', end='', flush=True)
    c = tm.time()
    camera = lightfield.get_camera('stanford_archive')

    metadata_file = os.path.join(dataset_path, 'metadata.json')
    with open(metadata_file) as f:
        metadata = json.load(f)

    camera.data_size_ts = np.array([metadata['array_size']['t'], metadata['array_size']['s']])
    camera.pixel_size_vu = np.array([metadata['pixel_size_uv']['v'], metadata['pixel_size_uv']['u']], dtype=np.float32)
    camera.aperture_f1 = np.array([metadata['aperture_f1'], ], dtype=np.float32)
    camera.aperture_f2 = np.array([metadata['aperture_f2'], ], dtype=np.float32)
    camera.f1 = np.array([metadata['f1'], ], dtype=np.float32)
    camera.f2 = np.mean(camera.pixel_size_yx / camera.pixel_size_vu) * camera.z1
    camera.pixel_size_ts = camera.pixel_size_yx * camera.data_size_vu
    print('\b\b: Done in %g seconds.' % (tm.time()-c))

    (camera.data_size_ts, binning_shape, extra_cols, extra_rows) = _compute_shape_multiple(camera.data_size_ts, binning=binning)

    lf_img_size = np.concatenate((camera.data_size_vu, camera.data_size_ts, (3, )))
    lf_img = np.empty(lf_img_size, dtype=data_type)

    shifts_v = np.empty((camera.data_size_vu[0], camera.data_size_vu[1]))
    shifts_u = np.empty((camera.data_size_vu[0], camera.data_size_vu[1]))

    print('Loading single images: ', end='', flush=True)
    c = tm.time()
    for ii_v in range(camera.data_size_vu[0]):
        for ii_u in range(camera.data_size_vu[1]):
            prnt_str = 'v %02d/%02d, u %02d/%02d' % (ii_v, camera.data_size_vu[0], ii_u, camera.data_size_vu[1])
            print(prnt_str, end='', flush=True)

            filename = os.path.join(dataset_path, 'out_%02d_%02d_*.png' % (ii_v, ii_u))
            filenames = glob.glob(filename)
            filename = filenames[0]

            pieces = filename.split('_')
            shift_v, shift_u = float(pieces[-2]), float(pieces[-1][:-4])
            shifts_v[ii_v, ii_u] = shift_v
            shifts_u[ii_v, ii_u] = shift_u

            img_rgb = mim.imread(filename).astype(data_type)

            img_rgb = np.pad(img_rgb, ((extra_cols[0], extra_cols[1]), (extra_rows[0], extra_rows[1]), (0, 0)), mode='edge')
            img_rgb = np.reshape(img_rgb, binning_shape)
            img_rgb = np.sum(img_rgb, axis=(1, 3)) / (binning ** 2)

            lf_img[ii_v, ii_u, :, :, :] = img_rgb

            print(('\b') * len(prnt_str), end='', flush=True)
    print('Done (%d, %d) = %d in %g seconds.\n' % (
        camera.data_size_vu[0], camera.data_size_vu[1], np.prod(camera.data_size_vu), tm.time()-c))

    # recentering shifts
    shifts_v -= np.sum(shifts_v) / shifts_v.size
    shifts_u -= np.sum(shifts_u) / shifts_u.size
    spacings_v = np.abs(np.diff(shifts_v, axis=0))
    spacings_u = np.abs(np.diff(shifts_u, axis=1))
    spacing_v = np.sum(spacings_v) / spacings_v.size
    spacing_u = np.sum(spacings_u) / spacings_u.size

    grid_v = np.linspace(-camera.data_size_vu[0]/2, camera.data_size_vu[0]/2, camera.data_size_vu[0]) * spacing_v
    grid_u = np.linspace(-camera.data_size_vu[1]/2, camera.data_size_vu[1]/2, camera.data_size_vu[1]) * spacing_u
    grid_v = np.flip(grid_v, axis=0)
    grid_u = np.flip(grid_u, axis=0)

    (grid_v, grid_u) = np.meshgrid(grid_v, grid_u, indexing='ij')

    shifts_v -= grid_v
    shifts_u -= grid_u

    shifts_v *= (camera.pixel_size_vu[0] / spacing_v)
    shifts_u *= (camera.pixel_size_vu[1] / spacing_u)

    shifts_vu = (shifts_v, shifts_u)

    lf_img = np.transpose(lf_img, axes=(2, 3, 0, 1, 4))

    if mode.lower() == 'grayscale':
        (lf_img, _) = colors.deal_with_channels(lf_img, mode=mode, rgb2gs_mode=rgb2gs_mode)
        return lightfield.Lightfield(camera_type=camera, data=lf_img, shifts_vu=shifts_vu)

    elif mode.lower() == 'rgb':
        ((lf_img_r, lf_img_g, lf_img_b), _) = colors.deal_with_channels(lf_img, mode=mode)

        lf_r = lightfield.Lightfield(camera_type=camera.clone(), data=lf_img_r, shifts_vu=shifts_vu)
        lf_g = lightfield.Lightfield(camera_type=camera.clone(), data=lf_img_g, shifts_vu=shifts_vu)
        lf_b = lightfield.Lightfield(camera_type=camera.clone(), data=lf_img_b, shifts_vu=shifts_vu)

        (lf_r.camera.wavelength_range,
         lf_g.camera.wavelength_range,
         lf_b.camera.wavelength_range) = colors.get_rgb_wavelengths()

        return (lf_r, lf_g, lf_b)


def _flexray_parse_source_det_positions(script_path):
    script_lines = []
    movement_list = []
    # Make sure file gets closed after being iterated
    with open(script_path, 'r') as f:
        for line in f.readlines():
            match = re.split('\t', line)
            script_lines.append(match)

    current_phase = -1
    binnings = []
    rois = []
    for line in script_lines:
        if line[0] == 'camera':
            if line[1] == 'set mode':
                print('creating new phase')
                movement_list.append([])
                current_phase += 1
                hw_bin = int(line[2][2])
                sw_bin = int(line[2][5])
                binnings.append(hw_bin * sw_bin)
            elif line[1] == 'set ROI':
                rois.append(np.array([int(x) for x in line[2].split(' ')]))
        elif current_phase >= 0 and line[1] == 'move absolute':
            movement_list[current_phase].append((line[0], float(line[2])))
    num_phases = current_phase + 1

    tube_grid_size = np.zeros((num_phases, 2), dtype=np.intp)
    det_grid_size = np.zeros((num_phases, 2), dtype=np.intp)
    for ii_p in range(num_phases):
        for mov in movement_list[ii_p]:
            if mov[0] == 'tra_tube':
                tube_grid_size[ii_p, 1] += 1
            elif mov[0] == 'tra_det':
                det_grid_size[ii_p, 1] += 1
            elif mov[0] == 'ver_tube':
                tube_grid_size[ii_p, 0] += 1
            elif mov[0] == 'ver_det':
                det_grid_size[ii_p, 0] += 1

    print('\nFound %d acquisitons phases:' % num_phases)
    for ii_p in range(num_phases):
        print(' %d) Phase has: ' % ii_p, end='', flush=True)
        print('binning %d, ROI [%d, %d, %d, %d], [%d, %d = %d x %d] source positions and [%d, %d] detector positions'
              % (binnings[ii_p], rois[ii_p][0], rois[ii_p][1], rois[ii_p][2], rois[ii_p][3],
                 tube_grid_size[ii_p, 0], tube_grid_size[ii_p, 1],
                 tube_grid_size[ii_p, 1] / tube_grid_size[ii_p, 0], tube_grid_size[ii_p, 0],
                 det_grid_size[ii_p, 0], det_grid_size[ii_p, 1]))

    det_is_fixed = np.any(det_grid_size == 0)

    out_grid_size = np.array((tube_grid_size[-1, 0], tube_grid_size[-1, 1] / tube_grid_size[-1, 0], 2), dtype=np.intp)
    positions_tube = np.empty(out_grid_size, dtype=np.float32)
    positions_det = np.empty(out_grid_size, dtype=np.float32)

    curr_h_pos_tube = 0
    curr_v_pos_tube = -1
    curr_h_pos_det = 0
    curr_v_pos_det = -1

    curr_vert_tube = 0
    curr_vert_det = 0
    for mov in movement_list[-1]:
        if mov[0] == 'tra_tube':
            positions_tube[curr_v_pos_tube, curr_h_pos_tube, :] = [mov[1], curr_vert_tube]
            curr_h_pos_tube += 1
        elif mov[0] == 'tra_det':
            positions_det[curr_v_pos_det, curr_h_pos_det, :] = [mov[1], curr_vert_det]
            curr_h_pos_det += 1
        elif mov[0] == 'ver_tube':
            curr_vert_tube = mov[1]
            curr_v_pos_tube += 1
            curr_h_pos_tube = 0
        elif mov[0] == 'ver_det':
            curr_vert_det = mov[1]
            curr_v_pos_det += 1
            curr_h_pos_det = 0

    return (positions_tube, positions_det, num_phases, np.array(binnings, dtype=np.intp), rois, out_grid_size[:-1], det_is_fixed)


def from_flexray(dset_path, crop_fixed_det=True, data_type=np.float32):
    """Imports light-fields from FleXray scanner data.

    :param dset_path: Directory of the sub-aperture images (containing a .ini file) (string)
    :param crop_fixed_det: Crops the images, in case of fixed detector acquisitions (Boolean, default: True)
    :param data_type: Datatype of the output light-field data (np.dtype)

    :returns: The imported light-field datastructure
    :rtype: lightfield.Lightfield
    """
    print('Initializing metadata..', end='', flush=True)
    c = tm.time()
    ini_file = os.path.join(dset_path, 'data set settings.ini')
    config = configparser.ConfigParser()
    config.read(ini_file)

    script_name = os.path.split(dset_path)
    if script_name[1] == '':
        script_name = os.path.split(script_name[0])
    # script_path = os.path.join(dset_path, '%s.txt' % script_name[1])
    script_path = os.path.join(dset_path, 'script_executed.txt')

    (positions_tube, positions_det, num_phases, binnings, rois, grid_size,
     det_is_fixed) = _flexray_parse_source_det_positions(script_path)

    res_uv = np.abs(np.array((
            positions_tube[0, 1, 0] - positions_tube[0, 0, 0],
            positions_tube[1, 0, 1] - positions_tube[0, 0, 1])))

    camera = lightfield.get_camera('flexray', down_sampling_st=binnings[-1])
    camera.data_size_vu = np.array(grid_size, dtype=np.intp)
    camera.data_size_ts = ((rois[-1][2:] - rois[-1][0:2] + 1) / binnings[-1]).astype(np.intp)
    camera.data_size_ts = np.array([camera.data_size_ts[1], camera.data_size_ts[0]])
    camera.pixel_size_vu = res_uv

    so_dist = float(config['Settings']['source_object_distance'])
    sd_dist = float(config['Settings']['source_detector_distance'])
    camera.z1 = so_dist
    camera.pixel_size_ts *= (so_dist / sd_dist)
    camera.pixel_size_yx = camera.pixel_size_ts / camera.data_size_vu

    camera.f1 = camera.get_f1(so_dist)
    camera.f2 = camera.z1 * np.mean(camera.pixel_size_yx / camera.data_size_vu)
    print('\b\b: Done in %g seconds.' % (tm.time()-c))

    lf_img_size = np.concatenate((camera.data_size_vu, camera.data_size_ts))
    lf_img = np.empty(lf_img_size, dtype=data_type)

    print('Loading single images: ', end='', flush=True)
    c = tm.time()
    for ii_v in range(camera.data_size_vu[0]):
        for ii_u in range(camera.data_size_vu[1]):
            prnt_str = 'v %02d/%02d, u %02d/%02d' % (
                    ii_v, camera.data_size_vu[0], ii_u, camera.data_size_vu[1])
            print(prnt_str, end='', flush=True)

            filename = os.path.join(dset_path, 'scan_%02d_%02d.tif' % (ii_v, ii_u))
            lf_img[ii_v, ii_u, :, :] = mim.imread(filename).astype(data_type)

            print(('\b') * len(prnt_str), end='', flush=True)
    print('Done (%d, %d) = %d in %g seconds.\n' % (
        camera.data_size_vu[0], camera.data_size_vu[1], np.prod(camera.data_size_vu), tm.time()-c))

    print('Loading dark and bright-field..', end='', flush=True)
    if num_phases > 1:
        filename = os.path.join(dset_path, 'dark.tif')
        d0 = mim.imread(filename).astype(data_type)

        i0 = np.empty_like(lf_img)

        flat_fnames = os.path.join(dset_path, 'flat_*.tif')
        num_flats = len(glob.glob(flat_fnames))
        if num_flats == 9:
            mult = (camera.data_size_vu - 1) / 2
            flat_imgs = np.empty(np.concatenate(((3, 3), d0.shape)), dtype=data_type)
            for ii_v in range(3):
                for ii_u in range(3):
                    filename = os.path.join(dset_path, 'flat_%02d_%02d.tif' % (
                            ii_v * mult[0], ii_u * mult[1]))
                    flat_imgs[ii_v, ii_u, ...] = mim.imread(filename)

            # This procedure is less performing (on a theoretical level), but due
            # to the crappy implementation of scipy it saves a ton of memory
            (out_v_points, out_u_points) = (
                np.linspace(0, 2, camera.data_size_vu[0]),
                np.linspace(0, 2, camera.data_size_vu[1]))
            for ii_v in range(out_v_points.size):
                ind_v_low = np.floor(out_v_points[ii_v]).astype(np.intp)
                ind_v_high = (ind_v_low + 1).astype(np.intp)
                coeff_v_low = ind_v_high.astype(np.float32) - out_v_points[ii_v]
                coeff_v_high = 1 - coeff_v_low

                for ii_u in range(out_u_points.size):
                    ind_u_low = np.floor(out_u_points[ii_u]).astype(np.intp)
                    ind_u_high = (ind_u_low + 1).astype(np.intp)
                    coeff_u_low = ind_u_high.astype(np.float32) - out_u_points[ii_u]
                    coeff_u_high = 1 - coeff_u_low

                    i0[ii_v, ii_u, ...] = coeff_v_low * coeff_u_low * flat_imgs[ind_v_low, ind_u_low, ...]
                    if coeff_u_high > np.finfo(np.float32).eps:
                        i0[ii_v, ii_u, ...] += coeff_v_low * coeff_u_high * flat_imgs[ind_v_low, ind_u_high, ...]
                    if coeff_v_high > np.finfo(np.float32).eps:
                        i0[ii_v, ii_u, ...] += coeff_v_high * coeff_u_low * flat_imgs[ind_v_high, ind_u_low, ...]
                    if coeff_v_high > np.finfo(np.float32).eps and coeff_u_high > np.finfo(np.float32).eps:
                        i0[ii_v, ii_u, ...] += coeff_v_high * coeff_u_high * flat_imgs[ind_v_high, ind_u_high, ...]
        else:
            for ii_v in range(camera.data_size_vu[0]):
                for ii_u in range(camera.data_size_vu[1]):
                    filename = os.path.join(dset_path, 'flat_%02d_%02d.tif' % (ii_v, ii_u))
                    i0[ii_v, ii_u, ...] = mim.imread(filename)
    else:
        filename = os.path.join(dset_path, 'di0000.tif')
        d0 = mim.imread(filename).astype(data_type)

        filename = os.path.join(dset_path, 'io0000.tif')
        i0 = mim.imread(filename).astype(data_type)
    print('\b\b: Done in %g seconds.' % (tm.time()-c))

    lf_img -= d0
    lf_img /= i0
    lf_img[lf_img < np.finfo(np.float32).eps] = np.finfo(np.float32).eps
    lf_img = -np.log(lf_img)
    lf_img = np.flip(lf_img, axis=0)

    if det_is_fixed and crop_fixed_det:
        sd_so_diff = sd_dist - so_dist
        camera_physical_pixsize = camera.pixel_size_ts * (sd_dist / so_dist)
        compute_shift = lambda pos_vu: pos_vu * camera.pixel_size_vu / camera_physical_pixsize * sd_so_diff / so_dist

        half_size_vu = (camera.data_size_vu.astype(np.float32) - 1) / 2
        max_shift_ts = compute_shift(half_size_vu)

        half_size_ts = (camera.data_size_ts.astype(np.float32) - 1) / 2

        final_size_ts = camera.data_size_ts - 2 * max_shift_ts
        half_final_size_ts = (final_size_ts - 1) / 2

        print('Cropping images:\n- cropped size (t, s): [%g, %g]\n- maximum shifts (t, s): [%g, %g]\n- processing: ' % (
            final_size_ts[0], final_size_ts[1], max_shift_ts[0], max_shift_ts[1]), end='', flush=True)
        c = tm.time()

        final_size_ts = np.floor(final_size_ts)
        lf_crop_size = np.concatenate((camera.data_size_vu, final_size_ts)).astype(np.intp)
        lf_img_crop = np.zeros(lf_crop_size, dtype=data_type)

        base_grid_t = np.linspace(
                -half_size_ts[0], half_size_ts[0], camera.data_size_ts[0])
        base_grid_s = np.linspace(
                -half_size_ts[1], half_size_ts[1], camera.data_size_ts[1])

        for ii_v in range(camera.data_size_vu[0]):
            for ii_u in range(camera.data_size_vu[1]):
                prnt_str = 'v %02d/%02d, u %02d/%02d' % (
                        ii_v, camera.data_size_vu[0], ii_u, camera.data_size_vu[1])
                print(prnt_str, end='', flush=True)

                shift_vu = np.array([ii_v, ii_u]).astype(np.float32) - half_size_vu
                shift_ts = compute_shift(shift_vu)
                base_int_t = np.linspace(
                        -half_final_size_ts[0] + shift_ts[0],
                        half_final_size_ts[0] + shift_ts[0], final_size_ts[0])
                base_int_s = np.linspace(
                        -half_final_size_ts[1] + shift_ts[1],
                        half_final_size_ts[1] + shift_ts[1], final_size_ts[1])
                base_int_ts = np.meshgrid(base_int_t, base_int_s, indexing='ij')
                base_int_ts = np.transpose(np.array(base_int_ts), axes=(1, 2, 0))

                interp_obj = spinterp.RegularGridInterpolator(
                        (base_grid_t, base_grid_s), lf_img[ii_v, ii_u, ...],
                        bounds_error=False, fill_value=0.0)

                lf_img_crop[ii_v, ii_u, ...] = interp_obj(base_int_ts)

                print(('\b') * len(prnt_str), end='', flush=True)

        lf_img = lf_img_crop
        camera.data_size_ts = final_size_ts.astype(np.intp)
        print('Done in %g seconds.' % (tm.time() - c))

    lf_img = np.transpose(lf_img, axes=(2, 3, 0, 1))
    return lightfield.Lightfield(camera_type=camera, data=lf_img)
