#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module implements the handling of the .vox data format.
It exposes routines that allow to create, calibrate and read vox files.

Convetions:
- arrays are understood in the C convention

Rationale behind version 0:
- Single light-field per file
- Contains all the necessary metadata for interpretation of the light-field
- Based on [HDF5](https://www.hdfgroup.org/solutions/hdf5/) and vaguely
inspired by [DataExchange](https://github.com/data-exchange/dxchange)

The updated structure can be found at: https://cicwi.github.io/plenoptomos

@author: Nicola Vigano, Charlotte Herzog, Pablo Martinez

Created on Tue Nov 21 23:50:47 2017
"""

import os
import shutil

import configparser
import h5py

import numpy as np
import scipy.interpolate as spinterp
import scipy.signal as spsig
import matplotlib.pyplot as plt

try:
    import imageio as iio
except ImportError as ex:
    print('WARNING: error importing Imageio, using matplotlib instead')
    print('Error message:\n', ex)
    import matplotlib.image as iio

from . import lightfield


def _load_raw_sensor_image(raw_file, file_ext):
    im = iio.imread(raw_file)
    if len(im.shape) == 3 and im.shape[2] > 1:
        im = im[..., 0]
    if file_ext[1:].lower() == 'png':
        im *= 255
    return im


def _assign_optional_attr(obj, conf, label):
    try:
        val = conf[label]
    except KeyError:
        return

    obj.attrs[label] = val


def create_vox_from_raw(
        raw_det_file, conf_file, out_file=None, raw_det_white=None,
        raw_det_dark=None, raw_det_mask=None):
    """ Creates a .vox light-field file, from a raw detector image and metadata.

    :param raw_det_file: the raw image file path (string)
    :param conf_file: the metadata file path (string)
    :param out_file: the uncalibrated light-field file path (string)
    :param raw_det_white: the flat-field image file path (string, default: None)
    :param raw_det_dark: the dark-field image file path (string, default: None)
    :param raw_det_mask: the light-field mask image file path (string, default: None)
    """
    config = configparser.ConfigParser()
    config.read(conf_file)

    (file_base, file_ext) = os.path.splitext(raw_det_file)
    if out_file is None:
        out_file = "%s.vox" % file_base

    data = _load_raw_sensor_image(raw_det_file, file_ext)
    white = dark = mask = None
    if raw_det_white is not None:
        (_, file_ext_w) = os.path.splitext(raw_det_white)
        white = _load_raw_sensor_image(raw_det_white, file_ext_w)
    if raw_det_dark is not None:
        (_, file_ext_d) = os.path.splitext(raw_det_dark)
        dark = _load_raw_sensor_image(raw_det_dark, file_ext_d)
    if raw_det_mask is not None:
        (_, file_ext_d) = os.path.splitext(raw_det_mask)
        mask = _load_raw_sensor_image(raw_det_mask, file_ext_d)

    create_vox_from_dict(data, config, out_file, white=white, dark=dark, mask=mask)


def create_vox_from_lf(lf: lightfield.Lightfield, out_file):
    """ Creates a .vox light-field file, from the light-field data class.

    :param lf: the light-field data class (lightfield.LightField)
    :param out_file: the uncalibrated light-field file path (string)
    """
    conf_cam = dict(manufacturer='', model=lf.camera.model)

    conf_ml = dict(
            size_y=lf.camera.data_size_vu[0], size_x=lf.camera.data_size_vu[1],
            physical_size_t=lf.camera.pixel_size_ts[0],
            physical_size_s=lf.camera.pixel_size_ts[1],
            micro_image_size_y=0.0, micro_image_size_x=0.0,
            f2=lf.camera.f2, aperture=lf.camera.aperture_f2)

    if lf.camera.is_focused():
        z_mla = lf.camera.z1 + lf.camera.a
        z_sensor = lf.camera.z1 + lf.camera.a + lf.camera.b
    else:
        z_mla = lf.camera.z1
        z_sensor = lf.camera.z1 + lf.camera.f2

    conf_mla = dict(
            size_t=lf.camera.data_size_ts[0], size_s=lf.camera.data_size_ts[1],
            x=0.0, y=0.0, z=z_mla)

    conf_Ml = dict(
            pixel_size_v=lf.camera.pixel_size_vu[0],
            pixel_size_u=lf.camera.pixel_size_vu[1],
            f1=lf.camera.f1, aperture=lf.camera.aperture_f1)

    conf_s = dict(
            size_y=lf.camera.data_size_ts[0]*lf.camera.data_size_vu[0],
            size_x=lf.camera.data_size_ts[1]*lf.camera.data_size_vu[1],
            pixel_size_y=lf.camera.pixel_size_yx[0],
            pixel_size_x=lf.camera.pixel_size_yx[1],
            x=0.0, y=0.0, z=z_sensor)

    config = dict(
            camera=conf_cam, micro_lens=conf_ml, micro_lenses_array=conf_mla,
            main_lens=conf_Ml, sensor=conf_s)

    data = lf.get_raw_detector_picture(image='data')
    if lf.flat is not None:
        flat = lf.get_raw_detector_picture(image='flat')
    else:
        flat = None
    if lf.mask is not None:
        mask = lf.get_raw_detector_picture(image='mask')
    else:
        mask = None
    create_vox_from_dict(data, config, out_file, white=flat, mask=mask)


def create_vox_from_dict(data, config, out_file, white=None, dark=None, mask=None):
    """ Creates a .vox light-field file, from a raw detector image and a
    dictionary containing metadata.

    :param data: the raw image (numpy.array_like)
    :param config: the dictionary containing metadata (numpy.array_like)
    :param out_file: the uncalibrated light-field file path (string)
    :param white: the flat-field image (numpy.array_like, default: None)
    :param dark: the dark-field image (numpy.array_like, default: None)
    :param mask: the light-field mask image (numpy.array_like, default: None)
    """

    with h5py.File(out_file, 'w') as f:
        f.attrs['description'] = 'VoxelDataFormat'
        f.attrs['version'] = 'v0'

        data_g = f.create_group('data')

        data_im = data_g.create_dataset('image', data=data, compression="gzip", compression_opts=9)
        data_im.attrs['units'] = 'counts'
        data_im.attrs['axes'] = 'tau:sigma'
        data_im.attrs['mode'] = 'raw'

        if white is not None:
            data_im = data_g.create_dataset('white', data=white, compression="gzip", compression_opts=9)
            data_im.attrs['units'] = 'counts'
            data_im.attrs['axes'] = 'tau:sigma'
            data_im.attrs['mode'] = 'raw'

        if dark is not None:
            data_im = data_g.create_dataset('dark', data=dark, compression="gzip", compression_opts=9)
            data_im.attrs['units'] = 'counts'
            data_im.attrs['axes'] = 'tau:sigma'
            data_im.attrs['mode'] = 'raw'

        if mask is not None:
            data_im = data_g.create_dataset('mask', data=mask, compression="gzip", compression_opts=9)
            data_im.attrs['units'] = 'counts'
            data_im.attrs['axes'] = 'tau:sigma'
            data_im.attrs['mode'] = 'raw'

        # Creating INSTRUMENT group
        instrument = f.create_group('instrument')

        # Creating CAMERA group
        camera_g = instrument.create_group('camera')
        try:
            conf_cam = config['camera']

            _assign_optional_attr(camera_g, conf_cam, 'manufacturer')
            _assign_optional_attr(camera_g, conf_cam, 'model')
        except KeyError:
            pass

        # Creating MICRO_LENS group
        conf_ml = config['micro_lens']
        ml_g = camera_g.create_group('micro_lens')

        ml_size = ml_g.create_dataset('size', (2, ), dtype='i')
        ml_size[:] = np.array((int(conf_ml['size_y']), int(conf_ml['size_x'])))
        ml_size.attrs['axes'] = 'tau:sigma'
        ml_size.attrs['units'] = 'pixels'

        ml_psize = ml_g.create_dataset('physical_size', (2, ), dtype='f')
        ml_psize[:] = np.array((float(conf_ml['physical_size_t']), float(conf_ml['physical_size_s'])))
        ml_psize.attrs['axes'] = 't:s'
        ml_psize.attrs['units'] = 'mm'

        ml_mi_size = ml_g.create_dataset('micro_image_size', (2, ), dtype='f')
        ml_mi_size[:] = np.array((float(conf_ml['micro_image_size_y']), float(conf_ml['micro_image_size_x'])))
        ml_mi_size.attrs['axes'] = 'tau:sigma'
        ml_mi_size.attrs['units'] = 'mm'

        ml_f2 = ml_g.create_dataset('f2', data=float(conf_ml['f2']))
        ml_f2.attrs['units'] = 'mm'

        ml_a = ml_g.create_dataset('aperture', data=float(conf_ml['aperture']))
        ml_a.attrs['units'] = 'mm'

        # Creating MICRO_LENSES_ARRAY group
        conf_mla = config['micro_lenses_array']
        mla_g = camera_g.create_group('micro_lenses_array')

        _assign_optional_attr(mla_g, conf_mla, 'manufacturer')
        _assign_optional_attr(mla_g, conf_mla, 'model')

        mla_size = mla_g.create_dataset('size', (2, ), dtype='i')
        mla_size[:] = np.array((int(conf_mla['size_t']), int(conf_mla['size_s'])))
        mla_size.attrs['axes'] = 'tau:sigma'

        mla_pos = mla_g.create_dataset('position', (3, ), dtype='f')
        mla_pos[:] = np.array((float(conf_mla['x']), float(conf_mla['y']), float(conf_mla['z'])))
        mla_pos.attrs['axes'] = 'x:y:z'
        mla_pos.attrs['units'] = 'mm'

        # Creating MAIN_LENS group
        conf_Ml = config['main_lens']
        Ml_g = camera_g.create_group('main_lens')

        _assign_optional_attr(Ml_g, conf_Ml, 'manufacturer')
        _assign_optional_attr(Ml_g, conf_Ml, 'model')

        Ml_psize = Ml_g.create_dataset('pixel_size', (2, ), dtype='f')
        Ml_psize[:] = np.array((float(conf_Ml['pixel_size_v']), float(conf_Ml['pixel_size_u'])))
        Ml_psize.attrs['axes'] = 'v:u'
        Ml_psize.attrs['units'] = 'mm'

        Ml_f1 = Ml_g.create_dataset('f1', data=float(conf_Ml['f1']))
        Ml_f1.attrs['units'] = 'mm'

        Ml_a = Ml_g.create_dataset('aperture', data=float(conf_Ml['aperture']))
        Ml_a.attrs['units'] = 'mm'

        # Creating SENSOR group
        conf_s = config['sensor']
        s_g = camera_g.create_group('sensor')

        _assign_optional_attr(s_g, conf_s, 'manufacturer')
        _assign_optional_attr(s_g, conf_s, 'model')

        s_size = s_g.create_dataset('size', (2, ), dtype='i')
        s_size[:] = np.array((int(conf_s['size_y']), int(conf_s['size_x'])))
        s_size.attrs['axes'] = 'tau:sigma'
        s_size.attrs['units'] = 'pixels'

        s_p_size = s_g.create_dataset('pixel_size', (2, ), dtype='f')
        s_p_size[:] = np.array((float(conf_s['pixel_size_y']), float(conf_s['pixel_size_x'])))
        s_p_size.attrs['axes'] = 'tau:sigma'
        s_p_size.attrs['units'] = 'mm'

        s_pos = s_g.create_dataset('position', (3, ), dtype='f')
        s_pos[:] = np.array((float(conf_s['x']), float(conf_s['y']), float(conf_s['z'])))
        s_pos.attrs['axes'] = 'x:y:z'
        s_pos.attrs['units'] = 'mm'

        # And now we should check a couple of things to see if they are
        # consistent, like: the main lens pixel size -> (u, v) resolutions
        if np.any(Ml_psize[:] == 0):
            Ml_psize[:] = s_p_size[:] / (s_pos[2] - mla_pos[2]) * mla_pos[2]

###############################################################################


def _find_offsets_and_pitch(
        w_im, peak_rm_front=(None, None), peak_rm_back=(None, None),
        peak_skip_front=(None, None), peak_skip_back=(None, None),
        verbose=False, interactive=False, cwt_size=np.arange(1, 25)):

    win = spsig.general_gaussian(15, p=0.5, sig=2)
    win /= np.sum(win)

    if verbose or interactive:
        f, ax = plt.subplots(2, 2)
        ax[0, 0].set_title('Peaks found (dim 0)')
        ax[1, 0].set_title('Peaks found (dim 1)')
        ax[0, 1].set_title('Fitted peaks (dim 0)')
        ax[1, 1].set_title('Fitted peaks(dim 1)')
        f.tight_layout()

    peak_rm_front = np.array(peak_rm_front)
    peak_rm_back = np.array(peak_rm_back)
    peak_skip_front = np.array(peak_skip_front)
    peak_skip_back = np.array(peak_skip_back)

    mean_dist = np.empty((2, ))
    offset = np.empty((2, ))
    for ii_d in (0, 1):
        summed_micro_imgs = np.sum(w_im, axis=(1 - ii_d))
        summed_micro_imgs_filt = spsig.convolve(summed_micro_imgs, win, mode='same')

        peak_pos = spsig.find_peaks_cwt(summed_micro_imgs_filt, cwt_size)
        if verbose:
            print(peak_pos)

        max_peaks = np.max(summed_micro_imgs)
        if verbose or interactive:

            ax[ii_d, 0].plot(summed_micro_imgs)
            for ii in peak_pos:
                ax[ii_d, 0].plot((ii, ii), (0, max_peaks), 'r-')
            f.tight_layout()
            plt.draw()
            plt.show(block=False)

        if interactive:
            print('Dimension %d, please indicate the spurious peaks to remove in the front and back:' % ii_d)
            peak_rm_front[ii_d] = int(input(' - front? '))
            peak_rm_back[ii_d] = int(input(' - back? '))

        if peak_rm_front[ii_d] is not None:
            peak_pos = peak_pos[peak_rm_front[ii_d]:]
        if not (peak_rm_back[ii_d] is None or peak_rm_back[ii_d] == 0):
            peak_pos = peak_pos[:-peak_rm_back[ii_d]]

        if interactive:
            print('Dimension %d, ' % ii_d, end='', flush=True)
            print('please indicate the misidentified peaks to not consider for averange and offset, in the front and back:')
            peak_skip_front[ii_d] = int(input(' - front? '))
            peak_skip_back[ii_d] = int(input(' - back? '))

        peak_pos_to_use = peak_pos
        if peak_skip_front[ii_d] is not None:
            peak_pos_to_use = peak_pos_to_use[peak_skip_front[ii_d]:]
        if not (peak_skip_back[ii_d] is None or peak_skip_back[ii_d] == 0):
            peak_pos_to_use = peak_pos_to_use[:-peak_skip_back[ii_d]]

        dists = np.diff(peak_pos_to_use)
        mean_dist[ii_d] = np.mean(dists)

        range_offset = (peak_skip_front[ii_d] or 0) + 0.5
        peak_ind = np.arange(range_offset, len(peak_pos_to_use) + range_offset)

        peak_offsets = peak_pos_to_use - peak_ind * mean_dist[ii_d]
        offset[ii_d] = np.mean(peak_offsets)

        if verbose or interactive:
            print('Dimension %d: mean micro-image size: %g, mean array offset %g' % (ii_d, mean_dist[ii_d], offset[ii_d]))

            ax[ii_d, 1].plot(summed_micro_imgs)

            peak_ind = np.arange(0.5, len(peak_pos)+0.5)
            for ii in peak_ind:
                peak_position = ii * mean_dist[ii_d] + offset[ii_d]
                ax[ii_d, 1].plot((peak_position, peak_position), (0, max_peaks), 'r-')
            plt.draw()
            f.tight_layout()
            plt.show(block=False)

    return (mean_dist, offset)


def calibrate_raw_image(vox_file_in, vox_file_out=None, pitch=None, offset=None):
    """ Calibrate the array offsets and lenslet pitch for a .vox light-field file

    :param vox_file_in: the uncalibrated light-field file path (string)
    :param vox_file_out: the calibrated light-field file path (string)
    :param pitch: OPTIONAL lenslet pitch, that would otherwise be calibrated (<2x1> list or tuple, default: None)
    :param offset: OPTIONAL MLA offset, that would otherwise be calibrated (<2x1> list or tuple, default: None)
    """
    if vox_file_out is None:
        (out_file_base, out_file_ext) = os.path.splitext(vox_file_in)
        vox_file_out = '%s_calibrated%s' % (out_file_base, out_file_ext)

    if os.path.exists(vox_file_out):
        os.remove(vox_file_out)
    shutil.copy2(vox_file_in, vox_file_out)

    with h5py.File(vox_file_in, 'r') as f_in:
        w_im = f_in['/data/white'][()]
        satisfied = pitch is not None and offset is not None
        while not satisfied:
            (pitch, offset) = _find_offsets_and_pitch(w_im, interactive=True)
            answer = input('Are you satisfied? (y/n [y]): ')
            satisfied = answer.lower() in ('y', '')

        with h5py.File(vox_file_out, 'r+') as f_out:
            det_p_size = f_in['/instrument/camera/sensor/pixel_size'][()]
            f_out['/instrument/camera/micro_lens/physical_size'][:] = pitch * det_p_size
            mla_xy = f_out['/instrument/camera/micro_lenses_array/position'][0:2]
            f_out['/instrument/camera/sensor/position'][0:2] = mla_xy + offset * det_p_size

    return (pitch, offset)

###############################################################################


def _has_fields_for_raw_to_microimage(vox_file):
    return not (
        np.any(vox_file['/instrument/camera/main_lens/pixel_size'][()] == 0)
        or np.any(vox_file['/instrument/camera/micro_lens/physical_size'][()] == 0))


def raw_to_microimage_exact(data, offsets, pitch):
    lenslets_last_inds_x = np.arange((offsets[1] + pitch[1]), data.shape[1] + 1, pitch[1])
    lenslets_last_inds_y = np.arange((offsets[0] + pitch[0]), data.shape[0] + 1, pitch[0])

    data = data[offsets[0]:lenslets_last_inds_y[-1], offsets[1]:lenslets_last_inds_x[-1]]

    array_size = np.array([len(lenslets_last_inds_y), len(lenslets_last_inds_x)])
    out_shape_tsvu = np.concatenate((array_size, pitch))
    return (data, out_shape_tsvu)


def raw_to_microimage_interp(data, offsets, pitch_in, pitch_out, array_size=None):
    pitch_out = pitch_out.astype(np.intp)
    # Let's first identify the lenslet interpolation grid:
    interp_points_x = np.linspace(0, pitch_in[1], pitch_out[1] + 1)
    interp_points_x = (interp_points_x[1:] + interp_points_x[:-1]) / 2

    interp_points_y = np.linspace(0, pitch_in[0], pitch_out[0] + 1)
    interp_points_y = (interp_points_y[1:] + interp_points_y[:-1]) / 2

    # And we now replicate it over all the lenslets to obtain the global interpolation grid:
    if array_size is None:
        array_size = np.floor((data.shape[0:2] - offsets) / pitch_in).astype(np.intp)

    offsets_x = pitch_in[1] * np.arange(0, array_size[1]) + offsets[1] + 0.5
    offsets_x = np.tile(offsets_x, reps=(pitch_out[1], 1))

    interp_points_x = np.reshape(interp_points_x, (-1, 1))
    interp_points_x = np.tile(interp_points_x, reps=(1, array_size[1]))
    interp_points_x = interp_points_x + offsets_x
    interp_points_x = np.squeeze(np.reshape(interp_points_x.T, (-1, 1)))

    offsets_y = pitch_in[0] * np.arange(0, array_size[0]) + offsets[0] + 0.5
    offsets_y = np.tile(offsets_y, reps=(pitch_out[0], 1))

    interp_points_y = np.reshape(interp_points_y, (-1, 1))
    interp_points_y = np.tile(interp_points_y, reps=(1, array_size[0]))
    interp_points_y = interp_points_y + offsets_y
    interp_points_y = np.squeeze(np.reshape(interp_points_y.T, (-1, 1)))

    # Apply interpolation (+ 1 is used to match exactly original Matlab code):
    samp_x = np.arange(1, data.shape[1]+1)
    samp_y = np.arange(1, data.shape[0]+1)
    out_grid = np.meshgrid(interp_points_y, interp_points_x, indexing='ij')
    out_grid = np.array(out_grid)
    out_grid = np.squeeze(np.transpose(out_grid, axes=(1, 2, 0)))

    out_shape_tsvu = np.concatenate((array_size, pitch_out))
    out_data = spinterp.interpn((samp_y, samp_x), data, out_grid, bounds_error=False, fill_value=0)
    return (out_data, out_shape_tsvu)


def transform_2D_to_4D(data, out_shape_tsvu):
    # Organize the lightfield as a 4-D structure:
    intermediate_sizes = (out_shape_tsvu[2], out_shape_tsvu[0], out_shape_tsvu[3], out_shape_tsvu[1])
    data = np.reshape(data, intermediate_sizes, order='F')
    return np.transpose(data, axes=(1, 3, 0, 2))


def _load_raw_image(f):
    im = f['/data/image']
    if not im.attrs['mode'] == 'raw':
        raise ValueError('This function loads VOX data in raw detector format')
    if not _has_fields_for_raw_to_microimage(f):
        raise ValueError('You should calibrate your images first!')

    mla_xy = f['/instrument/camera/micro_lenses_array/position'][0:2]
    sensor_xy = f['/instrument/camera/sensor/position'][0:2]
    sensor_pixel_size = f['/instrument/camera/sensor/pixel_size'][()]
    offsets = (sensor_xy - mla_xy) / sensor_pixel_size
    pixel_size_ts = f['/instrument/camera/micro_lens/physical_size'][()]
    pitch_in = pixel_size_ts / sensor_pixel_size
    pitch_out = f['/instrument/camera/micro_lens/size'][()]

    offsets = np.array(offsets)
    pitch_in = np.array(pitch_in)
    pitch_out = np.array(pitch_out)
    if np.any(pitch_out == 0):
        pitch_out = np.ceil(pitch_in - 0.1)

    pitch_diff = np.abs(pitch_in - pitch_out)
    offset_diff = np.abs(offsets - np.round(offsets))

    if np.all(pitch_diff < np.finfo(np.float16).eps) \
            and np.all(offset_diff < np.finfo(np.float16).eps):
        pitch_in = np.round(pitch_in).astype(np.intp)
        offsets = np.round(offsets).astype(np.intp)

        extract = lambda x: raw_to_microimage_exact(x, offsets, pitch_in)
    else:
        extract = lambda x: raw_to_microimage_interp(x, offsets, pitch_in, pitch_out)

    (data, lf_shape) = extract(im[()])
    data_out = [data]
    if '/data/white' in f:
        im_w = f['/data/white']
        (white, _) = extract(im_w[()])
        data_out.append(white)
    if '/data/dark' in f:
        im_d = f['/data/dark']
        (dark, _) = extract(im_d[()])
        data_out.append(dark)

    for ii in range(len(data_out)):
        data_out[ii] = transform_2D_to_4D(data_out[ii], lf_shape)

    return (data_out, lf_shape)


def load(vox_file):
    """ Loads a .vox light-field file

    :param vox_file: the light-field file path (string)

    :returns: the loaded light-field
    :rtype: light-field.Lightfield
    """

    camera = lightfield.Camera()

    with h5py.File(vox_file, 'r') as f:
        im = f['/data/image']
        if im.attrs['mode'] == 'raw':
            (data, lf_shape) = _load_raw_image(f)
            lf_data = data[0]
            camera.data_size_ts = lf_shape[0:2]
            camera.data_size_vu = lf_shape[2:4]
            lf_mode = 'micro-image'
            if len(data) > 1:
                white = data[1]
            else:
                white = None
        else:
            lf_data = im[()]
            camera.data_size_ts = f['/instrument/camera/micro_lenses_array/size'][()]
            camera.data_size_vu = f['/instrument/camera/micro_lens/size'][()]
            lf_mode = im.attrs['mode']
            if '/data/white' in f:
                white = f['/data/white'][()]
            else:
                white = None

            if '/data/dark' in f:
                dark = f['/data/dark'][()]
                lf_data -= dark
                lf_data = np.fmax(lf_data, 0)
                white -= dark
                white = np.fmax(white, 0)

        camera.aperture_f1 = f['/instrument/camera/main_lens/aperture'][()]
        camera.aperture_f2 = f['/instrument/camera/micro_lens/aperture'][()]
        camera.f1 = f['/instrument/camera/main_lens/f1'][()]
        camera.f2 = f['/instrument/camera/micro_lens/f2'][()]
        camera.z1 = f['/instrument/camera/micro_lenses_array/position'][2]
        camera.pixel_size_ts = f['/instrument/camera/micro_lens/physical_size'][()]
        camera.pixel_size_vu = f['/instrument/camera/main_lens/pixel_size'][()]
        camera.pixel_size_yx = f['/instrument/camera/sensor/pixel_size'][()]

        mla_z = f['/instrument/camera/micro_lenses_array/position'][2]
        sensor_z = f['/instrument/camera/sensor/position'][2]
        z2 = sensor_z - mla_z
        if (np.abs(z2 - camera.f2) / camera.f2) > 1e-2:
            print('Identified focused acquisition setup (z2: %g, f2 %g)... adapting parameters' % (z2, camera.f2))
            camera.b = z2
            camera.a = camera.f2 * z2 / (z2 - camera.f2)
            camera.z1 -= camera.a
            # We also have to adjust the uv resolution!
            camera.pixel_size_vu = (camera.a + camera.z1) / camera.b * camera.pixel_size_yx

    return lightfield.Lightfield(camera, data=lf_data, mode=lf_mode, flat=white)
