#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module implements a few different solvers from the tomography reconstruction literature.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France

Created on Wed Mar  8 11:27:12 2017
"""

import numpy as np
import numpy.linalg as npla

import scipy.sparse as sps

from . import utils_io

import time as tm

try:
    import pywt
    has_pywt = True
    use_swtn = pywt.version.version >= '1.0.2'
except ImportError:
    has_pywt = False
    use_swtn = False
    print('WARNING - pywt was not found')


class Solver(object):

    def __init__(self, verbose=False, dump_file=None, tol=1e-5):
        self.verbose = verbose
        self.dump_file = dump_file
        self.tol = tol

    def mult_matrix(self, M, x, data_shape=None):
        """
        Multiplies a matrix M by and array x.
        """
        if data_shape is None:
            data_shape = x.shape
        return np.reshape(np.dot(M, x.flatten()), data_shape)

    def initialize_data_operators(self, A, At):
        if At is None:
            if isinstance(A, np.ndarray):
                At = A.transpose((1, 0))
            elif isinstance(A, sps.dia_matrix):
                At = A.transpose()
            else:
                At = lambda x: np.dot(A.T, x)
        if isinstance(At, np.ndarray) or isinstance(At, sps.dia_matrix):
            At_m = At
            At = lambda x: self.mult_matrix(At_m, x)
        if isinstance(A, np.ndarray) or isinstance(A, sps.dia_matrix):
            A_m = A
            A = lambda x: self.mult_matrix(A_m, x)
        return (A, At)

    def upper(self):
        return type(self).__name__.upper()

    def lower(self):
        return type(self).__name__.lower()


class BPJ(Solver):

    def __init__(
            self, verbose=False, weight_det=None, dump_file=None, tol=1e-5):
        Solver.__init__(self, verbose=verbose, dump_file=dump_file, tol=tol)
        self.weight_det = weight_det

    def __call__(
            self, A, b, num_iter=None, At=None, upper_limit=None,
            lower_limit=None):
        """
        """
        (A, At) = self.initialize_data_operators(A, At)

        data_type = b.dtype

        c_in = tm.time()

        renorm_bwd = At(np.ones(b.shape, data_type))
        renorm_bwd = np.abs(renorm_bwd)
        renorm_bwd[(renorm_bwd / np.max(renorm_bwd)) < 1e-5] = 1
        renorm_bwd = 1 / renorm_bwd

        c_init = tm.time()

        if self.verbose:
            print("- Performing BPJ (init: %g seconds): " % (c_init - c_in), end='', flush=True)

        x = At(b) * renorm_bwd

        if lower_limit is not None:
            x = np.fmax(x, lower_limit)
        if upper_limit is not None:
            x = np.fmin(x, upper_limit)

        if self.verbose:
            print("Done in %g seconds." % (tm.time() - c_in))

        return (x, 0)


class Sirt(Solver):

    def __init__(
            self, verbose=False, weight_det=None, dump_file=None, tol=1e-5):
        Solver.__init__(self, verbose=verbose, dump_file=dump_file, tol=tol)
        self.weight_det = weight_det

    def __call__(
            self, A, b, num_iter, x0=None, At=None, upper_limit=None,
            lower_limit=None, tol=1e-5):
        """
        """
        (A, At) = self.initialize_data_operators(A, At)

        data_type = b.dtype

        c_in = tm.time()

        renorm_bwd = At(np.ones(b.shape, data_type))
        renorm_bwd = np.abs(renorm_bwd)
        renorm_bwd[(renorm_bwd / np.max(renorm_bwd)) < 1e-5] = 1
        renorm_bwd = 1 / renorm_bwd

        if self.weight_det is None:
            self.weight_det = A(np.ones(renorm_bwd.shape, dtype=data_type))
        renorm_fwd = np.array(self.weight_det)
        renorm_fwd = np.abs(renorm_fwd)
        renorm_fwd[(renorm_fwd / np.max(renorm_fwd)) < 1e-5] = 1
        renorm_fwd = 1 / renorm_fwd

        if x0 is None:
            x0 = At(b * renorm_fwd) * renorm_bwd
        x = x0

        res_norm_0 = npla.norm(b.flatten())
        rel_res_norms = np.empty((num_iter, ))

        if self.dump_file is not None:
            out_x = np.empty(np.concatenate(((num_iter, ), x.shape)), dtype=x.dtype)

        c_init = tm.time()

        if self.verbose:
            print("- Performing SIRT iterations (init: %g seconds): " % (c_init - c_in), end='', flush=True)
        for ii in range(num_iter):
            if self.dump_file is not None:
                out_x[ii, ...] = x

            if self.verbose:
                prnt_str = "%03d/%03d (avg: %g seconds)" % (ii, num_iter, (tm.time() - c_init) / np.fmax(ii, 1))
                print(prnt_str, end='', flush=True)

            res = b - A(x)

            rel_res_norms[ii] = npla.norm(res.flatten()) / res_norm_0
            if self.tol is not None and rel_res_norms[ii] < self.tol:
                break

            x += (At(res * renorm_fwd)) * renorm_bwd

            if lower_limit is not None:
                x = np.fmax(x, lower_limit)
            if upper_limit is not None:
                x = np.fmin(x, upper_limit)

            if self.verbose:
                print(('\b') * len(prnt_str), end='', flush=True)
                print((' ') * len(prnt_str), end='', flush=True)
                print(('\b') * len(prnt_str), end='', flush=True)

        if self.verbose:
            print("Done in %g seconds." % (tm.time() - c_in))

        if self.dump_file is not None:
            utils_io.save_field_toh5(self.dump_file, 'iterations', out_x)

        return (x, rel_res_norms)


class CP_uc(Solver):

    def __init__(
            self, verbose=False, weight_det=None, dump_file=None, tol=1e-5,
            data_term='l2'):
        Solver.__init__(self, verbose=verbose, dump_file=dump_file, tol=tol)
        self.weight_det = weight_det
        self.data_term = data_term

    def __call__(
            self, A, b, num_iter, x0=None, At=None, upper_limit=None,
            lower_limit=None):
        """
        """
        (A, At) = self.initialize_data_operators(A, At)

        data_type = b.dtype

        c_in = tm.time()

        tau = At(np.ones(b.shape, dtype=data_type))
        tau = np.abs(tau)
        tau[(tau / np.max(tau)) < 1e-5] = 1
        tau = 1 / tau

        if self.weight_det is None:
            self.weight_det = A(np.ones(tau.shape, dtype=data_type))
        sigma = np.array(self.weight_det)
        sigma = np.abs(self.weight_det)
        sigma[(sigma / np.max(sigma)) < 1e-5] = 1
        sigma = 1 / sigma

        sigma1 = 1 / (1 + sigma)

        if x0 is None:
            x0 = np.zeros_like(tau)
        x = x0
        x_ehn = x

        p = np.zeros(b.shape, dtype=data_type)

        res_norm_0 = npla.norm(b.flatten())
        rel_res_norms = np.empty((num_iter, ))

        if self.dump_file is not None:
            out_x = np.empty(np.concatenate(((num_iter, ), x.shape)), dtype=x.dtype)

        if self.data_term.lower() == 'kl':
            b_kl = 4 * sigma * b

        c_init = tm.time()

        if self.verbose:
            print("- Performing CP-%s iterations (init: %g seconds): " % (self.data_term, c_init - c_in), end='', flush=True)
        for ii in range(num_iter):
            if self.verbose:
                prnt_str = "%03d/%03d (avg: %g seconds)" % (ii, num_iter, (tm.time() - c_init) / np.fmax(ii, 1))
                print(prnt_str, end='', flush=True)

            fp = A(x_ehn)

            res = fp - b

            rel_res_norms[ii] = npla.norm(res.flatten()) / res_norm_0
            if self.tol is not None and rel_res_norms[ii] < self.tol:
                break

            if self.data_term.lower() == 'kl':
                p += fp * sigma
                p = (1 + p - np.sqrt((p - 1) ** 2 + b_kl)) / 2
            else:
                p += res * sigma
                if self.data_term.lower() == 'l1':
                    p /= np.fmax(1, np.abs(p))
                elif self.data_term.lower() == 'l2':
                    p *= sigma1
                else:
                    raise ValueError("Unknown data term: %s" % self.data_term)

            x_new = x - At(p) * tau

            if lower_limit is not None:
                x_new = np.fmax(x_new, lower_limit)
            if upper_limit is not None:
                x_new = np.fmin(x_new, upper_limit)

            x_ehn = x_new + (x_new - x)
            x = x_new

            if self.dump_file is not None:
                out_x[ii, ...] = x

            if self.verbose:
                print(('\b') * len(prnt_str), end='', flush=True)
                print((' ') * len(prnt_str), end='', flush=True)
                print(('\b') * len(prnt_str), end='', flush=True)

        if self.verbose:
            print("Done in %g seconds." % (tm.time() - c_in))

        if self.dump_file is not None:
            utils_io.save_field_toh5(self.dump_file, 'iterations', out_x)

        return (x, rel_res_norms)


class Operations(object):

    def __init__(self, axes=None):
        self.axes = axes

    def gradient(self, x):
        num_dims = len(x.shape)
        d = []
        for ax in self.axes:
            pad_pattern = [(0, 0)] * num_dims
            pad_pattern[ax] = (0, 1)
            temp_d = np.pad(x, pad_pattern, mode='constant')
            temp_d = np.diff(temp_d, n=1, axis=ax)
            d.append(temp_d)
        return d

    def divergence(self, x):
        num_dims = len(x.shape)-1
        for ii, ax in enumerate(self.axes):
            pad_pattern = [(0, 0)] * num_dims
            pad_pattern[ax] = (1, 0)
            temp_d = np.pad(x[ii, ...], pad_pattern, mode='constant')
            temp_d = np.diff(temp_d, n=1, axis=ax)
            if ii == 0:
                final_d = temp_d
            else:
                final_d += temp_d
        return final_d

    def laplacian(self, x):
        num_dims = len(x.shape)
        d = []
        for ax in self.axes:
            pad_pattern = [(0, 0)] * num_dims
            pad_pattern[ax] = (1, 1)
            temp_d = np.pad(x, pad_pattern, mode='edge')
            temp_d = np.diff(temp_d, n=2, axis=ax)
            d.append(temp_d)
        return np.sum(d, axis=0)


class CP_tv(Solver, Operations):

    def __init__(
            self, verbose=False, weight_det=None, dump_file=None, tol=1e-5,
            data_term='l2', lambda_tv=1e-2, axes=None):
        Solver.__init__(self, verbose=verbose, dump_file=dump_file, tol=tol)
        Operations.__init__(self, axes)

        self.weight_det = weight_det

        self.data_term = data_term
        self.lambda_tv = lambda_tv
        self.axes = axes

    def __call__(
            self, A, b, num_iter, x0=None, At=None, upper_limit=None,
            lower_limit=None):
        """
        """
        (A, At) = self.initialize_data_operators(A, At)

        data_type = b.dtype

        c_in = tm.time()

        tau = At(np.ones(b.shape, dtype=data_type))
        tau = np.abs(tau)

        if self.axes is None:
            self.axes = range(len(tau.shape))

        tau[(tau / np.max(tau)) < 1e-5] = 1
        tau = 1 / (tau + 2 * len(self.axes) * self.lambda_tv)

        if self.weight_det is None:
            self.weight_det = A(np.ones(tau.shape, dtype=data_type))
        sigma = np.array(self.weight_det)
        sigma = np.abs(sigma)
        sigma[(sigma / np.max(sigma)) < 1e-5] = 1
        sigma = 1 / sigma

        sigma1 = 1 / (1 + sigma)

        if x0 is None:
            x0 = np.zeros_like(tau)
        x = x0
        x_ehn = x

        p = np.zeros(b.shape, dtype=data_type)

        q_g = np.zeros(np.concatenate(((len(self.axes), ), x.shape)), dtype=data_type)

        res_norm_0 = npla.norm(b.flatten())
        rel_res_norms = np.empty((num_iter, ))

        if self.dump_file is not None:
            out_x = np.empty(np.concatenate(((num_iter, ), x.shape)), dtype=x.dtype)

        if self.data_term.lower() == 'kl':
            b_kl = 4 * sigma * b

        c_init = tm.time()

        if self.verbose:
            print("- Performing CP-%s-TV iterations (init: %g seconds): " % (self.data_term, (c_init - c_in)),
                  end='', flush=True)
        for ii in range(num_iter):
            if self.verbose:
                prnt_str = "%03d/%03d (avg: %g seconds)" % (ii, num_iter, (tm.time() - c_init) / np.fmax(ii, 1))
                print(prnt_str, end='', flush=True)

            fp = A(x_ehn)

            res = fp - b

            rel_res_norms[ii] = npla.norm(res.flatten()) / res_norm_0
            if self.tol is not None and rel_res_norms[ii] < self.tol:
                break

            if self.data_term.lower() == 'kl':
                p += fp * sigma
                p = (1 + p - np.sqrt((p - 1) ** 2 + b_kl)) / 2
            else:
                p += res * sigma
                if self.data_term.lower() == 'l1':
                    p /= np.fmax(1, np.abs(p))
                elif self.data_term.lower() == 'l2':
                    p *= sigma1
                else:
                    raise ValueError("Unknown data term: %s" % self.data_term)

            d = self.gradient(x_ehn)
            d_2 = np.stack(d) / 2
            q_g += d_2
            grad_l2_norm = np.fmax(1, np.sqrt(np.sum(np.abs(q_g) ** 2, axis=0)))
            q_g /= grad_l2_norm

            x_new = x + (self.lambda_tv * self.divergence(q_g) - At(p)) * tau

            if lower_limit is not None:
                x_new = np.fmax(x_new, lower_limit)
            if upper_limit is not None:
                x_new = np.fmin(x_new, upper_limit)

            x_ehn = x_new + (x_new - x)
            x = x_new

            if self.dump_file is not None:
                out_x[ii, ...] = x

            if self.verbose:
                print(('\b') * len(prnt_str), end='', flush=True)
                print((' ') * len(prnt_str), end='', flush=True)
                print(('\b') * len(prnt_str), end='', flush=True)

        if self.verbose:
            print("Done in %g seconds." % (tm.time() - c_in))

        if self.dump_file is not None:
            utils_io.save_field_toh5(self.dump_file, 'iterations', out_x)

        return (x, rel_res_norms)


class CP_smooth(Solver, Operations):

    def __init__(
            self, verbose=False, weight_det=None, dump_file=None, tol=1e-5,
            data_term='l2', lambda_d2=1e-2, axes=None):
        Solver.__init__(self, verbose=verbose, dump_file=dump_file, tol=tol)
        Operations.__init__(self, axes)

        self.weight_det = weight_det

        self.data_term = data_term
        self.lambda_d2 = lambda_d2
        self.axes = axes

    def __call__(
            self, A, b, num_iter, x0=None, At=None, upper_limit=None,
            lower_limit=None):
        """
        """
        (A, At) = self.initialize_data_operators(A, At)

        data_type = b.dtype

        c_in = tm.time()

        tau = At(np.ones(b.shape, dtype=data_type))
        tau = np.abs(tau)

        if self.axes is None:
            self.axes = range(len(tau.shape))

        tau[(tau / np.max(tau)) < 1e-5] = 1
        tau = 1 / (tau + 4 * len(self.axes) * self.lambda_d2)

        if self.weight_det is None:
            self.weight_det = A(np.ones(tau.shape, dtype=data_type))
        sigma = np.array(self.weight_det)
        sigma = np.abs(sigma)
        sigma[(sigma / np.max(sigma)) < 1e-5] = 1
        sigma = 1 / sigma

        sigma1 = 1 / (1 + sigma)

        if x0 is None:
            x0 = np.zeros_like(tau)
        x = x0
        x_ehn = x

        p = np.zeros(b.shape, dtype=data_type)

        q_g = np.zeros(x.shape, dtype=data_type)

        res_norm_0 = npla.norm(b.flatten())
        rel_res_norms = np.empty((num_iter, ))

        if self.dump_file is not None:
            out_x = np.empty(np.concatenate(((num_iter, ), x.shape)), dtype=x.dtype)

        if self.data_term.lower() == 'kl':
            b_kl = 4 * sigma * b

        c_init = tm.time()

        if self.verbose:
            print("- Performing CP-%s-TV iterations (init: %g seconds): " % (self.data_term, (c_init - c_in)),
                  end='', flush=True)
        for ii in range(num_iter):
            if self.verbose:
                prnt_str = "%03d/%03d (avg: %g seconds)" % (ii, num_iter, (tm.time() - c_init) / np.fmax(ii, 1))
                print(prnt_str, end='', flush=True)

            fp = A(x_ehn)

            res = fp - b

            rel_res_norms[ii] = npla.norm(res.flatten()) / res_norm_0
            if self.tol is not None and rel_res_norms[ii] < self.tol:
                break

            if self.data_term.lower() == 'kl':
                p += fp * sigma
                p = (1 + p - np.sqrt((p - 1) ** 2 + b_kl)) / 2
            else:
                p += res * sigma
                if self.data_term.lower() == 'l1':
                    p /= np.fmax(1, np.abs(p))
                elif self.data_term.lower() == 'l2':
                    p *= sigma1
                else:
                    raise ValueError("Unknown data term: %s" % self.data_term)

            q_g += self.laplacian(x_ehn) / (4 * len(self.axes))
            q_g /= np.fmax(1, np.abs(q_g))

            x_new = x - (At(p) + self.lambda_d2 * self.laplacian(q_g)) * tau

            if lower_limit is not None:
                x_new = np.fmax(x_new, lower_limit)
            if upper_limit is not None:
                x_new = np.fmin(x_new, upper_limit)

            x_ehn = x_new + (x_new - x)
            x = x_new

            if self.dump_file is not None:
                out_x[ii, ...] = x

            if self.verbose:
                print(('\b') * len(prnt_str), end='', flush=True)
                print((' ') * len(prnt_str), end='', flush=True)
                print(('\b') * len(prnt_str), end='', flush=True)

        if self.verbose:
            print("Done in %g seconds." % (tm.time() - c_in))

        if self.dump_file is not None:
            utils_io.save_field_toh5(self.dump_file, 'iterations', out_x)

        return (x, rel_res_norms)


class CP_wl(Solver):

    def __init__(
            self, verbose=False, weight_det=None, dump_file=None, tol=1e-5,
            data_term='l2', lambda_wl=1e-2, axes=None, wl_type='db4',
            use_decimated=False, decomp_lvl=3):
        Solver.__init__(self, verbose=verbose, dump_file=dump_file, tol=tol)
        self.weight_det = weight_det

        self.data_term = data_term
        self.lambda_wl = lambda_wl
        self.axes = axes

        self.wl_type = wl_type
        self.use_decimated = use_decimated
        self.decomp_lvl = decomp_lvl

    def initialize_wl_operators(self):
        if self.use_decimated:
            H = lambda x: pywt.wavedecn(x, wavelet=self.wl_type, axes=self.axes, level=self.decomp_lvl)
            Ht = lambda x: pywt.waverecn(x, wavelet=self.wl_type, axes=self.axes)
        else:
            if use_swtn:
                H = lambda x: pywt.swtn(x, wavelet=self.wl_type, axes=self.axes, level=self.decomp_lvl)
                Ht = lambda x: pywt.iswtn(x, wavelet=self.wl_type, axes=self.axes)
            else:
                H = lambda x: pywt.swt2(np.squeeze(x), wavelet=self.wl_type, axes=self.axes, level=self.decomp_lvl)
#                Ht = lambda x : pywt.iswt2(x, wavelet=self.wl_type)
                Ht = lambda x: pywt.iswt2(x, wavelet=self.wl_type)[np.newaxis, ...]
        return (H, Ht)

    def __call__(
            self, A, b, num_iter, x0=None, At=None, upper_limit=None, lower_limit=None):
        """ChambollePock implementization of wavelet regularized minimization.
        """
        if not has_pywt:
            raise ImportError('The module pywt was not found, please install it before you try using wavelet minimization')

        (A, At) = self.initialize_data_operators(A, At)
        (H, Ht) = self.initialize_wl_operators()

        data_type = b.dtype

        c_in = tm.time()

        tau = At(np.ones(b.shape, dtype=data_type))
        tau[np.abs(tau) < 1e-3] = 1
        if self.use_decimated and self.decomp_lvl is None:
            self.decomp_lvl = pywt.dwt_max_level(len(tau), pywt.Wavelet(self.wl_type).dec_len)
        tau = 1 / (tau + self.lambda_wl * (2 ** self.decomp_lvl))

        if self.weight_det is None:
            self.weight_det = A(np.ones(tau.shape, dtype=data_type))
        sigma = np.array(self.weight_det)
        sigma[np.abs(sigma) < 1e-3] = 1
        sigma = 1 / sigma

        sigma1 = 1 / (1 + sigma)

        if self.axes is None:
            self.axes = range(len(tau.shape))

        if x0 is None:
            x0 = np.zeros_like(tau)
        x = x0
        x_ehn = x
        p = np.zeros(b.shape, dtype=data_type)

        q_wl = H(np.zeros_like(x))

        res_norm_0 = npla.norm(b.flatten())
        rel_res_norms = np.empty((num_iter, ))

        if self.use_decimated or use_swtn:
            sigma2 = 1 / (2 ** np.arange(self.decomp_lvl, 0, -1))
        else:
            sigma2 = 1 / (2 ** np.arange(1, self.decomp_lvl+1))

        if self.dump_file is not None:
            out_x = np.empty(np.concatenate(((num_iter, ), x.shape)), dtype=x.dtype)

        if self.data_term.lower() == 'kl':
            b_kl = 4 * sigma * b

        c_init = tm.time()

        if self.verbose:
            print("- Performing CP-%s-%s iterations (init: %g seconds): " % (
                    self.data_term, self.wl_type.lower(), (c_init - c_in)), end='', flush=True)
        for ii in range(num_iter):
            if self.verbose:
                prnt_str = "%03d/%03d (avg: %g seconds)" % (ii, num_iter, (tm.time() - c_init) / np.fmax(ii, 1))
                print(prnt_str, end='', flush=True)

            fp = A(x_ehn)

            res = fp - b

            rel_res_norms[ii] = npla.norm(res.flatten()) / res_norm_0
            if self.tol is not None and rel_res_norms[ii] < self.tol:
                break

            if self.data_term.lower() == 'kl':
                p += fp * sigma
                p = (1 + p - np.sqrt((p - 1) ** 2 + b_kl)) / 2
            else:
                p += res * sigma
                if self.data_term.lower() == 'l1':
                    p /= np.fmax(1, np.abs(p))
                elif self.data_term.lower() == 'l2':
                    p *= sigma1
                else:
                    raise ValueError("Unknown data term: %s" % self.data_term)

            d = H(x_ehn)

            if self.use_decimated:
                q_wl[0] += d[0] * sigma2[0]
                q_wl[0] /= np.fmax(1, np.abs(q_wl[0]))
                for ii_l in range(self.decomp_lvl):
                    for k in q_wl[ii_l+1].keys():
                        q_wl[ii_l+1][k] += d[ii_l+1][k] * sigma2[ii_l]
                        q_wl[ii_l+1][k] /= np.fmax(1, np.abs(q_wl[ii_l+1][k]))
            else:
                for ii_l in range(self.decomp_lvl):
                    if use_swtn:
                        for k in q_wl[ii_l].keys():
                            q_wl[ii_l][k] += d[ii_l][k] * sigma2[ii_l]
                            q_wl[ii_l][k] /= np.fmax(1, np.abs(q_wl[ii_l][k]))
                    else:
                        q_wl[ii_l][0][:] += d[ii_l][0][:] * sigma2[ii_l]
                        q_wl[ii_l][0][:] /= np.fmax(1, np.abs(q_wl[ii_l][0][:]))
                        for ii_c in range(2 ** len(self.axes) - 1):
                            q_wl[ii_l][1][ii_c][:] += d[ii_l][1][ii_c][:] * sigma2[ii_l]
                            q_wl[ii_l][1][ii_c][:] /= np.fmax(1, np.abs(q_wl[ii_l][1][ii_c][:]))

            x_new = x - (At(p) + self.lambda_wl * Ht(q_wl)) * tau

            if lower_limit is not None:
                x_new = np.fmax(x_new, lower_limit)
            if upper_limit is not None:
                x_new = np.fmin(x_new, upper_limit)

            x_ehn = x_new + (x_new - x)
            x = x_new

            if self.dump_file is not None:
                out_x[ii, ...] = x

            if self.verbose:
                print(('\b') * len(prnt_str), end='', flush=True)
                print((' ') * len(prnt_str), end='', flush=True)
                print(('\b') * len(prnt_str), end='', flush=True)

        if self.verbose:
            print("Done in %g seconds." % (tm.time() - c_in))

        if self.dump_file is not None:
            utils_io.save_field_toh5(self.dump_file, 'iterations', out_x)

        return (x, rel_res_norms)


class Segment(Solver, Operations):

    def __init__(self, verbose=False, lambda_tv=1e-2, lambda_d2=1e-2, axes=None):
        Solver.__init__(self, verbose=verbose)
        Operations.__init__(self, axes)
        self.lambda_tv = lambda_tv
        self.lambda_d2 = lambda_d2

    def levelset(self, im, mus, iterations=50, w_norm_p=2, data_term='l1'):
        c_in = tm.time()

        mus = np.array(mus)
        mus_shape = np.concatenate((mus.shape, np.ones((len(im.shape), ), dtype=np.intp)))
        mus_exp = np.reshape(mus, mus_shape)
        W_prime = np.expand_dims(im, axis=0) - mus_exp
        W_prime = np.abs(W_prime) ** w_norm_p

        W_second = 1 / (W_prime + np.finfo(np.float32).eps * (W_prime == 0))
        W_mus = W_second / np.sum(W_second, axis=0)

        tau = np.sum(W_mus + (W_mus == 0), axis=0)
        sigma_mus = 1.0 / (W_mus + np.finfo(np.float32).eps * (W_mus == 0))
        sigma1_mus = 1.0 / (1.0 + sigma_mus)
        q_mus = np.zeros(np.concatenate(([mus.size], im.shape)))

        if self.lambda_tv is not None:
            tau += 2 * self.lambda_tv * len(im.shape)
            sigma_tv = 0.5
            q_tv = np.zeros(np.concatenate(([len(im.shape)], im.shape)))

        if self.lambda_d2 is not None:
            tau += 4 * self.lambda_d2 * len(im.shape)
            sigma_smooth = 1.0 / (4.0 * len(im.shape))
            q_l = np.zeros_like(im)

        tau = 1 / tau

        x = np.zeros_like(im)
        x[:] = mus[np.argmax(W_mus, axis=0)]
        xe = x

        c_init = tm.time()

        if self.verbose:
            print("- Performing CP-%s iterations (init: %g seconds): " % (data_term.lower(), (c_init - c_in)),
                  end='', flush=True)
        for ii in range(iterations):
            if self.verbose:
                prnt_str = "%03d/%03d (avg: %g seconds)" % (ii, iterations, (tm.time() - c_init) / np.fmax(ii, 1))
                print(prnt_str, end='', flush=True)

            q_mus += mus_exp - xe
            if data_term.lower() == 'l1':
                q_mus /= np.fmax(1, np.abs(q_mus))
            elif data_term.lower() == 'l12':
                q_mus /= np.fmax(1, np.sqrt(np.sum(q_mus ** 2, axis=0)))
            elif data_term.lower() == 'l2':
                q_mus *= sigma1_mus

            upd = np.sum(W_mus * q_mus, axis=0)

            if self.lambda_tv is not None:
                q_tv += np.stack(self.gradient(xe)) * sigma_tv
                q_tv /= np.fmax(1, np.sqrt(np.sum(q_tv ** 2, axis=0)))

                upd += self.lambda_tv * self.divergence(q_tv)

            if self.lambda_d2 is not None:
                q_l += self.laplacian(xe) * sigma_smooth
                q_l /= np.fmax(1, np.abs(q_l))

                upd -= self.lambda_d2 * self.laplacian(q_l)

            xn = x + tau * upd

            xe = xn + (xn - x)
            x = xn

            if self.verbose:
                print(('\b') * len(prnt_str), end='', flush=True)
                print((' ') * len(prnt_str), end='', flush=True)
                print(('\b') * len(prnt_str), end='', flush=True)

        if self.verbose:
            print("Done in %g seconds." % (tm.time() - c_in))

        return x

    def simple(self, vol, rhos):
        rhos = np.array(rhos)
        pos = np.argsort(rhos)
        rhos = rhos[pos]
        thr = rhos[0:-1] + np.diff(rhos) / 2

        x = np.zeros_like(vol)
        for ii, t in enumerate(thr):
            x[vol > t] = pos[ii+1]
        return x
