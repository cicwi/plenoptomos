#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This example illustrates the concepts discussed in:

* N. Viganò, et al., “Tomographic approach for the quantitative scene
reconstruction from light field images,” Opt. Express, vol. 26, no. 18,
p. 22574, Sep. 2018.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France

Created on Tue Jan 22 15:42:25 2019
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mim

import plenoptomos as pleno

print("Setting up the known volume..")
cwi_img = mim.imread("examples/data/cwi_logo_small.png")
vox_img = mim.imread("examples/data/voxel_logo_small.png")

alphas_ph = np.array((1.1, 1, 0.90))
vols_shape = (alphas_ph.size, 256, 512)

masks = np.zeros(vols_shape, dtype=np.float32)
masks[2, 136 : cwi_img.shape[0] + 136, 244 : cwi_img.shape[1] + 244] = cwi_img[:, :, 3]
masks[1, 0 : vox_img.shape[0], 0 : vox_img.shape[1]] = vox_img[:, :, 3]
masks[0, 136 : cwi_img.shape[0] + 136, 0 : cwi_img.shape[1]] = np.rot90(cwi_img[:, :, 3], k=2)

vols_r = np.zeros(vols_shape, dtype=np.float32)
vols_r[2, 136 : cwi_img.shape[0] + 136, 244 : cwi_img.shape[1] + 244] = cwi_img[:, :, 0] * (cwi_img[:, :, 3] > 0)
vols_r[1, 0 : vox_img.shape[0], 0 : vox_img.shape[1]] = vox_img[:, :, 0]
vols_r[0, 136 : cwi_img.shape[0] + 136, 0 : cwi_img.shape[1]] = np.rot90(cwi_img[:, :, 0] * (cwi_img[:, :, 3] > 0), k=2)

vols_g = np.zeros(vols_shape, dtype=np.float32)
vols_g[2, 136 : cwi_img.shape[0] + 136, 244 : cwi_img.shape[1] + 244] = cwi_img[:, :, 1] * (cwi_img[:, :, 3] > 0)
vols_g[1, 0 : vox_img.shape[0], 0 : vox_img.shape[1]] = vox_img[:, :, 1]
vols_g[0, 136 : cwi_img.shape[0] + 136, 0 : cwi_img.shape[1]] = np.rot90(cwi_img[:, :, 1] * (cwi_img[:, :, 3] > 0), k=2)

vols_b = np.zeros(vols_shape, dtype=np.float32)
vols_b[2, 136 : cwi_img.shape[0] + 136, 244 : cwi_img.shape[1] + 244] = cwi_img[:, :, 2] * (cwi_img[:, :, 3] > 0)
vols_b[1, 0 : vox_img.shape[0], 0 : vox_img.shape[1]] = vox_img[:, :, 2]
vols_b[0, 136 : cwi_img.shape[0] + 136, 0 : cwi_img.shape[1]] = np.rot90(cwi_img[:, :, 2] * (cwi_img[:, :, 3] > 0), k=2)

print("Setting up the camera structure (containing light-field metadata)..")
camera = pleno.lightfield.get_camera("synthetic")

z0 = camera.get_focused_distance()
z0s_ph = z0 * alphas_ph  # The layer positions in the phantom

# adding some border to avoid the single images falling out of the sub-aperture images
border = 40
camera.data_size_ts += 2 * border

camera_r = camera.clone()
camera_g = camera.clone()
camera_b = camera.clone()

(camera_r.wavelength_range, camera_g.wavelength_range, camera_b.wavelength_range) = pleno.colors.get_rgb_wavelengths()

masks_pad = np.pad(masks, ((0,), (border,), (border,)), mode="constant")
vols_r_pad = np.pad(vols_r, ((0,), (border,), (border,)), mode="constant")
vols_g_pad = np.pad(vols_g, ((0,), (border,), (border,)), mode="constant")
vols_b_pad = np.pad(vols_b, ((0,), (border,), (border,)), mode="constant")

print("Creating the synthetic light-field: forward projecting the volume to the sub-aperture images..")
lf_r = pleno.tomo.compute_forwardprojection(camera_r, z0s_ph, vols_r_pad, masks_pad)
lf_g = pleno.tomo.compute_forwardprojection(camera_g, z0s_ph, vols_g_pad, masks_pad)
lf_b = pleno.tomo.compute_forwardprojection(camera_b, z0s_ph, vols_b_pad, masks_pad)

print("\nPerforming refocusing...\n")
# The conversions are done using the formulas from:
# [1] N. Viganò, et al., “Tomographic approach for the quantitative scene reconstruction from light field images,”
# Opt. Express, vol. 26, no. 18, p. 22574, Sep. 2018.

M = z0 / camera.z1

# Cone beam / object space position of one of the CWI logos
alphas_z0_con = np.array((0.9,))
z0s_con = z0 * alphas_z0_con

# Cone beam / image space position of one of the CWI logos
alphas_z1_con = M * alphas_z0_con / (1 - alphas_z0_con * (1 - M))
z1s_con = camera.z1 * alphas_z1_con

# Parallel beam / object space position of one of the CWI logos
alphas_z0_par = 2 - 1 / alphas_z0_con
z0s_par = z0 * alphas_z0_par

print("Cone beam / object space position of one of the CWI logo. Alphas:", alphas_z0_con, "Distances:", z0s_con)
print("Cone beam / image space position of one of the CWI logos. Alphas:", alphas_z1_con, "Distances:", z1s_con)
print("Parallel beam / object space position of one of the CWI logos. Alphas:", alphas_z0_par, "Distances:", z0s_par)

print("\nIntegration refocusing example (Cone beam, object space)...\n")
refocused_int_r = pleno.refocus.compute_refocus_integration(lf_r, z0s_con, beam_geometry="cone", domain="object")
refocused_int_g = pleno.refocus.compute_refocus_integration(lf_g, z0s_con, beam_geometry="cone", domain="object")
refocused_int_b = pleno.refocus.compute_refocus_integration(lf_b, z0s_con, beam_geometry="cone", domain="object")
refocused_int = pleno.colors.merge_rgb_images(refocused_int_r, refocused_int_g, refocused_int_b)

print("\nBack-projection refocusing example (Cone beam, object space)...\n")
refocused_bpj_r = pleno.tomo.compute_refocus_backprojection(lf_r, z0s_con, beam_geometry="cone", domain="object")
refocused_bpj_g = pleno.tomo.compute_refocus_backprojection(lf_g, z0s_con, beam_geometry="cone", domain="object")
refocused_bpj_b = pleno.tomo.compute_refocus_backprojection(lf_b, z0s_con, beam_geometry="cone", domain="object")
refocused_bpj_co = pleno.colors.merge_rgb_images(refocused_bpj_r, refocused_bpj_g, refocused_bpj_b)

print("\nBack-projection refocusing example (Cone beam, image space)...\n")
refocused_bpj_r = pleno.tomo.compute_refocus_backprojection(lf_r, z1s_con, beam_geometry="cone", domain="image")
refocused_bpj_g = pleno.tomo.compute_refocus_backprojection(lf_g, z1s_con, beam_geometry="cone", domain="image")
refocused_bpj_b = pleno.tomo.compute_refocus_backprojection(lf_b, z1s_con, beam_geometry="cone", domain="image")
refocused_bpj_ci = pleno.colors.merge_rgb_images(refocused_bpj_r, refocused_bpj_g, refocused_bpj_b)

print("\nBack-projection refocusing example (Parallel beam, object space)...\n")
refocused_bpj_r = pleno.tomo.compute_refocus_backprojection(lf_r, z0s_par, beam_geometry="parallel", domain="object")
refocused_bpj_g = pleno.tomo.compute_refocus_backprojection(lf_g, z0s_par, beam_geometry="parallel", domain="object")
refocused_bpj_b = pleno.tomo.compute_refocus_backprojection(lf_b, z0s_par, beam_geometry="parallel", domain="object")
refocused_bpj_po = pleno.colors.merge_rgb_images(refocused_bpj_r, refocused_bpj_g, refocused_bpj_b)

f, ax = plt.subplots(2, 2, sharex=True, sharey=True)
ax[0, 0].imshow(np.squeeze(refocused_int))
ax[0, 0].set_title("Integration")
ax[0, 1].imshow(np.squeeze(refocused_bpj_co))
ax[0, 1].set_title("Back-projection - Cone, Object")
ax[1, 0].imshow(np.squeeze(refocused_bpj_ci))
ax[1, 0].set_title("Back-projection - Cone, Image")
ax[1, 1].imshow(np.squeeze(refocused_bpj_po))
ax[1, 1].set_title("Back-projection - Parallel, Object")
f.tight_layout()
plt.show()
