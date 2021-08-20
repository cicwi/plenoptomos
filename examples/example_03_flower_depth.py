#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example demonstrating refocusing capabilities.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France

Created on Tue Mar 19 12:29:02 2019
"""

import numpy as np
import matplotlib.pyplot as plt

import os

import plenoptomos as pleno

data_dir = "examples/data/"
dpath = os.path.join(data_dir, "flowers_plants_30_eslf.png")
jpath = os.path.join(data_dir, "flowers_plants_30.json")

error_msg_lf_files = (
    "Please download the following files from the Stanford light-field archive and put them in %s:\n - %s\n - %s"
)

if not os.path.exists(dpath):
    print('Example files: "%s" and "%s" do not exist.' % (dpath, jpath))
    dpath_url = "http://lightfields.stanford.edu/images/flowers_plants/raw/flowers_plants_30_eslf.png"
    jpath_url = "http://lightfields.stanford.edu/images/flowers_plants/metadata/flowers_plants_30.json"
    try:
        import urllib.request as ur

        print("They will be now downloaded:\n - %s" % dpath_url)
        ur.urlretrieve(dpath_url, dpath)
        print(" - %s" % jpath_url)
        ur.urlretrieve(jpath_url, jpath)
    except ImportError:
        raise ValueError(error_msg_lf_files % (data_dir, dpath_url, jpath_url))

print("Importing the light-field from the Lytro eslf format..")
lf = pleno.import_lf.from_lytro(dpath, jpath, source="eslf")

z0 = lf.camera.get_focused_distance()

print("Computing refocusing distances..")
alphas_con = np.linspace(0.5, 3.0, 46)
alphas_par = lf.camera.get_alphas(alphas_con, beam_geometry_in="cone", beam_geometry_out="parallel")
z0s = z0 * alphas_par

# we choose the 3 most interesting ones, to display what they look like
dists = [6, 10, 21]
refocused_imgs = pleno.tomo.compute_refocus_iterative(lf, z0s[np.r_[dists]], beam_geometry="parallel", iterations=3)

print("Computing depth cues..")
dc = pleno.depth.compute_depth_cues(lf, z0s)

print("Using depth cues to generate a depth-map..")
dm = pleno.depth.compute_depth_map(dc, lambda_tv=1.0, lambda_d2=None)

(f, axs) = plt.subplots(3, 3, sharex=True, sharey=True)
axs[0, 0].imshow(dc["depth_defocus"], vmin=0, vmax=len(z0s) - 1)
axs[0, 0].set_title("Depth from defocus")
axs[1, 0].imshow(dc["confidence_defocus"])
axs[1, 0].set_title("Confidence of defocus")

axs[0, 1].imshow(dc["depth_correspondence"], vmin=0, vmax=len(z0s) - 1)
axs[0, 1].set_title("Depth from correspondence")
axs[1, 1].imshow(dc["confidence_correspondence"])
axs[1, 1].set_title("Confidence of correspondence")

axs[0, 2].imshow(dm, vmin=0, vmax=len(z0s) - 1)
axs[0, 2].set_title("Depth map - DC")
axs[1, 2].imshow(lf.get_photograph())
axs[1, 2].set_title("Acquisition focus")

for ii, d in enumerate(dists):
    axs[2, ii].imshow(refocused_imgs[ii, ...])
    axs[2, ii].set_title("Image %d" % d)

plt.tight_layout()
plt.show(block=False)
