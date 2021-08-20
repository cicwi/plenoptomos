#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 15:24:12 2017

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

import os

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mim
import matplotlib.cm as mcm


class ShiftsFinderVolume3D(object):

    ax_ud = [1, 0, 0]
    ax_lr = [2, 2, 1]
    planes_vis = ["XY", "XZ", "YZ"]

    axs_rot = ["Z", "Y", "X"]

    def __init__(self, data_fix, data_move, do_clear=True, clims=None):
        self.data_fix = data_fix
        self.data_move = data_move
        self.do_clear = do_clear
        self.clims = clims
        self.ref_pos = (np.array(data_fix.shape) / 2).astype(np.intp)

        self.shift_vol = [0, 0, 0]
        self.ax_perp = 0
        self.title = "Plane: %s, slice %d, rot angle: %g\nShift: x %d y %d z %d"

        self.f, self.ax = plt.subplots(2, 2, sharex=True, sharey=True)
        self.ax[0, 0].set_title("Sum")
        self.ax[0, 1].set_title("Difference")
        self.ax[1, 0].set_title("Fix")
        self.ax[1, 1].set_title("Move")

        self.f.canvas.mpl_connect("key_press_event", self._key_event)
        self.update("")

    def _key_event(self, e):
        direction = "ud"
        if e.key == "up":
            self.shift_vol[self.ax_ud[self.ax_perp]] -= 1
        elif e.key == "down":
            self.shift_vol[self.ax_ud[self.ax_perp]] += 1
        elif e.key == "left":
            self.shift_vol[self.ax_lr[self.ax_perp]] -= 1
            direction = "lr"
        elif e.key == "right":
            self.shift_vol[self.ax_lr[self.ax_perp]] += 1
            direction = "lr"
        elif e.key == "pageup":
            self.shift_vol[self.ax_ud[self.ax_perp]] -= 10
        elif e.key == "pagedown":
            self.shift_vol[self.ax_ud[self.ax_perp]] += 10
        elif e.key == "z":
            self.ax_perp = 0
        elif e.key == "y":
            self.ax_perp = 1
        elif e.key == "x":
            self.ax_perp = 2
        elif e.key == "escape":
            plt.close(self.f)
        else:
            print(e.key)
            return

        self.update(direction)

    def update(self, direction):
        slices_fix = [slice(None)] * 3
        slices_fix[self.ax_perp] = slice(self.ref_pos[self.ax_perp], self.ref_pos[self.ax_perp] + 1)
        slices_mov = [slice(None)] * 3
        slices_mov[self.ax_perp] = slice(
            self.ref_pos[self.ax_perp] - self.shift_vol[self.ax_perp],
            self.ref_pos[self.ax_perp] - self.shift_vol[self.ax_perp] + 1,
        )

        slice_fix = np.squeeze(self.data_fix[slices_fix])

        slice_mov = self.data_move[slices_mov]
        slice_mov = np.roll(slice_mov, axis=self.ax_ud[self.ax_perp], shift=self.shift_vol[self.ax_ud[self.ax_perp]])
        slice_mov = np.roll(slice_mov, axis=self.ax_lr[self.ax_perp], shift=self.shift_vol[self.ax_lr[self.ax_perp]])
        slice_mov = np.squeeze(slice_mov)

        slice_sum = (slice_fix + slice_mov) / 2
        slice_dif = (slice_fix - slice_mov) / 2

        self.update_ax(self.ax[0, 0], slice_sum)
        self.update_ax(self.ax[0, 1], slice_dif)
        self.update_ax(self.ax[1, 0], slice_fix)
        self.update_ax(self.ax[1, 1], slice_mov)

        self.f.suptitle(
            self.title
            % (
                self.planes_vis[self.ax_perp],
                self.ref_pos[self.ax_perp],
                0.0,
                self.shift_vol[2],
                self.shift_vol[1],
                self.shift_vol[0],
            )
        )
        self.f.canvas.draw()

    def update_ax(self, ax, data):
        if self.do_clear:
            ax.cla()
            for im in ax.get_images():
                im.remove()
            im = ax.imshow(data)
        else:
            try:
                im = ax.get_images()[0]
                im.set_data(data)
            except IndexError:
                im = ax.imshow(data)
        if self.clims is not None:
            im.set_clim(self.clims[0], self.clims[1])
        return im

    def show(self, *args, **kwords):
        plt.show(*args, **kwords)


class MovieCreator(object):
    def __init__(self, out_dir, lf, z0s, focal_stack, dmap, all_focus, use_overlay=True):
        self.out_dir = out_dir

        self.lf = lf
        self.z0s = z0s
        self.focal_stack = focal_stack
        self.dmap = dmap
        self.all_focus = all_focus

        self.use_overlay = use_overlay
        self.counter = 0
        self.last_frame = None

    def to_01(self, img):
        return (img - np.min(img)) / (np.max(img) - np.min(img))

    def render_text(self, img_size, t):
        f = plt.figure(None, frameon=False, figsize=np.array(img_size) / 75, dpi=75)
        ax = f.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, img_size[0])
        ax.set_ylim(img_size[1], 0)
        ax.set_axis_off()

        border = 8
        ax.text(border, img_size[1] - border, t, fontdict=dict(fontsize=12), va="bottom", ha="left")
        f.canvas.draw()
        (w, h) = f.canvas.get_width_height()
        buf = np.fromstring(f.canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w, h, 4)
        buf = np.sum(buf[..., 1:], axis=2)
        buf = self.to_01(buf)
        return 1 - buf

    def render_distance(self, img_size, z):
        return self.render_text(img_size, r"$z_0 = %g$ mm" % z)

    def create_sublf_frame(self, sublf_size, sublf_center_vu=None):
        lf_center_vu = np.array((self.lf.camera.data_size_vu - 1) / 2)

        if sublf_center_vu is None:
            sublf_center_vu = lf_center_vu.astype(np.intp)
        elif sublf_center_vu[0] is None:
            sublf_center_vu[0] = int(lf_center_vu[0])
        elif sublf_center_vu[1] is None:
            sublf_center_vu[1] = int(lf_center_vu[1])

        lf_sub = self.lf.get_sub_lightfield(
            sub_aperture_size=sublf_size, center_v=sublf_center_vu[0], center_u=sublf_center_vu[1]
        )
        img = lf_sub.get_photograph()
        if self.use_overlay:
            pos_vu = (sublf_center_vu - lf_center_vu) * self.lf.camera.pixel_size_vu
            t = r"$v = %g$ mm" % pos_vu[0] + "\n" + r"$u = %g$ mm" % pos_vu[1]
            imt = self.render_text(img.shape, t)
            img = self.overlay_text_on_image(img, imt)
        self.create_simple_frame(img)

    def overlay_text_on_image(self, img_ref, img_text):
        text_mask = img_text > 0.5
        rest_mask = np.logical_not(text_mask).astype(img_ref.dtype)
        text_mask = text_mask.astype(img_ref.dtype)
        return img_ref * rest_mask + text_mask * np.max(img_ref)

    def create_refocus_frame(self, z_ind):
        z0 = np.array((self.z0s[z_ind],))
        img_ref = self.focal_stack[z_ind, ...]
        if self.use_overlay:
            img_z = self.render_distance(img_ref.shape, z0)
            img_ref = self.overlay_text_on_image(img_ref, img_z)
        self.create_simple_frame(self.to_01(img_ref))

    def create_simple_frame(self, img):
        img_name = os.path.join(self.out_dir, "frame_%03d.png" % self.counter)
        print("Creating frame: %d, in: %s" % (self.counter, img_name))
        mim.imsave(img_name, np.squeeze(img))
        self.counter += 1
        self.last_frame = img

    def create_transition_frame(self, from_img, to_img, percentage):
        center_img = (np.array(from_img.shape) - 1) / 2
        xx = np.linspace(-center_img[0], center_img[0], from_img.shape[0])
        yy = np.linspace(-center_img[1], center_img[1], from_img.shape[1])
        [xx, yy] = np.meshgrid(xx, yy, indexing="ij")
        dist = np.sqrt(xx ** 2 + yy ** 2)
        radius_max = np.sqrt(np.sum(center_img ** 2))
        from_mask = dist > (percentage * radius_max)
        to_mask = dist < (percentage * radius_max)
        composite_img = self.to_01(from_img) * from_mask + self.to_01(to_img) * to_mask
        self.create_simple_frame(composite_img)

    def create_static_frames(self, num_frames):
        for ii in range(num_frames):
            self.create_simple_frame(self.last_frame)

    def create_transition(self, from_img, to_img, num_frames):
        transitions_steps = np.squeeze(np.array([range(num_frames)]) / num_frames)
        for ts in transitions_steps:
            self.create_transition_frame(from_img, to_img, ts)

    def create_transition_sublf(self, final_sublf_size, direction="to"):
        half_sub_aperture_size = int((np.min(self.lf.camera.data_size_vu) - 1) / 2)
        sub_lightfields_range = range(final_sublf_size, half_sub_aperture_size)
        if direction.lower() == "to":
            sub_lightfields_range = reversed(sub_lightfields_range)
        for ii in sub_lightfields_range:
            self.create_sublf_frame(ii)

    def create_dmap_frames(self, num_frames=1):
        if self.use_overlay:
            text_frame = self.render_text(self.dmap.shape, "Depth-map")
            out_frame = self.overlay_text_on_image(self.dmap, text_frame)
        else:
            out_frame = self.dmap
        self.create_simple_frame(out_frame)
        self.create_static_frames(num_frames - 1)

    def create_allfocus_frames(self, num_frames=1):
        if self.use_overlay:
            text_frame = self.render_text(self.all_focus.shape, "All-in-focus")
            out_frame = self.overlay_text_on_image(self.all_focus, text_frame)
        else:
            out_frame = self.all_focus
        self.create_simple_frame(out_frame)
        self.create_static_frames(num_frames - 1)
