# -*- coding: utf-8 -*-
"""
Simple visualization tools for refocused stacks.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France

Created on Fri Dec  8 15:24:12 2017
"""

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as mcm


class Scroller(object):
    """Class to be used in conjunction with RefocusVisualizer."""

    def __init__(
            self, f, ax, data, pos_val=None, title=None, do_clear=True, clims=None, cmap=mcm.viridis, colorbar=True):
        self.f = f
        self.ax = ax
        self.title = title
        self.pos_val = pos_val
        self.data = data
        self.do_clear = do_clear
        self.clims = clims
        self.cmap = cmap
        self.colorbar = colorbar

        self.depth_stack = self.data[0].shape[0]
        self.num_stacks = len(self.data)
        self.curr_pos = 0

        if isinstance(self.clims, (list, tuple)):
            if len(self.clims) > 1:
                self.clims_is_global = False
            else:
                self.clims_is_global = True
                self.clims = self.clims[0]
        else:
            self.clims_is_global = True

        if self.title is None:
            self.title = 'index %d/%d'
            if self.pos_val is not None:
                self.title += '\n(distance: %g)'

        self.f.canvas.mpl_connect('key_press_event', self._key_event)
        self.f.canvas.mpl_connect('scroll_event', self._scroll_event)
        self.f.canvas.mpl_connect('button_press_event', self._button_event)
        self.update()

    def _key_event(self, e):
        if e.key == 'right':
            self.curr_pos += 1
        elif e.key == 'left':
            self.curr_pos -= 1
        elif e.key == 'up':
            self.curr_pos += 1
        elif e.key == 'down':
            self.curr_pos -= 1
        elif e.key == 'pageup':
            self.curr_pos += 10
        elif e.key == 'pagedown':
            self.curr_pos -= 10
        elif e.key == 'escape':
            plt.close(self.f)
        else:
            print(e.key)
            return

        self.curr_pos = self.curr_pos % self.depth_stack
        self.update()

    def _scroll_event(self, e):
        if e.button == 'up':
            self.curr_pos += 1
        elif e.button == 'down':
            self.curr_pos -= 1
        else:
            print(e.key)
            return

        self.curr_pos = self.curr_pos % self.depth_stack
        self.update()

    def _button_event(self, e):
        print(e)

    def update(self):
        """Updates the axes, to show the new requested image."""
        if isinstance(self.ax, np.ndarray):
            for ii in range(self.num_stacks):
                update_cb = self.colorbar and (not self.clims_is_global or ii == (self.num_stacks - 1))
                im = self._update_ax(self.ax[ii], self.data[ii], update_cb)
                self.ax[ii].set_title(
                    self.title % (self.curr_pos+1, len(self.pos_val[ii]), self.pos_val[ii][self.curr_pos]))
                if self.clims is not None:
                    if self.clims_is_global:
                        im.set_clim(self.clims[0], self.clims[1])
                    else:
                        im.set_clim(self.clims[ii][0], self.clims[ii][1])
        else:
            im = self._update_ax(self.ax, self.data[0], self.colorbar)
            self.ax.set_title(self.title % (
                self.curr_pos+1, self.depth_stack, self.pos_val[0][self.curr_pos]))
            if self.clims is not None:
                if self.clims_is_global:
                    im.set_clim(self.clims[0], self.clims[1])
                else:
                    im.set_clim(self.clims[0][0], self.clims[0][1])
        self.f.canvas.draw()

    def _update_ax(self, ax, data, update_cb):
        if self.do_clear:
            for im in ax.get_images():
                if update_cb:
                    im.colorbar.remove()
                im.remove()
            ax.cla()
            im = ax.imshow(data[self.curr_pos, ...], cmap=self.cmap)
            if update_cb:
                plt.colorbar(im, ax=ax)
        else:
            try:
                im = ax.get_images()[0]
                im.set_data(data[self.curr_pos, ...])
            except IndexError:
                im = ax.imshow(data[self.curr_pos, ...], cmap=self.cmap)
                if update_cb:
                    plt.colorbar(im, ax=ax)
        return im


class RefocusVisualizer(object):
    """Simple visualization tool for refocusing stacks."""

    def __init__(
            self, data, pos_val=None, ref_img=None, transpose_axes=None,
            share_axes=False, clims=None, do_clear=True, cmap=mcm.viridis,
            colorbar=True):
        """Visualization tool for refocusing stacks.

        :param data: Refocus stacks
        :type data: `numpy.array_like` or list / tuple of `numpy.array_like`
        :param pos_val: Refocusing distance associated to each image in the stack, defaults to None
        :type pos_val: `numpy.array_like` or list / tuple of `numpy.array_like`, optional
        :param ref_img: Reference image to compare with refocusing stacks, defaults to None
        :type ref_img: `numpy.array_like`, optional
        :param transpose_axes: Requested axes order, defaults to None
        :type transpose_axes: tuple, optional
        :param share_axes: Whether to share axes, defaults to False
        :type share_axes: boolean, optional
        :param clims: Color limits, defaults to None
        :type clims: tuple(float, float), optional
        :param do_clear: Whether to clear axes at each update, defaults to True
        :type do_clear: boolean, optional
        :param cmap: Color map for visualization of the images, defaults to mcm.viridis
        :type cmap: `matplotlib.cm`, optional
        :param colorbar: Whether to show a colorbar or not, defaults to True
        :type colorbar: boolean, optional

        :raises ValueError: Errors produce in case of mismatching data sizes.
        """
        self.data = data
        self.pos_val = pos_val
        self.ref_img = ref_img

        if isinstance(self.data, (list, tuple)):
            num_data_stacks = len(self.data)
        elif isinstance(self.data, np.ndarray):
            if len(self.data.shape) == 2:
                self.data = self.data[None, ...]

            if len(self.data.shape) == 3:
                num_data_stacks = 1
                self.data = [self.data]
            elif len(self.data.shape) == 4:
                num_data_stacks = 1
                self.data = [self.data[ii, ...] for ii in range(self.data.shape[0])]
            else:
                raise ValueError(('What is the incoming data?', self.data))
        else:
            raise ValueError(('What is the incoming data?', self.data))

        print('Found %d data stacks' % num_data_stacks)
        len_stack = self.data[0].shape[0]
        print('- Stack size: %d' % len_stack)
        print('- Data shape: %s' % np.array(self.data[0].shape[1:]))

        if self.pos_val is None:
            self.pos_val = (np.zeros((len_stack, )), ) * num_data_stacks
        elif isinstance(self.pos_val, np.ndarray):
            self.pos_val = [self.pos_val] * num_data_stacks
        elif len(self.pos_val) == 1 and isinstance(self.pos_val[0], np.ndarray):
            self.pos_val = self.pos_val * num_data_stacks
        elif not len(self.pos_val) == num_data_stacks:
            raise ValueError('The positions arrays should be as many as the data arrays, or a common one for all')

        if transpose_axes is not None:
            if self.ref_img is not None:
                self.ref_img = self.ref_img.transpose(np.array(transpose_axes[1:]) - 1)
            for ii in range(num_data_stacks):
                self.data[ii] = self.data[ii].transpose(transpose_axes)

        if clims is None:
            clims = [[x.min(), x.max()] for x in self.data]
        elif isinstance(clims, str):
            if clims.lower() == 'global':
                clims = [[x.min(), x.max()] for x in self.data]
                clims = np.stack(clims, axis=0)
                clims = np.array([np.min(clims[:, 0]), np.max(clims[:, 1])])

        tot_plots = num_data_stacks + int(self.ref_img is not None)
        self.f = plt.figure()
        self.ax = self.f.subplots(1, tot_plots, sharex=share_axes, sharey=share_axes)

        if self.ref_img is not None:
            self.ax[0].imshow(self.ref_img)
            self.scroller = Scroller(
                self.f, self.ax[1:], data=self.data, pos_val=self.pos_val, clims=clims, do_clear=do_clear, cmap=cmap)
        else:
            self.scroller = Scroller(
                self.f, self.ax, data=self.data, pos_val=self.pos_val, clims=clims, do_clear=do_clear, cmap=cmap)
        self.f.set_tight_layout(True)

    def show(self, *args, **kwords):
        plt.show(*args, **kwords)
