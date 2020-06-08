#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 18:07:48 2018

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

import numpy as np


class GeometryTransformation(object):
    """Geometry transformation class. It allows to manipulate position and direction vectors.
    """

    def __init__(self, t=np.eye(3, dtype=np.float32), s=np.zeros((3, ), dtype=np.float32)):
        self.t = t  # The transformation
        self.s = s  # The shift (translation)

    def __mul__(self, other):
        """Implement multiplication operation.

        :param other: Either another transformation to compose or a vector.
        :type other: Either `GeometryTransformation` or `numpy.array_like`

        :return: Result of the multiplication.
        :rtype: Either `GeometryTransformation` or `numpy.array_like`
        """
        if isinstance(other, GeometryTransformation):
            new_t = np.dot(self.t, other.t)
            new_s = self.s + np.dot(self.t, other.s)
            return GeometryTransformation(t=new_t, s=new_s)
        else:
            return self.matvec(other)

    def apply_position(self, vec):
        """Implements application of the transformation to a position vector.

        :param vec: Position vector
        :type vec: `numpy.array_like`

        :return: Result of the transformation
        :rtype: `numpy.array_like`
        """
        return np.dot(self.t, vec) + self.s[..., np.newaxis]

    def apply_direction(self, vec):
        """Implements application of the transformation to a direction vector.

        :param vec: Direction vector
        :type vec: `numpy.array_like`

        :return: Result of the transformation
        :rtype: `numpy.array_like`
        """
        return np.dot(self.t, vec)

    def _matvec(self, vec):
        """Implement transformation to vector operation.

        :param vec: Either position or direction vectors.
        :type vec: numpy.array_like <3, > or <4, >
        """
        if vec.shape[0] == 3:
            return self.apply_direction(vec)
        elif vec.shape[0] == 4:
            if vec[3] == 0:
                return self.apply_direction(vec[:3, ...])
            elif vec[3] == 1:
                return self.apply_position(vec[:3, ...])
            else:
                raise ValueError('4th component of 4-vectors can only be 1 or 0, while %d was found' % vec[3])
        else:
            raise ValueError('This function only accepts 3 and 4-vectors, while %d-vector was found' % vec.shape[0])

    def rmatvec(self, vec):
        """Implement the transpose of the transformation to vector operation.

        :param vec: Direction vectors.
        :type vec: numpy.array_like <3, > or <4, >
        """
        if vec.shape[0] == 3:
            return self.apply_direction(vec)
        elif vec.shape[0] == 4:
            if vec[3] == 0:
                return self.apply_direction(vec[:3, ...])
            else:
                raise ValueError('4th component of 4-vectors can only be 0 (direction), while %d was found' % vec[3])
        else:
            raise ValueError('This function only accepts 3-vectors, while %d-vector was found' % vec.shape[0])

    @staticmethod
    def get_rotation_components(rot_axis):
        """Computes the rotation components for the given rotation axis.

        :param rot_axis: The rotation axis.
        :type rot_axis: <3, > numpy.array_like

        :return: The rotation components
        :rtype: tuple(<3, 3> numpy.array_like, <3, 3> numpy.array_like, <3, 3> numpy.array_like)
        """
        rot_axis = np.array(rot_axis, dtype=np.float32)
        rot_axis /= np.sqrt(np.dot(rot_axis, rot_axis))

        r_comp_const = np.outer(rot_axis, rot_axis)
        r_comp_cos = np.eye(3, dtype=np.float32) - r_comp_const
        r_comp_sin = np.zeros((3, 3), dtype=np.float32)
        r_comp_sin[[0, 0, 1], [1, 2, 2]] = [-rot_axis[2],  rot_axis[1], -rot_axis[0]]
        r_comp_sin -= r_comp_sin.transpose()

        return (r_comp_const, r_comp_cos, r_comp_sin)

    @staticmethod
    def get_rotation_matrix(rotcomp, angle):
        """Computes the rotation matrix for the given rotation components and angle.

        :param rotcomp: The rotation components, computed with `get_rotation_components`
        :type rotcomp: tuple(<3, 3> numpy.array_like, <3, 3> numpy.array_like, <3, 3> numpy.array_like)
        :param angle: The rotation angle
        :type angle: float

        :return: The rotation matrix
        :rtype: <3, 3> `numpy.array_like`
        """
        cos_as = np.cos(angle)
        sin_as = np.sin(angle)

        (r_comp_const, r_comp_cos, r_comp_sin) = rotcomp

        return r_comp_const + cos_as * r_comp_cos + sin_as * r_comp_sin

    @staticmethod
    def get_axis_angle(vec_ref, vec_dir):
        """Computes the rotation axis and angle of the rotation that transformed vec_ref into vec_dir.

        :param vec_ref: The initial vector.
        :type vec_ref: `numpy.array_like`
        :param vec_dir: The result vector.
        :type vec_dir: `numpy.array_like`

        :return: The normalized rotation axis and angle.
        :rtype: tuple(`numpy.array_like`, float)
        """
        vec_ref = np.array(vec_ref, dtype=np.float32)
        vec_dir = np.array(vec_dir, dtype=np.float32)
        vec_ref /= np.sqrt(np.dot(vec_ref, vec_ref))
        vec_dir /= np.sqrt(np.dot(vec_dir, vec_dir))

        vec_axis = np.cross(vec_ref, vec_dir)
        vec_axis_norm = np.sqrt(np.dot(vec_axis, vec_axis))
        vec_angle = np.arccos(np.dot(vec_ref, vec_dir))

        if vec_axis_norm < 1e-5:
            return None
        vec_axis /= vec_axis_norm

        return (vec_axis, vec_angle)

    @staticmethod
    def get_translation(point):
        """Create a transflation transformation.

        :param point: Translation position.
        :type point: `numpy.array_like`

        :return: The translation transformation
        :rtype: `GeometryTransformation`
        """
        return GeometryTransformation(s=point)

    @staticmethod
    def get_scale(scale):
        """Creates a scaling transformation.

        :param scale: The scaling factor.
        :type scale: float

        :return: The scaling transformation
        :rtype: `GeometryTransformation`
        """
        t = np.diag(np.array(scale, dtype=np.float32))
        return GeometryTransformation(t=t)

    @staticmethod
    def get_rototranslation(axis, angles, axis_origin=None):
        """Create a roto-translation transformation.

        :param axis: Rotation axis
        :type axis: `numpy.array_like`
        :param angles: rotation angle
        :type angles: float
        :param axis_origin: Origin of the rotation axis
        :type axis_origin: `numpy.array_like`, optional. Default: None

        :return: The roto-translation transformation
        :rtype: `GeometryTransformation`
        """
        r_comp = GeometryTransformation.get_rotation_components(axis)
        r = GeometryTransformation.get_rotation_matrix(r_comp, angles)
        if axis_origin is not None:
            axis_origin = np.array(axis_origin, dtype=np.float32)
            s = axis_origin - np.dot(r, axis_origin)
            return GeometryTransformation(t=r, s=s)
        else:
            return GeometryTransformation(t=r)

    @staticmethod
    def get_transform(t_type, *args, **keywords):
        """Creates a transformation.

        :param t_type: Type of transformation. Options are: 'rot' | 'tr'
        :type t_type: string

        :return: The chosen transformation
        :rtype: `GeometryTransofrm`
        """
        if t_type.lower() == 'rot':
            return GeometryTransformation.get_rototranslation(*args, **keywords)
        elif t_type.lower() == 'tr':
            return GeometryTransformation.get_translation(*args, **keywords)
