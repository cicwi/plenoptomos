#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Top-level package for plenoptomos.

Created on Thu Mar  2 17:44:15 2017
"""

__author__ = """Nicola VIGANÒ"""
__email__ = 'N.R.Vigano@cwi.nl'


def __get_version():
    import os.path
    version_filename = os.path.join(os.path.dirname(__file__), 'VERSION')
    with open(version_filename) as version_file:
        version = version_file.read().strip()
    return version


__version__ = __get_version()

# Import all definitions from main module.
from . import lightfield  # noqa: F401, F402

from . import refocus  # noqa: F401, F402
try:
    from . import tomo  # noqa: F401, F402
    from . import depth  # noqa: F401, F402
except ImportError as ex:
    print('WARNING: error while importing tomography module.\nAdvanced refocusing and depth estimation will not be available')
    print('Error message:\n', ex)

from . import utils_io  # noqa: F401, F402
from . import import_lf  # noqa: F401, F402
from . import data_format  # noqa: F401, F402

from . import solvers  # noqa: F401, F402
from . import utils_proc  # noqa: F401, F402

from . import colors  # noqa: F401, F402
