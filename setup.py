#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages
import os.path

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('CHANGELOG.md') as history_file:
    history = history_file.read()

with open(os.path.join('plenoptomos', 'VERSION')) as version_file:
    version = version_file.read().strip()

requirements = [
    # Add your project's requirements here, e.g.,
    'numpy',
    'scipy',
    'matplotlib',
    'astra-toolbox',
    'h5py',
    'configparser',
    'pillow',
    'imageio',
    'tqdm',
]

setup_requirements = []

test_requirements = []

dev_requirements = [
    'autopep8',
    'rope',
    'jedi',
    'flake8',
    'importmagic',
    'autopep8',
    'black',
    'yapf',
    'snakeviz',
    # Documentation
    'sphinx',
    'sphinx_rtd_theme',
    'recommonmark',
    # Other
    'watchdog',
    'coverage',
]

setup(
    author="Nicola VIGANÒ",
    author_email='N.R.Vigano@cwi.nl',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Plenoptic imaging reconstruction tools",
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='plenoptomos',
    name='plenoptomos',
    packages=find_packages(include=['plenoptomos']),
    python_requires='>=3.5',
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    extras_require={'dev': dev_requirements},
    url='https://github.com/cicwi/plenoptomos',
    version=version,
    zip_safe=False,
)
