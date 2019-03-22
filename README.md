# plenoptomos

Plenoptic imaging reconstruction tools, developed in the context of the VOXEL
project, part of the European Union's Horizon 2020 research and innovation
programme (VOXEL H2020-FETOPEN-2014-2015-RIA  GA 665207).

It provides a Python package that allows to read and manipulate light-field images.
The provided features are:
* support for a few types of datasets and formats (including the .vox format).
* support for refocusing using standard algorithms like the integration and Fourier methods.
* support for GPU accelerated tomography based refocusing, including the following algorithms:
  * unfiltered back-projection
  * SIRT
  * Chambolle-Pock (Least-square, TV)
* support for depth estimation

* Free software: GNU General Public License v3
* Documentation: [https://cicwi.github.io/plenoptomos]

## Getting Started

It takes a few steps to setup plenoptomos on your
machine. We recommend installing
[Anaconda package manager](https://www.anaconda.com/download/) for
Python 3.

### Installing with conda

Simply install with:
```
conda install -c cicwi plenoptomos
```

### Installing from source

To install plenoptomos, simply clone this GitHub
project. Go to the cloned directory and run PIP installer:
```
git clone https://github.com/cicwi/plenoptomos.git
cd plenoptomos
pip install -e .
```

### Running the examples

To learn more about the functionality of the package check out our
examples folder.

## Authors and contributors

* **Nicola Viganò** - *Initial work*

See also the list of [contributors](https://github.com/cicwi/plenoptomos/contributors) who participated in this project.

## How to contribute

Contributions are always welcome. Please submit pull requests against the `master` branch.

If you have any issues, questions, or remarks, then please open an issue on GitHub.

## License

This project is licensed under the GNU General Public License v3 - see the [LICENSE.md](LICENSE.md) file for details.
