# plenoptomos

Plenoptic imaging reconstruction tools, developed in the context of the VOXEL
project, part of the European Union's Horizon 2020 research and innovation
programme ([VOXEL H2020-FETOPEN-2014-2015-RIA  GA 665207](https://ec.europa.eu/programmes/horizon2020/en/news/3d-x-ray-imaging-very-low-dose))

It provides a Python package that allows to read and manipulate light-field images.
The provided features are:
* support for a few types of datasets and formats (including the .vox format).
* support for refocusing using standard algorithms like the integration [3] and Fourier [4] methods.
* support for GPU accelerated tomography based refocusing, including the following algorithms:
  * unfiltered back-projection [1]
  * SIRT [2]
  * Chambolle-Pock (Least-square, TV) [2]
* support for depth estimation [5]

This work is based on the following articles:
- [1] N. Viganò, et al., “Tomographic approach for the quantitative scene reconstruction from light field images,” Opt. Express, vol. 26, no. 18, p. 22574, Sep. 2018.
- [2] N. Viganò, et al., “Advanced light-field refocusing through tomographic modeling of the photographed scene,” Opt. Express, vol. 27, no. 6, p. 7834, Mar. 2019.

It also implements methods from, the following articles:
- [3] R. Ng, et al., “Light Field Photography with a Hand-held Plenoptic Camera,” Stanford Univ. Tech. Rep. CSTR 2005-02, 2005.
- [4] R. Ng, “Fourier slice photography,” ACM Trans. Graph., vol. 24, no. 3, p. 735, 2005.
- [5] M. W. Tao, et al., “Depth from combining defocus and correspondence using light-field cameras,” Proceedings of the IEEE International Conference on Computer Vision, 2013, pp. 673–680.

Other useful information:
* Free software: GNU General Public License v3
* Documentation: [https://cicwi.github.io/plenoptomos]


## Readiness

The author of this package is in the process of setting up this
package for optimal usability. The following has already been completed:

- [ ] Documentation
    - A package description has been written in the README
    - Documentation has been generated using `make docs`, committed,
        and pushed to GitHub.
	- GitHub pages have been setup in the project settings
	  with the "source" set to "master branch /docs folder".
- [ ] An initial release
	- In `CHANGELOG.md`, a release date has been added to v0.1.0 (change the YYYY-MM-DD).
	- The release has been marked a release on GitHub.
	- For more info, see the [Software Release Guide](https://cicwi.github.io/software-guides/software-release-guide).
- [x] A conda package
    - Required packages have been added to `setup.py`, for instance,
      ```
      requirements = [
          # Add your project's requirements here, e.g.,
          # 'astra-toolbox',
          # 'sacred>=0.7.2',
          # 'tables==3.4.4',
      ]
      ```
      has been replaced by
      ```
      requirements = [
          'astra-toolbox',
          'sacred>=0.7.2',
          'tables==3.4.4',
      ]
      ```
    - All "conda channels" that are required for building and
      installing the package have been added to the
      `Makefile`. Specifically, replace
      ```
      conda_package:
        conda install conda-build -y
        conda build conda/
      ```
      by
      ```
      conda_package:
        conda install conda-build -y
        conda build conda/ -c some-channel -c some-other-channel
      ```
    - Conda packages have been built successfully with `make conda_package`.
    - These conda packages have been uploaded to
      [Anaconda](https://anaconda.org). [This](http://docs.anaconda.com/anaconda-cloud/user-guide/getting-started/#cloud-getting-started-build-upload)
      is a good getting started guide.
    - The installation instructions (below) have been updated. Do not
      forget to add the required channels, e.g., `-c some-channel -c
      some-other-channel`, and your own channel, e.g., `-c cicwi`.


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

* **Nicola Viganò** - *Main developer*
* **Francesco Brun** - *Initial contributions*
* **Pablo Martinez Gil** - *Contributed to the creation and enabling of the .vox data format, and to the wavelet solver*
* **Charlotte Herzog** - *Contributed to the creation an

See also the list of [contributors](https://github.com/cicwi/plenoptomos/contributors) who participated in this project.

## How to contribute

Contributions are always welcome. Please submit pull requests against the `master` branch.

If you have any issues, questions, or remarks, then please open an issue on GitHub.

## License

This project is licensed under the GNU General Public License v3 - see the [LICENSE.md](LICENSE.md) file for details.

## Data format

This package comes with a suggested data format for the light-fields.
The datata format is described in the data_format.py module, and it addresses
the following points:

Rationale behind version 0:
- single image
- initial standardization containing all the necessary metadata
- first structure, vaguely inspired by DataExchange

Structure:
```
root
 +--<attribute> = description : string -> "VoxelDataFormat"
 +--<attribute> = version : string -> "v0"
 |
 +--<group> = data
 |     |
 |     +--<dataset> = image : integer [ <p x p x n x n> | <n x n x p x p> | <np x np> | etc ]
 |     |      +-<attribute> = units : string -> "counts"
 |     |      +-<attribute> = axes : string -> [ "v:u:t:s" | "t:s:v:u" | "tau:sigma" | etc ]
 |     |      +-<attribute> = mode : string -> [ "sub-aperture" | "micro-image" | "raw" | etc ]
 |     |
 |     +--<dataset> = white : integer [ <p x p x n x n> | <n x n x p x p> | <np x np> | etc ]
 |     |      +-<attribute> = units : string -> "counts"
 |     |      +-<attribute> = axes : string -> [ "v:u:t:s" | "t:s:v:u" | "tau:sigma" | etc ]
 |     |      +-<attribute> = mode : string -> [ "sub-aperture" | "micro-image" | "raw" | etc ]
 |     |
 |     +--<dataset> = dark : integer [ <p x p x n x n> | <n x n x p x p> | <np x np> | etc ]
 |            +-<attribute> = units : string -> "counts"
 |            +-<attribute> = axes : string -> [ "v:u:t:s" | "t:s:v:u" | "tau:sigma" | etc ]
 |            +-<attribute> = mode : string -> [ "sub-aperture" | "micro-image" | "raw" | etc ]
 |
 +--<group> = instrument
 |     |
 |     +--<group> = source ?
 |     |
 |     +--<group> = camera
 |     |     +--<attribute> = manufacturer : string [optional]
 |     |     +--<attribute> = model : string [optional]
 |     |     |
 |     |     +--<group> = micro_lenses_array
 |     |     |     +--<attribute> = manufacturer : string [optional]
 |     |     |     +--<attribute> = model : string [optional]
 |     |     |     |
 |     |     |     +--<dataset> = size : integer <- number of micro-lenses
 |     |     |     |      +-<attribute> axes : string -> "tau:sigma"
 |     |     |     |
 |     |     |     +--<dataset> = position : float <- position of the center of the MLA with respect with the center of the main lens
 |     |     |     |      +-<attribute> = axes : string -> "x:y:z"
 |     |     |     |      +-<attribute> = units : string -> [ {"mm"} | "um" | "m" ]
 |     |     |     |
 |     |     |     +--<dataset> = tilts ?
 |     |     |
 |     |     +--<group> = micro_lens
 |     |     |     |
 |     |     |     +--<dataset> = size : integer <- size of each microlens in detector pixels after regularization
 |     |     |     |      +-<attribute> axes : string -> "tau:sigma"
 |     |     |     |      +-<attribute> = units : string -> "pixels"
 |     |     |     |
 |     |     |     +--<dataset> = physical_size : float <- physical size of each micro-lens
 |     |     |     |      +-<attribute> axes : string -> "t:s"
 |     |     |     |      +-<attribute> = units : string -> [ {"mm"} | "um" | "m" ]
 |     |     |     |
 |     |     |     +--<dataset> = micro_image_size : float <- footprint of each microlens over detector pixels
 |     |     |     |      +-<attribute> axes : string -> "tau:sigma"
 |     |     |     |      +-<attribute> = units : string -> "pixels"
 |     |     |     |
 |     |     |     +--<dataset> = f2 : float <- focal length of each lenslet
 |     |     |     |      +-<attribute> = units : string -> [ {"mm"} | "um" | "m" ]
 |     |     |     |
 |     |     |     +--<dataset> = aperture : float <- f-number
 |     |     |
 |     |     +--<group> = main_lens
 |     |     |     +--<attribute> = manufacturer : string [optional]
 |     |     |     +--<attribute> = model : string [optional]
 |     |     |     |
 |     |     |     +--<dataset> = pixel_size : float <- angular resolution of the main-lens
 |     |     |     |      +-<attribute> axes : string -> "v:u"
 |     |     |     |      +-<attribute> = units : string -> [ {"mm"} | "um" | "m" ]
 |     |     |     |
 |     |     |     +--<dataset> = f1 : float <- focal length of each lenslet
 |     |     |     |      +-<attribute> = units : string -> [ {"mm"} | "um" | "m" ]
 |     |     |     |
 |     |     |     +--<dataset> = aperture : float <- f-number
 |     |     |
 |     |     +--<group> = sensor
 |     |           +--<attribute> = manufacturer : string [optional]
 |     |           +--<attribute> = model : string [optional]
 |     |           |
 |     |           +--<dataset> = size : integer <- number of pixels
 |     |           |      +-<attribute> axes : string -> "tau:sigma"
 |     |           |
 |     |           +--<dataset> = pixel_size : float <- physical size of each pixel
 |     |           |      +-<attribute> axes : string -> "tau:sigma"
 |     |           |      +-<attribute> = units : string -> [ {"mm"} | "um" | "m" ]
 |     |           |
 |     |           +--<dataset> = position : float <- position of the center of the sensor with respect with the center of the main lens
 |     |           |      +-<attribute> = axes : string -> "x:y:z"
 |     |           |      +-<attribute> = units : string -> [ {"mm"} | "um" | "m" ]
 |     |           |
 |     |           +--<dataset> = tilts ?
 |     |           |
 |     |           +--<dataset> = detection parameters ? (i.e. material, thickness, quantum efficiency)
 |     |
 |     +--<group> = monochromator ? (could be used for RGB images -> split into three-channels)
 |     |
 |     +--<group> = scintillator ? (might be related/redundant to/with detection parameters in detector)
 |
 +--<group> = sample [optional]
 |     |
 |     +--<dataset> = position : float <- position of the center of the sample with respect with the center of the main lens
 |     |      +-<attribute> = axes : string -> "x:y:z"
 |     |      +-<attribute> = units : string -> [ {"mm"} | "um" | "m" ]
 |     |
 |     +--<dataset> = size : float <- size of the sample
 |     |      +-<attribute> = axes : string -> "x:y:z"
 |     |      +-<attribute> = units : string -> [ {"mm"} | "um" | "m" ]
 |     |
 |     +--<dataset> name : string
 |     |
 |     +--<dataset> description : string [optional]
 |
 +--<group> = acquisition
       |
       +--<dataset> = type : string -> [ "reflection" | "transmission" ]
       |
       +--<dataset> = exposure_time : float
              +-<attribute> = units : string -> [ {"s"} | "ms" | "us" ]
```
