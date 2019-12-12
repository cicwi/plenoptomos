# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!--
## [Unreleased]
### Added
### Fixed
-->

## 0.1.1 - 2019-12-12
### Added
- Example on creating and using files in .vox data format
- Added TV regularization to PSF deconvolution
### Fixed
- Fixed and enabled OTF functionality in PSF convolution
- Allow loading when ASTRA is not installed
- Try using ImageIO for TIFF images before matplotlib
- Fixed renormalization in traditional integration algorithm for upsampled cases
- Small fixes to PSF deconvolution
- Fixed padding in refocus, when using wavelet regularization

## 0.1.0 - 2019-03-25
- Initial release.
### Added
* support for a few types of datasets and formats (including the .vox format).
* support for refocusing using standard algorithms like the integration and Fourier methods.
* support for GPU accelerated tomography based refocusing, including the following algorithms:
  * unfiltered back-projection
  * SIRT
  * Chambolle-Pock (Least-square, TV)
* support for depth estimation

[Unreleased]: https://www.github.com/cicwi/plenoptomos/compare/v0.1.0...develop
