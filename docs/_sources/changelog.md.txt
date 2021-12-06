# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!--
## [Unreleased]
### Added
### Fixed
-->

[Unreleased]: https://www.github.com/cicwi/plenoptomos/compare/v0.2.0...develop

## 0.2.0 - 2021-12-06
### Added
- Refactored tomo projector:
  * Removed support for multiple independent projection at the same time
  * Removed difference between modes independent and simultaneous modes
  * Moved from astra-toolbox' OpTomo to experimental direct_FPBP
- Multi-threading support for Integration algorithm
- tqdm support in depth estimation
- Decoupling between focal stack computation and cues computation in depth estimation
- Refactored (simplified) PSF code
- Simplified light-field mode conversion
### Fixed
- Unwanted write in-place
- Examples #1 and #2
- Negative values in KL data term
- PSF handling in reconstructions

## 0.1.4 - 2020-12-02
### Added
- Hidden support for different data fitting norms in the correspondence cue computation (depth-estimation)
- Normalization support for flat-field data
- Colorbar support in refocus visualization
### Fixed
- Loading TIFF data (relying on the ImageIO library now)
- Loading of flexray data
- Refocus visualization color limits
- Typos

## 0.1.3 - 2020-08-18
### Fixed
- Computation of depth sub-sampling quadratic fit
- Import of flexray light-fields
- Depth-map computation with cross-correlation

## 0.1.2 - 2020-06-11
### Added
- Support for pixel masks
- Support for saving in .vox data format
- Image processing functions (smoothing, lowpass and highpass filters, background subtraction, etc)
- Sub-aperture image warping
- Simple focal stack visualization tool
- Unsupported tools for: tomographic reconstructions, matrix testing, and creation of refocusing movies
- Depth-estimation quadratic refinement of peak positions
- Depth-estimation "2nd peak" confidence method
- Complete support for cross-correlation cue
### Fixed
- Test failure
- Roll-off correction for Fourier refocusing
- Formatting style (linting applied)
- Documentation (including docstrings quality)
- Defocus depth cue: use gradient instead of laplacian
- Correspondence depth cue: correct computation of confidence

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
- support for a few types of datasets and formats (including the .vox format).
- support for refocusing using standard algorithms like the integration and Fourier methods.
- support for GPU accelerated tomography based refocusing, including the following algorithms:
  * unfiltered back-projection
  * SIRT
  * Chambolle-Pock (Least-square, TV)
- support for depth estimation
