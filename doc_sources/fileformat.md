# File format

The plenoptomos toolbox comes with a suggested data format for the light-fields.
This data format is meant for storing the acquired light-fields and for their exchange.
It addresses the following points:

Rationale behind version 0:
- Single light-field per file
- Contains all the necessary metadata for interpretation of the light-field
- Based on [HDF5](https://www.hdfgroup.org/solutions/hdf5/) and vaguely inspired by [DataExchange](https://github.com/data-exchange/dxchange)

Convetions:
- Arrays are to be understood in the C convention
- (v, u) coordinates refer to positions on the main lens
- (t, s) coordinates refer to positions on the focal plane in the image space

Tree structure:
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
