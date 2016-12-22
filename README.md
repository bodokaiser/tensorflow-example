# MRtoUS

Generate US images from MR brain images.

## Setup

### Dataset

Download [here](http://www.bic.mni.mcgill.ca/%7Elaurence/data/data.html) (group 2).

### Toolset

Required to work with MINC format.

Install [here](http://bic-mni.github.io).

## Pre Processing

### Registration

1. Create Transformation

`register -sync 01_mr_tal.mnc 01a_us_tal.mnc 01_all.tag`

2. Apply Transformation

`mincresample 01a_us_tal.mnc 01_us_reg.mnc -transformation 01_reg.xfm -like 01_mr_tal.mnc`

### Conversion

1. HDF5

`mincconvert -2 01_mr_tal.mnc 01_mr.mnc`

2. TFRecord

Coming soon.

## Training

Coming soon.

## License

© 2017 Bodo Kaiser <bodo.kaiser@me.com>
