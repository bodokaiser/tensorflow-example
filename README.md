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

`python3 format.py convert 01_mr.mnc 91_us.mnc 01.tfrecord`

### Partition

Choose your patients and create corresponding `test`, `train` and `validation`
datasets.

`python3 format.py partition 13.tfrecord length=100 --offset=10`

You can use `python3 format.py count 13.tfrecord` to see how much slices
there are however note that us patches are not proportional to amount of slices.

## Evaluation

`python3 model.py train`

## License

© 2017 Bodo Kaiser <bodo.kaiser@me.com>
