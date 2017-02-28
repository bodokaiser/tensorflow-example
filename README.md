# MRtoUS

Generate US images from MR brain images.

## Setup

1. Download the [group2 dataset][dataset].
2. Download and install the [minc toolset][toolset].

[dataset]: http://www.bic.mni.mcgill.ca/%7Elaurence/data/data.html
[toolset]: http://bic-mni.github.io

## Usage

### Registration

```shell
# export the register transformation in the GUI
register -sync 01_mr_tal.mnc 01a_us_tal.mnc 01_all.tag

# apply the transformation to ultrasound images
mincresample 01a_us_tal.mnc 01_us_reg.mnc -transformation 01_reg.xfm -like 01_mr_tal.mnc
```

### Conversion

The MNI-BITE dataset uses [MINC-1.0][minc1] and [MINC-2.0][minc2] format. Later
extends [HDF5][hdf5] which has a maintained python implementation [h5py][h5py].

[h5py]: http://www.h5py.org
[hdf5]: https://en.wikipedia.org/wiki/Hierarchical_Data_Format
[minc1]: https://en.wikibooks.org/wiki/MINC/SoftwareDevelopment/MINC1_File_Format_Reference
[minc2]: https://en.wikibooks.org/wiki/MINC/SoftwareDevelopment/MINC2.0_File_Format_Reference

```shell
# ensure that all dataset is in MINC-2.0
mincconvert -2 01_mr_tal.mnc 01_mr.mnc

# use provided format python script to convert MINC-2.0 to TFRecord
python3 format.py convert 01_mr.mnc 01_us.mnc 01.tfrecord
```

### Partition

Choose your patients and create corresponding `test`, `train` and `validation`
datasets.

```shell
# extracts first 100 mr and us slices into validation.tfrecord
python3 format.py partition 13.tfrecord validation.tfrecord length=100
```

### Evaluation

```shell
python3 evaluate.py
```

## License

© 2017 Bodo Kaiser <bodo.kaiser@me.com>
