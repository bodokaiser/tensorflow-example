# MRtoUS

### Conversion
```shell
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
