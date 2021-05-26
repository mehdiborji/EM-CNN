# EM-CNN
Deep learning pipeline for classification of histopathology images using Expectation Maximization iterations. This is a simplified implementation of:


[Hou, L., Samaras, D., Kurc, T.M., Gao, Y., Davis, J.E. and Saltz, J.H., 2016. Patch-based convolutional neural network for whole slide tissue image classification. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 2424-2433).](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Hou_Patch-Based_Convolutional_Neural_CVPR_2016_paper.pdf)

Scripts:

To run the patch generation pipeline (based on openslide) use:
```
python patch_run_spleen.py
```
Slide classes, ids, and patch coordinates will be encoded into the file path and file name.

To run the main training, validation loop use:
```
python EM.py
```

Utility functions in `table_map` can be used to tabulate predictions, in `.txt` format, and also generate probability/prediction heatmaps overlayed on downscaled versions of the original whole slide files, in `.png` format. Data directory is comprised of `.npy` files that contain probability and patch data vectors for the inference runs at different steps of training scheme or final test on independent slides:

```
>>> import table_map
>>> data_dir = '../data_dir/'
>>> table_map._table_count(data_dir)
```

