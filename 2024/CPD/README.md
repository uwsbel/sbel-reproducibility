# Scripts for generating a contextualized performance difference (CPD)

## Object Detection comparison with ground truths and context based on bounding boxes

`python compare_detection.py -h`

Labels and file structures based on YoloV5 

## Segmentaiton comparison with ground truths and context based on semantic maps

`python compare_segmentation.py -h`

Labels and file structures based on Cityscapes


# Environment setup

Activate python environment with:

`$. activate.sh`


Exit the environment with:

`$. deactivate.sh` 

or 

`$deactivate`


# Accessing Cone Datasets 

The cone data can be downloaded from [here](https://1drv.ms/u/s!AnXeazVic6fV8AIeuBU_J9zlmrO7?e=IIyJZH).

The Cityscapes/GTAV/GTAV-EPE datasets can be obtained from their respective hosts. Please see our paper for references.


# Running CPD on cone data

## Setup

1. Make sure the python environment has been configured as described above or equivalently such that all dependencies have been met
2. Download the cone data and uncompress. The following instructions will assume the current directory shows the following:

```
$ls
activate.sh  compare_detection.py     deactivate.sh  README.md  requirements.txt
common       compare_segmentation.py  metrics        *real*       *sim*
```

where *real* and *sim* come from the downloaded cone dataset.


## Running CPD

`python compare_detection.py --a_path real/combined --a_preds real/combined/predictions_netreal --b_path sim/combined --b_preds sim/combined/predictions_netreal --n 300 -pt .8 -ps 120 120 --output_path results/netreal --verbose --save`

This may take a few minutes, with the results saved to `results/`.
