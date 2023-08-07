# EV-EdgeNet

Forked from EV-SegNet Reproducibility, which included fixes for Python3 + TF2.7
Altered to allow for semantic edge detection. 

# Based on Ev-SegNet
This work proses an approach for learning semantic segmentation from only event-based information (event-based cameras).

For more details, here is the [Paper](https://drive.google.com/file/d/1eTX6GXy5qP9I4PWdD4MkRRbEtfg65XCr/view?usp=sharing)


# Requirements
* Python 2.7+
* Tensorflow 1.11
* Opencv
* Keras
* Imgaug
* Sklearn


## Ev-SegNet Citation:
``` 
@inproceedings{alonso2019EvSegNet,
  title={EV-SegNet: Semantic Segmentation for Event-based Cameras},
  author={Alonso, I{\~n}igo and Murillo, Ana C},
  booktitle={IEEE International Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  year={2019}
}
```

## Dataset
EVIMO2

[Download it here](https://better-flow.github.io/evimo/download_evimo_2.html)

The semantic labels of the data were changed to the following 14 classes (from original 28 which had 3 unknown classes):
| Original Classes     | Processed Classes         |
|----------------------|---------------------------|
| objects/wand         | 1. Wrapped Box (5)        |
| objects/box_00       | 2. Car (6)                |
| objects/box_01       | 3. Cylinder (8)           |
| objects/box_02       | 4. Checkerboard (9)       |
| objects/box_03       | 5. Soft rectangular box (10) |
| objects/car_00       | 6. Can (11)               |
| objects/car_01       | 7. Stand (12)             |
| objects/checkerboard | 8. Car base (13)          |
| objects/turntable    | 9. Drones (14,15)         |
| objects/tabletop     | 10. Square sheets (17,18,19,20) |
| objects/marker_00    | 11. Tabletop (22)         |
| objects/marker_01    | 12. Gun toy (23)          |
| objects/marker_02    | 13. Turntable, circular (24) |
| objects/marker_03    | 14. Wheels (25,26,27,28)  |
| objects/can_00       |                           |
| objects/drone_00     |                           |
| objects/drone_01     |                           |
| objects/knife_00     |                           |
| objects/plane_00     |                           |
| objects/plane_01     |                           |
| objects/toy_00       |                           |
| objects/wheel_00     |                           |
| objects/wheel_01     |                           |
| objects/wheel_02     |                           |
| objects/wheel_03     |                           |




## How to replicate training results

First, get the pre-processed EVIMO2 datasets

1. Download flea_3_7 and Samsung_mono as npz files from here: https://better-flow.github.io/evimo/download_evimo_2.html 
2. Put them at the same level in a directory called `datasets`, at the level of the EV-EdgeNet directory
3. Run process_imgs.py
4. Make sure the split names are ‘train’ and ‘test’ (not ‘eval’)

## Next, run the training
1. For edge training on event data, run `python train_evimo.py  --dataset ../datasets/processed-fixed --batch_size 8  --epochs 200 --problem_type edges`
2. For segmentation training on event data, run `python train_evimo.py  --dataset ../datasets/processed-fixed --batch_size 8  --epochs 200 --problem_type segmentation`
3. For RGB training only, you can add the parameters --total_channels 3 --rgb_channels 3

## For **inference**, run the same line as training put use `--epochs 0` and specify the weights you want to use in `--model_path`:
1. for semantic edge detection with events, use `--model_path weights/edges-events`
2. for semantic edge detection with RGB images, use `--model_path weights/edges-rgb`
3. for semantic segmentation with events, use `--model_path weights/segmentation-events`
