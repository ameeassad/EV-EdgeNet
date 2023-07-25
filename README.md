# EV-EdgeNet

Forked from EV-SegNet Reproducibility, which included fixes for Python3 + TF2.7
Altered to allow for semantic edge detection. 

# Based on Ev-SegNet

[![Ev-SegNet](utils/image.png)](https://youtu.be/YQXBjWUSiaA)

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

The semantic labels of the data are:
objects/wand
objects/box_00
objects/box_01
objects/box_02
objects/box_03
objects/car_00
objects/car_01
objects/checkerboard
objects/turntable
objects/tabletop
objects/marker_00
objects/marker_01
objects/marker_02
objects/marker_03
objects/can_00
objects/drone_00
objects/drone_01
objects/knife_00
objects/plane_00
objects/plane_01
objects/toy_00
objects/wheel_00
objects/wheel_01
objects/wheel_02
objects/wheel_03


## Dataset preprocessing
Download samsung_mono and flea3_7 datasets with .npz files (not the .txt versions) and place them in the same directory. Then run `process_imgs.py` and change the folder names to `train` and `test` (since it might say `eval`).

## Replicate results
For testing the pre-trained model just execute:
```
python train_evimo.py --epochs 0
```

## Train from scratch


```
python train_evimo.py --epochs 500 --dataset path_to_dataset  --model_path path_to_model  --batch_size 8
```

Where [path_to_dataset] is the path to the downloaded dataset (uncompressed) and [path_to_model] is the path where the weights are going to be saved

