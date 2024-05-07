

# STDiff applied to CLEVRER</h1>
adapted from [XiYe20/STDiffProject](https://github.com/XiYe20/STDiffProject) (paper: [arXiv link](https://arxiv.org/abs/2312.06486))

<img src="./res/imageData.png" alt="STDiff Architecture" width="100%">
</p>

## Overview
<p align="center">
<img src="./res/NN_arch.png" alt="STDiff Architecture" width="100%">
</p>

## Installation
1. Install the custom diffusers library
```bash
git clone https://github.com/XiYe20/CustomDiffusers.git
cd CustomDiffusers
pip install -e .
```
2. Install the requirements
```bash
pip install -r requirements.txt
```

## Dataset

The dataset consists of 13000 video clips of 22 frames each in the `unlabeled` folder. The `val` and `train` folder are for inference.

available here: https://drive.google.com/file/d/1iYTFuf4DgxgYQzTQ_2da1vC9es_niPRr/view?usp=drive_link

Folder Structure \
  &nbsp;&nbsp;&nbsp;&nbsp; unlabeled/ \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; video_02000/ \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; image_0.png \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; image_1.png \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ... \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; image_21.png \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; video_02001/ \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ... \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; video_... \
&nbsp;&nbsp;&nbsp;&nbsp; train/ \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ...\
&nbsp;&nbsp;&nbsp;&nbsp; val/ \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ...

Note that there are also masks.npy files which are meant for segmentation.

## Training and Evaluation
Simiilar to STDiff project, accelerate is used for training. The configuration files are placed inside stdiff/configs.

### Training
1. Check train.sh, modify the visible gpus, num_process, modify the config.yaml file
2. Training
```bash
. ./train.sh
```

### Test
1. Check inference.sh, modify config.yaml for inference
2. Test
```bash
. ./inference.sh
```
