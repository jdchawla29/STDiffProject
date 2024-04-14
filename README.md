

# STDiff: Spatio-temporal diffusion for continuous stochastic video prediction</h1>
[arXiv](https://arxiv.org/abs/2312.06486) | Forked from [here](https://github.com/XiYe20/STDiffProject)

## Overview
<p align="center">
<img src="./documentations/NN_arch.png" alt="STDiff Architecture" width="100%">
</p>

## Installation
1. Install the custom diffusers library
```bash
git clone https://github.com/XiYe20/CustomDiffusers.git
cd CustomDiffusers
pip install -e .
```
2. Install the requirements of STDiff
```bash
pip install -r requirements.txt
```

## Dataset

[Animated CLEVR Dataset (NYU)](https://drive.google.com/file/d/1iYTFuf4DgxgYQzTQ_2da1vC9es_niPRr/view?usp=drive_link)

The dataset features synthetic videos with simple 3D shapes that interact with
each other according to basic physics principles. Objects in videos have three
shapes (cube, sphere, and cylinder), two materials (metal and rubber), and eight
colors (gray, red, blue, green, brown, cyan, purple, and yellow). In each video,
there is no identical objects, such that each combination of the three attributes
uniquely identifies one object.

Structure of Dataset





## Training and Evaluation
The STDiff project uses accelerate for training. The training configuration files and test configuration files for different datasets are placed inside stdiff/configs.

### Training
1. Check train_script.sh, modify the visible gpus, num_process, select the correct train_cofig file
2. Training
```bash
. ./train_script.sh
```

### Test
1. Check test_script.sh, select the correct test_cofig file
2. Test
```bash
. ./test_script.sh
```



