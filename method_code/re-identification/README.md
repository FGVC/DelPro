# Requirements

- [dnnutil](https://github.com/catalys1/dnnutil)
- pytorch 1.1.0
- numba (for thin plate spline augmentation)


## Installation

Install pytorch:
```
pip install pytorch
```
Download and install dnnutil:
```
git clone https://github.com/catalys1/dnnutil.git
pip install -e dnnutil
```
Note that the example usage of dnnutil on github is not currently up to date.

# Training a model

First, make sure you have a symlink to the data with the name `data` in the same folder as the main script. The files in `configs` define the different training pieces, such as the model, dataset, optimizer, trainer, etc.

Make sure that you set your `CUDA_VISIBLE_DEVICES` flag:
```
export CUDA_VISIBLE_DEVICES=0  # or whichever device you want to use
```
Then run the main file, pointing to the config you want to use:
```
python main.py start configs/siam-conf.json
```
