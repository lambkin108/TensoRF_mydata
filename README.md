
# TensoRF on mydata
Copy from https://github.com/apchenstu/TensoRF
and train on mydata

## Dataset
We use a dataset named mydata stored in the data/ directory.
The data is randomly split into a training set (80%) and a testing set (20%) to ensure fair evaluation.

## Environment
#### Tested on Ubuntu 20.04 + Pytorch 1.10.1 and RTX4090
Install environment:
```
conda create -n TensoRF python=3.8
conda activate TensoRF
pip install torch torchvision
pip install tqdm scikit-image opencv-python configargparse lpips imageio-ffmpeg kornia lpips tensorboard
```

## Quick Start
The training script is in `train.py`, to train a TensoRF:

```
python train.py --config configs/your_own_data.txt
```
## Rendering

```
python train.py --config configs/your_own_data.txt --ckpt path/to/your/checkpoint --render_only 1 --render_test 1 
```

You can just simply pass `--render_only 1` and `--ckpt path/to/your/checkpoint` to render images from a pre-trained
checkpoint. You may also need to specify what you want to render, like `--render_test 1`, `--render_train 1` or `--render_path 1`.
The rendering results are located in your checkpoint folder. 

## Extracting mesh
You can also export the mesh by passing `--export_mesh 1`:
```
python train.py --config configs/lego.txt --ckpt path/to/your/checkpoint --export_mesh 1
```
Note: Please re-train the model and don't use the pretrained checkpoints provided by us for mesh extraction, 
because some render parameters has changed.

## Citation
```
@INPROCEEDINGS{Chen2022ECCV,
  author = {Anpei Chen and Zexiang Xu and Andreas Geiger and Jingyi Yu and Hao Su},
  title = {TensoRF: Tensorial Radiance Fields},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year = {2022}
}
```
