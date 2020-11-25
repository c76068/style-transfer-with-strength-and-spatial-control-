# style-transfer-with-strength-and-spatial-control
This repository contains a pytorch implementation of our [group project](https://github.com/c76068/style-transfer-with-strength-and-spatial-control-/blob/master/Computer_Vision_Final_Report.pdf) for CS 461/661 Computer Vision in spring 2020 at Johns Hopkins University. We combined the style transfer approaches developed in [Controlling Perceptual Factors in Neural Style Transfer](https://ieeexplore.ieee.org/document/8099880) and [Real-Time Style Transfer With Strength Control](https://link.springer.com/chapter/10.1007/978-3-030-29891-3_19) to to build a pipeline for fast style transfer with spatial and strength control. Our implementation allows transfer of two distinct styles onto two regions of the input content image guided by masks generated from [semantic
segmentation](https://arxiv.org/abs/1706.05587). Our code was adapted from [
style-transfer-with-strength-control ](https://github.com/victorkitov/style-transfer-with-strength-control) and the code for computing receptive field was taken from [pytorch-receptive-field](https://github.com/Fangyh09/pytorch-receptive-field).

## Usage
Stylize image:
```
python test_2sty.py --content 'images/contents/foxcat.jpg' --model 'models/pencil_picasso_traincount100000_styweight50000.0_bs10.pth' --style_strength 1 --switch True --mask 1 --gpu_id 0
```

Train model:
```bash
python style-transfer-with-strength-control-master/train_2sty.py --style_image1 'style-transfer-with-strength-control-master/images/styles/pencil.jpg' --style_image2 'style-transfer-with-strength-control-master/images/styles/picasso.jpg' --dataset 'data' --batch_size 10 --max_train_count 100000 --style_weight 5e4 --save_model_dir 'style-transfer-with-strength-control-master/models/' --log_batches_interval 20 --gpu_id 0
```
