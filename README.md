# AdaIN_Style_in_Pytorch_From_Scratch
This is an unofficial PyTorch implementation of [Arbitrary Style Transfer in Real-time with 
Adaptive Instance Normalization [Huang, Belongie ICCV2017]](https://arxiv.org/abs/1703.06868).
Original implementation in lua can be found [here](https://github.com/xunhuang1995/AdaIN-style).

This paper proposes a very  effective approach that for the first time enables *arbitrary* style transfer in real-time, 
 in contrast to [a single style](https://arxiv.org/abs/1603.03417) or [32 styles](https://arxiv.org/abs/1610.07629). 
The original algorithm runs at 15 FPS with 512x512 images on a Pascal Titan X, and is around 720x 
speedup compared with the [original algorithm](https://arxiv.org/abs/1508.06576) of Gatys et al., 
without sacrificing any flexibility. 
In essence, this model uses a novel adaptive instance normalization (AdaIN) layer, which is similar to
 [instance normalization](https://arxiv.org/abs/1701.02096)  that aligns the
mean and variance of the content features with those of the style features. 

![Architecture](./architecture.jpg)

This repository provides a pre-trained model for you to stylize your own image.
You can either download the training dataset or prepare your own dataset to train the model from scratch.

## Requirements
To install all required packages, run the following 
```
$ pip install -r requirements.txt
```

Required:
* Python 3.6+
* [pytorch](https://pytorch.org)
* [torchvision](https://pytorch.org)
* Pillow
* numpy

Optional (for training):
* tqdm

## Test

1. Clone this repository  ```$ git clone https://github.com/benningtonlee7/AdaIn_Style_Transfer_From_Scratch_In_Pytorch.git ```

2. Prepare your own images. For convenience, I already provided some images in `content` and `style` folders
for you to try.

3. To generate an image, run ```test.py ```. Here are some basic usage:

```
Usage: test.py  [--h HELP]
                [--content CONTENT.JPG] 
                [--style STYLE.JPG]
                [--model MODEL_PATH]
                [--output OUTPUT_DIRECTORY] 
                [--alpha STYLIZATION_FACTOR] 
```
Some other options:
* `--crop`: Center crop both content and style images beforehand.
* `--ext`: Extension name for the output image.
* `--preserve-color`: To preserve the color of the content image

To see all available options, type the following command:
```
$ python test.py --help 
```
For instance, to transfer an image from a trained model ```models/adain_model.pth``` with ```content/content.jpg``` and ```style/style.jpg```,
you can type:
```
$ python test.py \
>    --model models/adain_model.pth \
>    --content content/content.jpg \
>    --style style/style.jpg
```
   
## Train

1. Download the content dataset [COCO](http://cocodataset.org/#download) and the style dataset [Wikiart](https://www.kaggle.com/c/painter-by-numbers),
unzip them, and rename them as `content` and `style` respectively.

2. Train the model with `train.py`. Here are some usage:

```
usage: train.py [--h HELP] 
                [--continual <.pth>] 
                [--continual ENABLE_CONTINUAL_TRAINING] 
                [--save_dir PATH_TO_SAVE_MODEL] 
                [--log_dir PATH_TO_SAVE_LOGS] 
                [--log_image_every INT] 
                [--save_interval INT] 
                [--cuda ENABLE_GPU] 
                [--n-threads INT] 
```
Hyper parameters:
```
Usage: train.py [--learning_rate] 
                [--learning_rate_decay ENABLE_DECAY_LEARNING_RATE]
                [--max_iter MAX_NUMBER_ITERATION] 
                [--batch_size INT] 
                [--content_weight FLOAT] 
                [--style_weight FLOAT]
```

For instance, to train model with ```COCO``` as content and ```wikiart``` as style, type:
```
$ python train.py \
>   --content-dir content \
>   --style-dir style
```
To check saved logs, simply type ```tensorboard --log_dir logs ```

For more, type the following command:
```
$ python train.py --help 
```
a
## Result
The following are some results.

| Original | Style | Output |
| ------------- | ------------- | -------------  |
|<img src=https://raw.githubusercontent.com/benningtonlee7/AdaIn_Style_Transfer_From_Scratch_In_Pytorch/main/content/houses.jpg width="270"> | <img src=https://raw.githubusercontent.com/benningtonlee7/AdaIn_Style_Transfer_From_Scratch_In_Pytorch/main/style/sketch.png width="200"> | <img src=https://raw.githubusercontent.com/benningtonlee7/AdaIn_Style_Transfer_From_Scratch_In_Pytorch/main/outputs/houses_stylized_sketch.jpg width="270"> |
| | <img src=https://raw.githubusercontent.com/benningtonlee7/AdaIn_Style_Transfer_From_Scratch_In_Pytorch/main/style/mondrian.jpg width="200"> | <img src=https://raw.githubusercontent.com/benningtonlee7/AdaIn_Style_Transfer_From_Scratch_In_Pytorch/main/outputs/houses_stylized_mondrian.jpg width="270">|
| | <img src=https://raw.githubusercontent.com/benningtonlee7/AdaIn_Style_Transfer_From_Scratch_In_Pytorch/main/style/asheville.jpg width="200">| <img src=https://raw.githubusercontent.com/benningtonlee7/AdaIn_Style_Transfer_From_Scratch_In_Pytorch/main/outputs/houses_stylized_asheville.jpg width="270">|
| | <img src=https://raw.githubusercontent.com/benningtonlee7/AdaIn_Style_Transfer_From_Scratch_In_Pytorch/main/style/hosi.jpg width="200">| <img src=https://raw.githubusercontent.com/benningtonlee7/AdaIn_Style_Transfer_From_Scratch_In_Pytorch/main/outputs/houses_stylized_hosi.jpg width="270">|


## References
* [1]: X. Huang and S. Belongie. "[Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868)", in ICCV, 2017.
* [2]: [Torch implementation of AdaIN](https://github.com/xunhuang1995/AdaIN-style)
* [3]: [AdaIN Style Transfer](https://github.com/naoto0804/pytorch-AdaIN)
* [4]: [MS-COCO Dataset](http://cocodataset.org/#download)
* [5]: [K. Nichol. Painter by numbers, wikiart. 2016. 5](https://www.kaggle.com/c/painter-by-numbers)
