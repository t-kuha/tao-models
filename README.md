# tao-models

Local inference using NVIDIA TAO model

## preparation

```shell
# Python environment
$ uv sync

# download TAO model
$ curl -L 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/pretrained_classification/resnet10/files?redirect=true&path=resnet_10.hdf5' -o 'resnet_10.hdf5'

# download label etc.
$ curl -L https://storage.googleapis.com/openimages/2017_11/classes_2017_11.tar.gz -o classes_2017_11.tar.gz
$ curl -L https://storage.googleapis.com/openimages/2017_11/annotations_human_2017_11.tar.gz -o annotations_human_2017_11.tar.gz

$ tar xf annotations_human_2017_11.tar.gz
$ tar xf classes_2017_11.tar.gz  
```

***

## references

- [TAO Pretrained Classification](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/pretrained_classification/version?version=resnet18)
- [Bad results, while running inference on the pretrained Image Classification models](https://forums.developer.nvidia.com/t/bad-results-while-running-inference-on-the-pretrained-image-classification-models/307864/4)

```text
For the public Google Open Images dataset, we reviewed a subset of images in each class and selected 176 classes with less noisy images and larger number of images. Then images with aspect ratio greater than 3 and W or H less than 100 pixels are eliminated. Finally, we got roughly 400K training images.
The pre-trained weights are trained on Open Image datasets, and they provide a much better starting point for training versus starting from a random initialization of weights. Refer to Overview - NVIDIA Docs and Overview - NVIDIA Docs.
```