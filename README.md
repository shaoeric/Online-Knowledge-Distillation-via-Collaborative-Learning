# Online Knowledge Distillation via Collaborative Learning

A simple reimplement Online Knowledge Distillation via Collaborative Learning with pytorch.



## Training
- Creating `./dataset/cifar100` directory and downloading CIFAR100 in it.

- ```shell
  python train.py --root ./dataset/cifar100/ --model_names resnet56 resnet20 --num_workers 4 --print_freq 10 --gpu-id 0
  ```

## Note

-  data.py refers to SSKD/data.py, making the dataloader yields different transformed images  for the same images. For customized dataset, it is easy to implement, only need to yield the transformed images in your `__getitem__`.
-  The parameters are not completely set according to the original paper.

## Requirements
- python 3.6
- pytorch >= 1.0
- torchvision
- GPU memory 4GB is enough if batch == 64

## Results
|  Teacher   |   Student    |  baseline   |   distill   |
| :--------: | :----------: | :---------: | :---------: |
|  wrn_40_2  |   wrn_16_1   | 75.61/73.26 | 77.53/67.73 |
|  wrn_40_2  |   wrn_40_1   | 75.61/71.98 | 77.75/73.12 |
|  resnet56  |   resnet20   | 72.34/69.06 | 74.40/70.58 |
| resnet110  |   resnet20   | 74.31/69.06 | 76.27/70.36 |
| resnet110  |   resnet32   | 74.31/71.14 |             |
| resnet32x4 |  resnet8x4   | 79.42/72.50 |             |
|   vgg13    |     vgg8     | 74.64/70.36 |             |
|   vgg13    | MobileNetV2  | 74.64/64.6  |             |
|  ResNet50  | MobileNetV2  | 79.34/64.6  |             |
|  ResNet50  |     vgg8     | 79.34/70.36 |             |
| resnet32x4 | ShuffleNetV1 | 79.42/70.5  |             |
| resnet32x4 | ShuffleNetV2 | 79.42/71.82 |             |
|  wrn_40_2  | ShuffleNetV1 | 75.61/70.5  |             |




## Acknowledgements
This repo is partly based on the following repos, thank the authors a lot.
- [HobbitLong/RepDistiller](https://github.com/HobbitLong/RepDistiller)
- [xuguodong03/SSKD](https://github.com/xuguodong03/SSKD)
- [AberHu/Knowledge-Distillation-Zoo](https://github.com/AberHu/Knowledge-Distillation-Zoo)

