python train.py --root /data/wyx/datasets/cifar100/ --model_names wrn_40_2 wrn_16_1 --num_workers 4 --print_freq 1 --gpu-id 1

python train.py --root /data/wyx/datasets/cifar100/ --model_names wrn_40_2 wrn_40_1 --num_workers 4 --print_freq 1 --gpu-id 1

python train.py --root /data/wyx/datasets/cifar100/ --model_names resnet56 resnet20 --num_workers 4 --print_freq 1 --gpu-id 1

python train.py --root /data/wyx/datasets/cifar100/ --model_names resnet110 resnet20 --num_workers 4 --print_freq 1 --gpu-id 1

python train.py --root /data/wyx/datasets/cifar100/ --model_names resnet110 resnet32 --num_workers 4 --print_freq 1 --gpu-id 1

python train.py --root /data/wyx/datasets/cifar100/ --model_names resnet32x4 resnet8x4 --num_workers 4 --print_freq 1 --gpu-id 1

python train.py --root /data/wyx/datasets/cifar100/ --model_names vgg13 vgg8 --num_workers 4 --print_freq 1 --gpu-id 1

python train.py --root /data/wyx/datasets/cifar100/ --model_names vgg13 MobileNetV2 --num_workers 4 --print_freq 1 --gpu-id 1

python train.py --root /data/wyx/datasets/cifar100/ --model_names ResNet50 MobileNetV2 --num_workers 4 --print_freq 1 --gpu-id 1

python train.py --root /data/wyx/datasets/cifar100/ --model_names ResNet50 vgg8 --num_workers 4 --print_freq 1 --gpu-id 1

python train.py --root /data/wyx/datasets/cifar100/ --model_names resnet32x4 ShuffleV1 --num_workers 4 --print_freq 1 --gpu-id 1

python train.py --root /data/wyx/datasets/cifar100/ --model_names wrn_40_2 ShuffleV1 --num_workers 4 --print_freq 1 --gpu-id 1