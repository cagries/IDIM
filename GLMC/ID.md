To use GLMC + ID:

1. Follow README.md to get GLMC up and running

2. Calculate normalized ID weights for the desired ID estimator (see `id-estimation`)

An example command to train with our method on CIFAR:
```
python main.py --dataset cifar100 -a resnet32 --num_classes 100 --imbanlance_rate 0.01 --beta 0.5 --lr 0.01 --epochs 200 -b 64 --momentum 0.9 --weight_decay 5e-3 --resample_weighting 0.5 --label_weighting 1.2  --contrast_weight 4 --id_sampling_weights_path <path-to-id-weights.npy>
```
or for ImageNet, one can use

```
python main.py --dataset ImageNet-LT --root data/ImageNet/ -a resnext50_32x4d --num_classes 1000 --beta 0.5 --lr 0.1 --epochs 135 -b 120 --momentum 0.9 --weight_decay 2e-4 --resample_weighting 0.2 --label_weighting 1.0 --contrast_weight 10 --id_sampling_weights_path <path-to-id-weights.npy>
```
