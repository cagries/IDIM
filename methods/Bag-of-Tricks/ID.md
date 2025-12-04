To use Bag of Tricks + ID:

0. Follow the README.md to get Bag of Tricks up and running

1. Calculate normalized ID weights for the desired ID estimator (see top-level `id-estimation` directory)

3. Create or use a config (such as those in `configs/`) in which the normalized ID values are used via resampling or re-weighting

An example command to use ID-based training:
```
$ python main/train.py --cfg configs/cifar10/cifar10_imb100_fish.yaml
```
or for CIFAR-100:
```
$ python main/train.py --cfg configs/cifar100/cifar100_imb10.yaml
```
