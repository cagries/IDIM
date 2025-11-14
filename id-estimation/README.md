We calculate ID estimates using the `scikit-dimension` library (Bac et al.). The scripts in this directory are to assist in estimating ID from various estimators (FisherS, MLE, TLE) on long-tailed datasets (CIFAR-LT, Places-LT, ImageNet-LT). These scripts can be run via:
```
$ python id_{cifar,places,imagenet}.py
```
Note that the Places-LT and ImageNet-LT needs to be downloaded and prepared beforehand. See the OLTR dataset (https://github.com/zhmiao/OpenLongTailRecognition-OLTR) on instructions to set up long-tailed variants of these datasets.
