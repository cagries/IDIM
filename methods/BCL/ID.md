To use BCL + ID:

0. Follow README.md to get BCL up and running

1. Calculate normalized ID weights for the desired ID estimator

2. Specifying the `dataset` (e.g., `imagenet`), an example command to train a ResNeXt with BCL + ID on ImageNet:
```
python main.py --data <path-to-extracted-imagenet-folder> \
  --lr 0.1 -p 200 --epochs 180 \
  --arch resnext50 --use_norm True \
  --wd 5e-4 --cos True \
  --cl_views rand-rand --id_sampling_weight_path <path-to-id-weights.npy>
```

```
