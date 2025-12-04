To use logit adjustment + ID:

0. Follow README.md to get logit adjustment  up and running

1. Calculate normalized ID weights for the desired ID estimator, or use pre-calculated inverse normalized weights under `data/`

2. An example command to train with our method on CIFAR:
```
python -m logit_adjustment.main --dataset=cifar10-lt --mode=posthoc
```

```
