To use SURE + ID:

1. Calculate normalized ID weights for the desired ID estimator

2. Specifying the `dataset` (e.g., such as `Cifar100_LT_100` below), run:

```
# python3 main.py \
--batch-size 128 \
--gpu 0 \
--epochs 200 \
--nb-run 1 \
--model-name resnet32 \
--optim-name fmfp \
--crl-weight 0 \
--mixup-weight 1 \
--mixup-beta 10 \
--use-cosine \
--save-dir ./results/CIFAR100_LT_100_out/res32_out \
Cifar100_LT_100

# python3 finetune.py \
--batch-size 128 \
--gpu 0 \ 
--nb-run 1 \ 
--model-name resnet32 \
--optim-name fmfp \
--fine-tune-lr 0.005 \
--reweighting-type exp \
--t 1 \ 
--crl-weight 0 \ 
--mixup-weight 1 \ 
--mixup-beta 10 \
--fine-tune-epochs 50 \
--use-cosine \
--save-dir ./results/CIFAR100_LT_100_out/res32_out \
--id_sampling_weights_path <path-to-normalized-id-weights.npy> \
Cifar100_LT_100
```
