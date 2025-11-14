Steps for using our method with DRO-LT:

1. Prepare the environment, following the README.md

2. Calculate the normalized ID estimates for each class in the dataset.

3. In the second stage, fine-tune with ID using:
```
PYTHONPATH="./" python main.py --gpu 1 --dataset cifar100 --imb_type exp --imb_factor 0.01 --resume True --epochs 100 --pretrained cifar100_resnet32_CE_None_exp_0.01_0 --feat_sampler resample,4 --feat_lr 0.005 --feat_loss ce_lt,robust_loss --cls_sampler none --cls_lr 0.01 --cls_loss ce --temperature 1 -b 128 --margin 1 --margin_type id --id_values <id1 id2 ... idN>
```
where id1..N are normalized ID scores.

