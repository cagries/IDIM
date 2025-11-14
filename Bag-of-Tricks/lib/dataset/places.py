import math
import os
import random
import time

import cv2
import torchvision
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from utils.utils import get_category_list


# class IMBALANCEPLACES(torchvision.datasets.Places365):
#     """
#     Class-imbalanced version of Places365 with varying imbalance factor.
#     """
#
#     def __init__(self,
#                  mode,
#                  cfg,  # General config file
#                  transform=None,
#                  target_transform=None,
#                  small=True,
#                  download=False):
#         """
#         Initialize an imbalanced Places365 dataset.
#         Args:
#             mode:       One of ['train', 'val']
#             cfg:        A configuration YAML file
#             root:       Root path to Places365 data
#             transform:  Torch transforms to be applied to the data
#             small:      Whether to use the small (224x224) images
#             download:   True if we need to download data
#         """
#
#         self.epoch = None
#         train = mode == "train"
#         split = "train-standard" if mode == "train" else "val"
#         root = cfg.DATASET.ROOT
#         super().__init__(root, split=split, small=small, transform=transform, download=download)
#         self.cfg = cfg
#         self.train = train
#         self.cfg = cfg
#         self.input_size = cfg.INPUT_SIZE
#         self.color_space = cfg.COLOR_SPACE
#         self.target_transform = target_transform
#
#         # Specific to Places
#         self.imgs = np.array(self.imgs)
#         self.cls_num = max(self.targets) + 1
#
#         print(f'Preparing Places-LT in mode: {mode}')
#         print("Use {} Mode to train network".format(self.color_space))
#
#         rand_number = cfg.DATASET.IMBALANCEPLACES.RANDOM_SEED
#         if self.train:
#             np.random.seed(rand_number)
#             random.seed(rand_number)
#             imb_factor = self.cfg.DATASET.IMBALANCEPLACES.RATIO
#             print('Using random seed: {}'.format(rand_number))
#             print('Using imbalance ratio: {}'.format(imb_factor))
#             img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type='exp', imb_factor=imb_factor)
#             self.gen_imbalanced_data(img_num_list)
#             # Using the transforms in PlacesLT
#             self.transform = transforms.Compose([
#                 transforms.RandomResizedCrop(224),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#             ])
#         else:
#             self.data_format_transform()
#             # Using the test transforms in Places-LT
#             self.transform = transforms.Compose([
#                 transforms.Resize(256),
#                 transforms.CenterCrop(224),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#             ])
#
#         self.data = self.all_info
#         print("{} Mode: Contain {} images".format(mode, len(self.data)))
#
#         self.class_weight, self.sum_weight = self.get_weight(self.get_annotations(), self.cls_num)
#         if self.cfg.TRAIN.SAMPLER.TYPE == "weighted sampler" and self.train:
#
#             self.class_dict = self._get_class_dict()
#
#             print('-' * 20 + 'in imbalanced Places dataset' + '-' * 20)
#             # print('class_dict is: ')
#             # print(self.class_dict)
#             print('class_weight is: ')
#             print(self.class_weight)
#             print('cls_num is: ')
#             print(self.cls_num)
#
#             num_list, cat_list = get_category_list(self.get_annotations(), self.cls_num, self.cfg)
#             self.instance_p = np.array([num / sum(num_list) for num in num_list])
#             self.class_p = np.array([1 / self.cls_num for _ in num_list])
#             num_list = [math.sqrt(num) for num in num_list]
#             self.square_p = np.array([num / sum(num_list) for num in num_list])
#             self.class_dict = self._get_class_dict()
#
#             if self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE == "custom":
#                 # Need to normalize for probabilities
#                 self.custom_p = np.array(self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.CUSTOM.WEIGHTS) / self.cls_num
#                 print('Custom sampling probabilities:', self.custom_p)
#                 print(self.custom_p.sum())
#         else:
#             print('Not using weighted sampling')
#
#     def update(self, epoch):
#         self.epoch = max(0, epoch - self.cfg.TRAIN.TWO_STAGE.START_EPOCH) if self.cfg.TRAIN.TWO_STAGE.DRS else epoch
#         if self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE == "progressive":
#             self.progress_p = epoch / self.cfg.TRAIN.MAX_EPOCH * self.class_p + (
#                     1 - epoch / self.cfg.TRAIN.MAX_EPOCH) * self.instance_p
#             print('self.progress_p', self.progress_p)
#
#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index
#
#         Returns:
#             tuple: (image, target) where target is index of the target class.
#         """
#         if self.cfg.TRAIN.SAMPLER.TYPE == "weighted sampler" and self.train \
#                 and (not self.cfg.TRAIN.TWO_STAGE.DRS or (self.cfg.TRAIN.TWO_STAGE.DRS and self.epoch)):
#             assert self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE in ["custom", "bayesian", "balance", 'square',
#                                                                     'progressive']
#             if self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE == "balance":
#                 sample_class = random.randint(0, self.cls_num - 1)
#             elif self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE == "bayesian":
#                 sample_class = np.random.choice(np.arange(self.cls_num), p=self.class_p)
#             elif self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE == "custom":
#                 sample_class = np.random.choice(np.arange(self.cls_num), p=self.custom_p)
#             elif self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE == "square":
#                 sample_class = np.random.choice(np.arange(self.cls_num), p=self.square_p)
#             else:
#                 sample_class = np.random.choice(np.arange(self.cls_num), p=self.progress_p)
#             sample_indexes = self.class_dict[sample_class]
#             index = random.choice(sample_indexes)
#
#         img, target = self.data[index]['image'], self.data[index]['category_id']
#         meta = dict()
#
#         # doing this so that it is consistent with all other datasets
#         # to return a PIL Image
#         img = Image.open(img)
#
#         if self.transform is not None:
#             img = self.transform(img)
#
#         if self.target_transform is not None:
#             target = self.target_transform(target)
#
#         return img, target, meta
#
#     def sample_class_index_by_weight(self):
#         rand_number, now_sum = random.random() * self.sum_weight, 0
#         for i in range(self.cls_num):
#             now_sum += self.class_weight[i]
#             if rand_number <= now_sum:
#                 return i
#
#     def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
#         img_max = len(self.imgs) / cls_num
#         img_num_per_cls = []
#         if imb_type == 'exp':
#             for cls_idx in range(cls_num):
#                 num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
#                 img_num_per_cls.append(int(num))
#         elif imb_type == 'step':
#             for cls_idx in range(cls_num // 2):
#                 img_num_per_cls.append(int(img_max))
#             for cls_idx in range(cls_num // 2):
#                 img_num_per_cls.append(int(img_max * imb_factor))
#         else:
#             img_num_per_cls.extend([int(img_max)] * cls_num)
#         return img_num_per_cls
#
#     def reset_epoch(self, cur_epoch):
#         self.epoch = cur_epoch
#
#     def _get_class_dict(self):
#         class_dict = dict()
#         for i, anno in enumerate(self.data):
#             cat_id = anno["category_id"]
#             if cat_id not in class_dict:
#                 class_dict[cat_id] = []
#             class_dict[cat_id].append(i)
#         return class_dict
#
#     def get_weight(self, annotations, num_classes):
#         num_list = [0] * num_classes
#         cat_list = []
#         for anno in annotations:
#             category_id = anno["category_id"]
#             num_list[category_id] += 1
#             cat_list.append(category_id)
#         max_num = max(num_list)
#         class_weight = [max_num / i for i in num_list]
#         sum_weight = sum(class_weight)
#         return class_weight, sum_weight
#
#     def _get_trans_image(self, img_idx):
#         now_info = self.data[img_idx]
#         img = now_info['image']
#         img = Image.fromarray(img)
#         return self.transform(img)[None, :, :, :]
#
#     def get_num_classes(self):
#         return self.cls_num
#
#     def get_annotations(self):
#         annos = []
#         for d in self.all_info:
#             annos.append({'category_id': int(d['category_id'])})
#         return annos
#
#     def imread_with_retry(self, fpath):
#         retry_time = 10
#         for k in range(retry_time):
#             try:
#                 img = cv2.imread(fpath)
#                 if img is None:
#                     print("img is None, try to re-read img")
#                     continue
#                 return img  # .convert('RGB')
#             except Exception as e:
#                 if k == retry_time - 1:
#                     assert False, "pillow open {} failed".format(fpath)
#                 time.sleep(0.1)
#
#     def _get_image(self, now_info):
#         fpath = os.path.join(now_info["fpath"])
#         img = self.imread_with_retry(fpath)
#
#         if self.color_space == "RGB":
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         return img
#
#     def gen_imbalanced_data(self, img_num_per_cls):
#         new_data = []
#         targets_np = np.array(self.targets, dtype=np.int64)
#         classes = np.unique(targets_np)
#         # np.random.shuffle(classes)
#         self.num_per_cls_dict = dict()
#         for the_class, the_img_num in zip(classes, img_num_per_cls):
#             self.num_per_cls_dict[the_class] = the_img_num
#             idx = np.where(targets_np == the_class)[0]
#             np.random.shuffle(idx)
#             selec_idx = idx[:the_img_num]
#             for img, _ in self.imgs[selec_idx, ...]:
#                 new_data.append({
#                     'image': img,
#                     'category_id': the_class
#                 })
#         self.all_info = new_data
#
#     def data_format_transform(self):
#         new_data = []
#         targets_np = np.array(self.targets, dtype=np.int64)
#         assert len(targets_np) == len(self.imgs)
#         for i in range(len(self.imgs)):
#             new_data.append({
#                 'image': self.imgs[i][0],
#                 'category_id': targets_np[i],
#             })
#         self.all_info = new_data
#
#     def __len__(self):
#         return len(self.data)
