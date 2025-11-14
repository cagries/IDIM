import os
import sys
import argparse

import skdim
import tqdm

import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset

from PIL import Image

def estimate_id(data, method):
    method = method.lower()
    if method == 'fishers':
        estimator = skdim.id.FisherS()
    elif method == 'tle':
        estimator = skdim.id.TLE()
    elif method == 'mle':
        estimator = skdim.id.MLE()
    else:
        raise NotImplementedError
    data = data.reshape(data.shape[0], -1)
    return estimator.fit_transform(data)

# OLTR-style ImageNet-LT
class ImageNetLT(Dataset):
    def __init__(self, root, txt):
        self.img_path = []
        self.labels = []
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        path = self.img_path[index]
        label = self.labels[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label

if __name__ == '__main__':
    imagenet_root = '/path/to/extracted/imagenet'
    imagenet_data = ImageNetLT(root=imagenet_root,
                           txt='ImageNet_LT_train.txt')
    num_classes = 1000
    method = 'mle'  # 'tle', 'fishers'
    targets = np.array(imagenet_data.labels)
    ids = np.zeros(shape=(num_classes,))
    for i in tqdm.tqdm(range(num_classes), smoothing=0):
        indices = np.where(targets == i)[0]
        class_data = np.array([imagenet_data[index][0].numpy() for index in indices])
        ids[i] = estimate_id(class_data, method)
        np.save(f'imagenetlt_id_{method}.npy', ids)

