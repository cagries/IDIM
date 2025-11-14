import os
import sys
import argparse

import skdim
import tqdm

import numpy as np
import torch 

from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100

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

class IMBCIFAR10(torch.utils.data.Dataset):
    def __init__(self, imb_factor, **kwargs):
        # Re-balance according to the imbalance factor
        def rebalance(data_x, data_y, imbalance_factor):
            num_ex = len(data_x)
            num_classes = 10
            class_example_limits = np.zeros(num_classes)
            for i in range(num_classes):
                class_example_limits[i] = int((num_ex // num_classes) * (imbalance_factor ** (i / (num_classes - 1.0))))
            print(class_example_limits)

            num_picked = np.zeros(num_classes)
            new_x = []
            new_y = []
            for i in tqdm.tqdm(range(num_ex)):
                if num_picked[data_y[i]] < class_example_limits[data_y[i]]:
                    new_x.append(data_x[i])
                    new_y.append(data_y[i])
                    num_picked[data_y[i]] += 1
            return np.array(new_x), np.array(new_y)
        
        self.cifar = CIFAR10(**kwargs)
        if kwargs.get('train'):
            print('Rebalancing train set')
            X, y = rebalance(self.cifar.data, self.cifar.targets, imb_factor)
            self.data = X
            self.targets = y
        else:
            print('Not rebalancing test set')
            self.data = self.cifar.data
            self.targets = self.cifar.targets
        if kwargs.get('transform'):
            self.transform = kwargs.get('transform')
        else:
            self.transform = None
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

class IMBCIFAR100(torch.utils.data.Dataset):
    def __init__(self, imb_factor, **kwargs):
        # Re-balance according to the imbalance factor
        def rebalance(data_x, data_y, imbalance_factor):
            num_ex = len(data_x)
            num_classes = 100
            class_example_limits = np.zeros(num_classes)
            for i in range(num_classes):
                class_example_limits[i] = int((num_ex // num_classes) * (imbalance_factor ** (i / (num_classes - 1.0))))
            print(class_example_limits)

            num_picked = np.zeros(num_classes)
            new_x = []
            new_y = []
            for i in tqdm.tqdm(range(num_ex)):
                if num_picked[data_y[i]] < class_example_limits[data_y[i]]:
                    new_x.append(data_x[i])
                    new_y.append(data_y[i])
                    num_picked[data_y[i]] += 1
            return np.array(new_x), np.array(new_y)
        
        self.cifar = CIFAR100(**kwargs)
        
        # Rebalance here
        if kwargs.get('train'):
            X, y = rebalance(self.cifar.data, self.cifar.targets, imb_factor)
            self.data = X
            self.targets = y
        else:
            self.data = self.cifar.data
            self.targets = self.cifar.targets
        if kwargs.get('transform'):
            self.transform = kwargs.get('transform')
        else:
            self.transform = None
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

if __name__ == '__main__':
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Modify below for CIFAR10/CIFAR100 and different imbalance factors
    imb_factor = 0.1
    dataset = 'CIFAR100'  # 'cifar10'
    method = 'fishers'  # 'tle', 'fishers'

    if dataset == 'CIFAR10':
        num_classes = 10
        cifar_data = IMBCIFAR10(imb_factor=imb_factor, root='.', train=True, download=True, transform=preprocess)
    else:
        num_classes = 100 
        cifar_data = IMBCIFAR100(imb_factor=imb_factor, root='.', train=True, download=True, transform=preprocess)
    data = cifar_data.data
    data = data.reshape(data.shape[0], -1)
    labels = cifar_data.targets
    ids = np.zeros(shape=(num_classes,))
    for i in tqdm.tqdm(range(num_classes), smoothing=0):
        indices = np.where(labels == i)[0]
        samples = data[indices]
        ids[i] = estimate_id(samples, method)
    print(f'IDs: {ids}')
    normalized_ids = (ids / ids.sum()) * num_classes
    np.save(f'cifarlt_{dataset}_id_{method}.npy', ids)

