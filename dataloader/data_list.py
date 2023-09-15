# from __future__ import print_function, division
import pickle

import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import random
from PIL import Image
import torch.utils.data as data
import os
import os.path


def make_dataset(image_list, labels):
    if labels:  
        len_ = len(image_list)
        images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:  
        if len(image_list[0].split()) > 2:
            images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        else:
            images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        return pil_loader(path)


def default_loader(path):
    return pil_loader(path)


class ImageList(object):
    
    def __init__(self, option, image_list, inp_name=None, labels=None, transform=None, target_transform=None,
                 loader=default_loader):  
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))

        self.imgs = imgs
        self.option = option
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        with open(inp_name, 'rb') as f:
            self.inp = pickle.load(f)  
        self.inp_name = inp_name

    def __getitem__(self, index):
        
        path, target = self.imgs[index]

        if self.option.data_name == 'coco':
            path = self.option.data_path + '/data/' + path[10:]
        if self.option.data_name == 'nuswide':
            path = self.option.data_path + path[13:]
        if self.option.data_name == 'imagenet':
            path = self.option.data_path + path[13:]
        if self.option.data_name == 'voc2012':
            path = self.option.data_path + '/' + path
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)


        return (img, path, self.inp), target

    def __len__(self):
        return len(self.imgs)
