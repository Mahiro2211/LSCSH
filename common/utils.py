import math, os, sys
import pickle
from urllib.request import urlretrieve
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
import random

# DEBUG switch
from common.logger import Logger

DEBUG_UTIL = False


def getDatabaseHashPoolPath(option, state):
    time = Logger.getTimeStr(state['start_time'])
    path = "../data/" + option.data_name + "/" + option.data_name + "_" + str(option.hash_bit) + "bit_" + str(
        state['epoch']) + "e_" + time + "_database.pkl"
    return path


def getTrainbaseHashPoolPath(option, state):
    time = Logger.getTimeStr(state['start_time'])
    path = "../data/" + option.data_name + "/" + option.data_name + "_" + str(option.hash_bit) + "bit_" + str(
        state['epoch']) + "e_" + time + "_trainbase.pkl"
    return path


def getWeightBestPath(option, state):
    time = Logger.getTimeStr(state['start_time'])
    path = "../data/" + option.data_name + "/" + option.data_name + "_" + time + "_weight.npy"
    return path


def loadHashPool(option, state, path, type='testbase'):
    file = open(path, 'rb')
    start = True
    if type == 'testbase':
        while True:
            try:
                data = pickle.load(file)
                hashcode_batch = data['output'].cpu()
                hashcode_batch.require_grad = False
                label_batch = data['target'].cpu()
                label_batch.require_grad = False
                if start:
                    hash_pool = hashcode_batch
                    labels = label_batch
                    start = False
                else:
                    hash_pool = torch.cat((hash_pool, hashcode_batch), dim=0)
                    labels = torch.cat((labels, label_batch), dim=0)
            except Exception:
                break
        return hash_pool, labels
    elif type == 'database':
        while True:
            try:
                data = pickle.load(file)
                hashcode_batch = data['output'].cpu()
                label_batch = data['target'].cpu()
                hashcode_batch.require_grad = False
                label_batch.require_grad = False
                if start:
                    hash_pool = hashcode_batch
                    labels = label_batch
                    start = False
                else:
                    hash_pool = torch.cat((hash_pool, hashcode_batch), dim=0)
                    labels = torch.cat((labels, label_batch), dim=0)
            except Exception:
                break
        return hash_pool, labels


def getTestbaseHashPoolPath(option, state):
    time = Logger.getTimeStr(state['start_time'])

    path = "../data/" + option.data_name + "/" + option.data_name + "_" + str(option.hash_bit) + "bit_" + str(
        state['epoch']) + "e_" + time + "_testbase.pkl"
    return path


class Warp(object):
    def __init__(self, size, interpolation=Image.BILINEAR):

        self.size = int(size)
        self.interpolation = interpolation

    def __call__(self, img):

        return img.resize((self.size, self.size), self.interpolation)

    def __str__(self):
        return self.__class__.__name__ + ' (size={size}, interpolation={interpolation})'.format(size=self.size,
                                                                                                interpolation=self.interpolation)


class MultiScaleCrop(object):

    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):

        self.scales = scales if scales is not None else [1, .875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
        self.interpolation = Image.BILINEAR  

    def __call__(self, img):

        im_size = img.size
        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h))
        ret_img_group = crop_img_group.resize((self.input_size[0], self.input_size[1]), self.interpolation)
        return ret_img_group

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]  
        crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):

        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  
        ret.append((4 * w_step, 0))  
        ret.append((0, 4 * h_step))  
        ret.append((4 * w_step, 4 * h_step))  
        ret.append((2 * w_step, 2 * h_step))  

        if more_fix_crop:
            ret.append((0, 2 * h_step))  
            ret.append((4 * w_step, 2 * h_step))  
            ret.append((2 * w_step, 4 * h_step))  
            ret.append((2 * w_step, 0 * h_step))  

            ret.append((1 * w_step, 1 * h_step))  
            ret.append((3 * w_step, 1 * h_step))  
            ret.append((1 * w_step, 3 * h_step))  
            ret.append((3 * w_step, 3 * h_step))  

        return ret

    def __str__(self):
        return self.__class__.__name__


def download_url(url, destination=None, progress_bar=True):
    

    def my_hook(t):
        last_b = [0]

        def inner(b=1, bsize=1, tsize=None):
            if tsize is not None:
                t.total = tsize
            if b > 0:
                t.update((b - last_b[0]) * bsize)
            last_b[0] = b

        return inner

    if progress_bar:
        with tqdm(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
            filename, _ = urlretrieve(url, filename=destination, reporthook=my_hook(t))
    else:
        filename, _ = urlretrieve(url, filename=destination)
