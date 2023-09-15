import torch
from torchvision.transforms import transforms
import dataloader.pre_process as prep
from common.utils import MultiScaleCrop, Warp
from dataloader.cifar10.cifar10 import cifar_dataset
from dataloader.data_list import ImageList

def getDataLoader(option):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(option.image_size),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        Warp(option.image_size),
        transforms.ToTensor(),
        normalize,
    ])

    if option.data_name == 'cifar10':
        train_loader, test_loader, database_loader, _, _, _ = cifar_dataset(option)
        return train_loader, test_loader, database_loader

    database_list = '../data/' + option.data_name + '/database.txt'
    test_list = '../data/' + option.data_name + '/test.txt'
    train_list = '../data/' + option.data_name + '/train.txt'
    train_data = ImageList(option, open(train_list).readlines(), option.word2vec_file,
                           transform=prep.image_train(resize_size=255, crop_size=224))
    database = ImageList(option, open(database_list).readlines(), option.word2vec_file,
                         transform=prep.image_test(resize_size=255, crop_size=224))

    database_loader = torch.utils.data.DataLoader(database, batch_size=option.batch_size, shuffle=False,
                                                  num_workers=option.workers)

    test_dataset = ImageList(option, open(test_list).readlines(), option.word2vec_file,
                             transform=prep.image_test(resize_size=255, crop_size=224))

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=option.batch_size, shuffle=False,
                                              num_workers=option.workers)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=option.batch_size,
                                               shuffle=False, num_workers=option.workers)

    return train_loader, test_loader, database_loader
    pass
