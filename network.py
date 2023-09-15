import os

import torch.nn as  nn
import torchvision
import torch


class CenterModel(nn.Module):
    def __init__(self, option):
        super(CenterModel, self).__init__()
        self.option = option
        self.class_num_dict = {'voc2012': 20, 'coco': 80, 'nuswide': 21, 'imagenet': 100, 'cifar10': 10}
        self.hash_bit = option.hash_bit
        self.last_layer = nn.Tanh()
        if self.option.center_update:
            self.to_center = nn.Sequential(nn.Linear(option.w2v_dim, 256), nn.ReLU(), nn.Linear(256, self.hash_bit),
                                           self.last_layer)

    def forward(self, word_embeddings):
        if self.option.center_update:
            hash_centers = self.to_center(word_embeddings.float())
        else:
            file_path = '../data/' + self.option.data_name + '/' + str(
                self.hash_bit) + '_' + self.option.data_name + '_' + str(
                self.class_num_dict[self.option.data_name]) + '_class.pkl'
            if os.path.exists(file_path):
                center_file = open(file_path, 'rb')
                hash_centers = torch.load(center_file)
            elif os.path.exists(self.option.centers_path):
                center_file = open(self.option.ceters_path, 'rb')
                hash_centers = torch.load(center_file)
        return hash_centers.cuda() if self.option.use_gpu and torch.cuda.is_available() else hash_centers, word_embeddings

    def getConfig_params(self):
        if self.option.center_update:
            return [
                {'params': self.to_center.parameters(), 'lr': self.option.lr_center},
            ]
        else:
            return []


class HashModel(nn.Module):
    def __init__(self, option):
        super(HashModel, self).__init__()
        self.option = option
        self.hash_bit = option.hash_bit
        self.base_model = getattr(torchvision.models, option.model_type)(pretrained=True)
        self.conv1 = self.base_model.conv1
        self.bn1 = self.base_model.bn1
        self.relu = self.base_model.relu
        self.maxpool = self.base_model.maxpool
        self.layer1 = self.base_model.layer1
        self.layer2 = self.base_model.layer2
        self.layer3 = self.base_model.layer3
        self.layer4 = self.base_model.layer4
        self.avgpool = self.base_model.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                                            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)
        self.fc1 = nn.Linear(self.base_model.fc.in_features, self.base_model.fc.in_features)
        self.activation1 = nn.ReLU()
        self.fc2 = nn.Linear(self.base_model.fc.in_features, self.base_model.fc.in_features)
        self.activation2 = nn.ReLU()
        self.fc3 = nn.Linear(self.base_model.fc.in_features, self.hash_bit)
        self.last_layer = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.hash_layer = nn.Sequential(self.fc1, self.activation1, self.dropout, self.fc2, self.activation2, self.fc3,
                                        self.last_layer)

    def forward(self, images):
        features = self.feature_layers(images)
        features = features.view(features.size(0), -1)
        hash_codes = self.hash_layer(features)
        return hash_codes

    def getConfig_params(self):
        return [
            {'params': self.feature_layers.parameters(), 'lr': self.option.lr * self.option.multi_lr},
            {'params': self.hash_layer.parameters(), 'lr': self.option.lr},
        ]


class AlexNetFc(nn.Module):
    def __init__(self, option):
        self.option = option
        super(AlexNetFc, self).__init__()
        self.base_model = torchvision.models.alexnet(pretrained=True)
        self.features = self.base_model.features
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module("classifier" + str(i), self.base_model.classifier[i])
        self.feature_layers = nn.Sequential(self.features, self.classifier)

        self.hash_bit = option.hash_bit
        feature_dim = self.base_model.classifier[6].in_features
        self.fc1 = nn.Linear(feature_dim, feature_dim)
        self.activation1 = nn.ReLU()
        self.fc2 = nn.Linear(feature_dim, feature_dim)
        self.activation2 = nn.ReLU()
        self.fc3 = nn.Linear(feature_dim, self.hash_bit)
        self.last_layer = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.hash_layer = nn.Sequential(self.fc1, self.activation1, self.fc2, self.activation2, self.fc3,
                                        self.last_layer)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        y = self.hash_layer(x)

        return y

    def getConfig_params(self):
        return [
            {'params': self.feature_layers.parameters(), 'lr': self.option.lr * self.option.multi_lr},
            {'params': self.hash_layer.parameters(), 'lr': self.option.lr},

        ]
