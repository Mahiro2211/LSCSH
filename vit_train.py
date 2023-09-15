import datetime
import os
import pickle
import shutil
import sys

import torch

from Loss import Loss, HashCenterLoss
from common import utils
from common.plot import Draw
from dataloader.DataSet_loader import getDataLoader
from evaluate.measure_utils import *
from vit_modeling import VisionTransformer, VIT_CONFIGS
from network import HashModel, CenterModel, AlexNetFc

import torch.optim as optim
from options import parser

class Engine(object):
    def __init__(self, option, state, config):
        self.option = option
        self.state = state
        self.class_num_dict = {'voc2012': 20, 'coco': 80, 'nuswide': 21}
        self.is_multi_label = True \
            if option.data_name in ['voc2012','coco', 'nuswide'] else False
        self.add_part_loss = False

        self.device = torch.device(option.device)
        self.running_use_gpu = False

    def useGPU(self, x):
        if self.option.use_gpu and self.option.device!='cpu' and torch.cuda.is_available():
            return x.cuda()
        else:
            return x

    def main(self):
        if torch.cuda.is_available() is False:
            Logger.info('No GPU available in this machine...\n')
            self.device = 'cpu'
            self.running_use_gpu = False
        else:
            if self.state['use_gpu']==False or self.state['device']=='cpu':
                self.device = 'cpu'
                self.running_use_gpu = False
                Logger.info('\tUse CPU for running...')
            else:
                self.running_use_gpu = True
                Logger.info('\tUse GPU for running...')

        Logger.info('self.device: {},  self.state["use_gpu"]: {}, self.running_use_gpu: {}'.\
                    format(self.device, self.state['use_gpu'], self.running_use_gpu)  )

        train_loader, test_loader, database_loader = getDataLoader(self.option)
        vit_config = VIT_CONFIGS["ViT-B_16"]
        Logger.info('vit_config:\t{}'.format('ViT-B_16'))
        hash_model = VisionTransformer(vit_config, img_size=option.image_size, zero_head=True,
                                       num_classes=option.num_class, hash_bit=option.hash_bit, option=self.option)

        hash_model.load_from(np.load(self.state["pretrained_dir"]))

        if self.option.center_model=='center':
            center_model = CenterModel(self.option)

        else:
            assert False, "No such a center model type\n"

        criterion = Loss(self.option, self.state)

        if self.option.center_update:
            criterion_center = HashCenterLoss(self.option, self.state)
        else:
            criterion_center = None

        optimizer_hash = torch.optim.Adam(hash_model.getConfig_params(), lr=option.lr, weight_decay=10 ** -5)
        if self.option.center_update:
            optimizer_center = torch.optim.Adam(center_model.getConfig_params(), lr=option.lr_center)
        else:
            optimizer_center = None

        if option.resume:
            start_epoch = self.resume(hash_model, center_model, optimizer_hash, optimizer_center)
        else:
            start_epoch = -1

        if self.option.use_gpu and torch.cuda.is_available():
            hash_model = torch.nn.DataParallel(hash_model,
                                               device_ids=[int(i) for i in range(torch.cuda.device_count())]).cuda()
            center_model = torch.nn.DataParallel(center_model,
                                                 device_ids=[int(i) for i in range(torch.cuda.device_count())]).cuda()
            criterion = criterion.cuda()
            if self.option.center_update:
                criterion_center = criterion_center.cuda()

        self.run_epoch(hash_model, center_model, criterion, criterion_center, optimizer_hash, optimizer_center,
                       train_loader, test_loader, database_loader, start_epoch)

    def run_epoch(self, hash_model, center_model, criterion, criterion_center, optimizer_hash, optimizer_center,
                  train_loader,
                  test_loader, database_loader, start_epoch):

        centerWeight_train = self.initCenterWeight(train_loader)

        self.state['best_MAP'] = 0.0
        self.state['best_epoch'] = 0
        self.state['Database_hashpool_path'] = None
        self.state['Testbase_hashpool_path'] = None
        self.state['Trainbase_hashpool_path'] = None
        self.state['final_result'] = None
        self.state['filename_previous_best'] = None

        self.state['interClass_loss'] = []

        Logger.divider("start training..")
 
        for epoch in range(start_epoch + 1, self.option.epochs):

            lr = self.adjust_learning_rate(optimizer_hash, epoch)
            lr_center = self.adjust_learning_rate(optimizer_center, epoch, 'centerNet')
            Logger.divider("epoch[{}] lr:{} lr_center:{}".format(epoch, lr, lr_center))

            if optimizer_center is not None:
                optimizer_center.zero_grad()
            self.on_start_epoch(epoch)
            ''''
            obtain hash centers by center_model
            '''

            hashCenter_pre, word_embedding = self.forward_hashCenter(center_model, train_loader)
            self.state['hash_center_pre'] = hashCenter_pre

            loss_epoch = self.train(hash_model, train_loader, criterion, hashCenter_pre, optimizer_hash,
                                    centerWeight_train, epoch)

            Logger.info("\tEpoch : {}, Mean Loss {}\n".format(epoch, np.mean(loss_epoch)))


            is_best = False
            if epoch >= self.option.start_test_epoch:
                (Precision, Recall1, MAP1), (MAP2, Recall2, Precision2) = self.test(hash_model, train_loader,
                                                                                    test_loader,
                                                                                    database_loader,
                                                                                    centerWeight_train, epoch)
                Logger.info(
                    "epoch {0} Result：{1}\n".format(epoch, ((Precision, Recall1, MAP1), (MAP2, Recall2, Precision2))))
 
                if self.option.center_update:

                    self.updateCenter(criterion_center, word_embedding, centerWeight_train, hashCenter_pre,
                                      optimizer_center)

                self.saveStatus(epoch, centerWeight_train, hashCenter_pre, MAP2,
                                ((Precision, Recall1, MAP1), (MAP2, Recall2, Precision2)))


                time = Logger.getTimeStr(state['start_time'])
                path = "../data/" + option.data_name + "/" + option.data_name + "_" + time + "_" + str(
                    epoch) + "_weight.npy"
                np.save(path, centerWeight_train)

                is_best = MAP2 >= self.state['best_MAP']
                Logger.info("MAP epoch {}\tMAP_best {}\tIs_best {}\tBest epoch {}".format(MAP2, self.state['best_MAP'],
                                                                                          MAP2 >= self.state[
                                                                                              'best_MAP'],
                                                                                          self.state['best_epoch']))

            self.on_end_epoch(option, state, epoch, hash_model, center_model, is_best, optimizer_hash, optimizer_center)

        Logger.info("start drawing ...")
        Logger.info(
            "Hash Pool Radius :{}\nMAP1 :{:.4f}\t Recall1 {:.4f}\tPrecision1 {:.4f}\t MAP2 {:.4f} \t Recall2 {:.4f} "
            "\t Precision2 {:.4f} ".format(
                option.R, self.state['final_result'][1][0], self.state['final_result'][1][1],
                self.state['final_result'][1][2],
                self.state['final_result'][0][2],
                self.state['final_result'][0][1], self.state['final_result'][0][0]
            ))

        pass

    def updateCenter(self, criterion_center, word_embedding, weightCenter, hashCenter_pre, optimizer):
        optimizer.zero_grad()
        Logger.info("\t<==update center==>")

        loss = criterion_center(word_embedding, hashCenter_pre,
                                weightCenter, utils.getTrainbaseHashPoolPath(self.option, self.state))
        Logger.info("adapter loss {}".format(loss.item()))
        loss.backward()
        optimizer.step()

    def on_start_epoch(self, epoch):
        self.state['epoch'] = epoch

        pass

    def on_end_epoch(self, option, state, epoch, model_hash, model_center, is_best, optimizer_hash, optimizer_center):

        model_dict = {
            'epoch': epoch,
            'model_hash_dict': model_hash.module.state_dict() if option.use_gpu and torch.cuda.is_available() else model_hash.state_dict(),
            'model_center_dict': model_center.module.state_dict() if option.use_gpu and torch.cuda.is_available() else model_center.state_dict(),
            'optimizer_hash_dict': optimizer_hash.state_dict(),
            'optimizer_center_dict': optimizer_center.state_dict(),
            'best_MAP': state['best_MAP']
        }
        self.save_checkpoint(option, state, model_dict, is_best)


    def train(self, model, train_loader, criterion, hash_center_pre, optimizer, centerWeight, epoch):
        model.train()


        loss_epoch = []

        train_loader = tqdm(train_loader, desc="Epoch [" + str(epoch) + "]==>Training:")
        for i, (input, target) in enumerate(train_loader):
            images = input[0]

            hash_code = model(images)

            centerWeight_batch = self.useGPU(
                torch.tensor(centerWeight[i * self.option.batch_size:(i + 1) * self.option.batch_size],
                             requires_grad=True))

            hash_centroid = self.getHashCenters(target, hash_center_pre.detach(), centerWeight_batch)


            if not self.option.fixed_weight:
                optimizer.zero_grad()
                centerWeight_batch.retain_grad()

                loss = criterion(hash_code.detach(), hash_centroid, hash_center_pre.detach(), centerWeight_batch,
                                 target)
                loss.backward(retain_graph=True)
                weight_grad = torch.where(torch.isnan(centerWeight_batch.grad), self.useGPU(torch.tensor(0.)),
                                          centerWeight_batch.grad)
                centerWeight_batch = centerWeight_batch - self.option.eta * weight_grad
                centerWeight_batch = self.simplexPro(centerWeight_batch)

                centerWeight[
                i * self.option.batch_size:(i + 1) * self.option.batch_size] = centerWeight_batch.cpu().detach().numpy()

            optimizer.zero_grad()

            loss = criterion(hash_code, hash_centroid.detach(), hash_center_pre.detach(), centerWeight_batch.detach(),
                             target)
            loss.backward()
            optimizer.step()
            loss_epoch.append(loss.data.cpu().numpy())
        return loss_epoch


    def test(self, model, train_loader, test_loader, database_loader, centerWeight, epoch):
        model.eval()
        self.predict_hash_code(model, database_loader, centerWeight, epoch, database_type="database")
        self.predict_hash_code(model, test_loader, centerWeight, epoch, database_type='testbase')
        self.predict_hash_code(model, train_loader, centerWeight, epoch, database_type='trainbase')


        database_hashcode, database_labels = utils.loadHashPool(self.option, self.state,
                                                                utils.getDatabaseHashPoolPath(self.option,
                                                                                              self.state),
                                                                'database')
        testbase_hashcode, testbase_labels = utils.loadHashPool(self.option, self.state,
                                                                utils.getTestbaseHashPoolPath(self.option, self.state))

        database_hashcode_numpy = database_hashcode.detach().cpu().numpy().astype('float32')

        del database_hashcode
        database_labels_numpy = database_labels.detach().cpu().numpy().astype('int8')

        del database_labels
        testbase_hashcode_numpy = testbase_hashcode.detach().cpu().numpy().astype('float32')

        del testbase_hashcode
        testbase_labels_numpy = testbase_labels.detach().cpu().numpy().astype('int8')

        del testbase_labels
        Logger.info("===> start calculate MAP!\n")


        Precision, Recall1, MAP1 = get_precision_recall_by_Hamming_Radius_optimized(
            database_hashcode_numpy,
            database_labels_numpy,
            testbase_hashcode_numpy,
            testbase_labels_numpy, fine_sign=True)
        MAP2, Recall2, P = mean_average_precision(database_hashcode_numpy,
                                                  testbase_hashcode_numpy,
                                                  database_labels_numpy,
                                                  testbase_labels_numpy, self.option)

        del database_hashcode_numpy, testbase_hashcode_numpy, database_labels_numpy, testbase_labels_numpy
        return (Precision, Recall1, MAP1), (MAP2, Recall2, P)

    def predict_hash_code(self, model, data_loader, centerWeight, epoch, database_type: str):
        model.eval()
        if database_type == 'database':
            file_path = open(utils.getDatabaseHashPoolPath(option, self.state), 'ab')
        elif database_type == 'testbase':
            file_path = open(utils.getTestbaseHashPoolPath(option, self.state), 'ab')
        elif database_type == 'trainbase':
            file_path = open(utils.getTrainbaseHashPoolPath(option, self.state), 'ab')

        data_loader = tqdm(data_loader, desc="epoch[" + str(epoch) + "]==>" + database_type + "==>Testing:")
        for i, (input, target) in enumerate(data_loader):
            images = input[0]
            hash_code = model(images)
            if database_type == 'trainbase':
                centerWeight_batch = self.useGPU(
                    torch.tensor(centerWeight[i * self.option.batch_size:(i + 1) * self.option.batch_size]))
                center = self.getHashCenters(target, self.state['hash_center_pre'], centerWeight_batch)
                save_obj = {
                    'output': hash_code.cpu(),
                    'target': target.cpu(),
                    'center': center.cpu(),
                    'weight': centerWeight_batch.cpu()
                }
            elif database_type == 'testbase' or database_type == 'database':
                save_obj = {
                    'output': hash_code.cpu(),
                    'target': target.cpu(),
                }

            self.pickleDump(save_obj, file_path)
        file_path.close()

    def adjust_learning_rate(self, optimizer, epoch, type='hashNet'):

        if type == 'hashNet':
            lr = option.lr * (0.7 ** (epoch // 10))
            optimizer.param_groups[0]['lr'] = option.multi_lr * lr
            optimizer.param_groups[1]['lr'] = lr
        elif type == 'centerNet' and self.option.center_update:

            lr = option.lr_center * (0.7 ** (epoch // 10))
            optimizer.param_groups[0]['lr'] = lr
        elif type == 'centerNet' and not self.option.center_update:
            lr = 0
        return lr

    def getHashCenters(self, labels, hash_centers, center_weight):
        '''
        get hash center for every image
        :param labels:
        :param hash_centers:
        :param center_weight:
        :return:
        '''
        if self.is_multi_label:
            hash_centers = self.Hash_center_multilables(labels, hash_centers, center_weight=center_weight)
        else:

            hash_label = (labels == 1).nonzero()[:, 1]
            hash_centers = hash_centers[hash_label]
        return hash_centers

    def Hash_center_multilables(self, labels,
                                Hash_center_pre,
                                center_weight):

        hash_centers = self.useGPU(torch.FloatTensor(torch.FloatStorage()))
        for (i, label) in enumerate(labels):
            one_labels = (label == 1).nonzero()  
            one_labels = one_labels.squeeze(1)
            Centers = Hash_center_pre[one_labels][:]
            center_weight_one = center_weight[i][one_labels]
            center_mean = torch.sum(Centers * center_weight_one.view(-1, 1), dim=0)
            hash_centers = torch.cat((hash_centers, center_mean.view(1, -1)), dim=0)
        return hash_centers

    def forward_hashCenter(self, centerModel, loader):

        for i, (input, target) in enumerate(loader):

            word_embedding = self.useGPU(input[2][0])
            hashCenter_pre = centerModel(word_embedding)        
            return hashCenter_pre

    def initCenterWeight(self, data_loader):

        if self.option.resume:
            if os.path.exists(self.option.resume_weight_path):
                return np.load(self.option.resume_weight_path)
            else:
                Logger.info(" lose weight path !")
                sys.exit()
        data_name = self.option.data_name
        file_path = '../data/' + data_name + '/' + data_name + '_initial_weight.npy'

        if os.path.exists(file_path):
            self.state['centerWeight_path'] = file_path
            return np.load(file_path)


        Logger.info("init center weight..")
        all_weight = None
        data_loader = tqdm(data_loader, desc="[init center weight]")
        for i, (input, target) in enumerate(data_loader):
            center_num = torch.sum(target > 0, dim=1)
            target[target <= 0] = 0.
            centerWeight = target.float() / center_num.view(-1, 1).float()

            if i == 0:
                all_weight = centerWeight.data.cpu().float()
            else:
                all_weight = torch.cat((all_weight, centerWeight.data.cpu().float()), 0)
        np.save(file_path, all_weight.cpu().numpy())
        return all_weight.cpu().numpy()

    def simplexPro(self, weightCenter_pre):
        for i in range(len(weightCenter_pre)):
            weight = weightCenter_pre[i]
            index = (weight > 0).nonzero().squeeze(1)
            X = torch.sort(weight[index])[0]
            Y = self.useGPU(torch.arange(1, len(index) + 1, 1).float())

            R = X + (1. / Y) * (1. - torch.cumsum(X, dim=0))
            Y[R <= 0.] = 0.
            rou = torch.max(Y).int()
            lambda_ = (1 / rou.float()) * (1 - torch.cumsum(X, dim=0)[rou - 1].float())
            temp = weight[index] + lambda_
            temp[temp < 0] = 0.
            weight[index] = temp
        return weightCenter_pre

    def pickleDump(self, content, filePath):
        pickle.dump(content, filePath)

    def save_checkpoint(self, option, state, model_dict, is_best, filename='checkpoint.pth.tar'):
        save_model_path = '../data/' + option.data_name + '/model'
        if option.data_name is not None:
            filename_ = filename
            filename = os.path.join(save_model_path, filename_)
            if not os.path.exists(save_model_path):
                os.makedirs(save_model_path)
        Logger.info('save model {filename}\n'.format(filename=filename))
        torch.save(model_dict, filename)  
        if is_best:

            if save_model_path is not None:

                if state['filename_previous_best'] is not None and os.path.exists(state['filename_previous_best']):
                    os.remove(state['filename_previous_best'])

                filename_best = os.path.join(save_model_path,
                                             'model_best_{score:.4f}.pth.tar'.format(score=model_dict['best_MAP']))
                shutil.copyfile(filename, filename_best)
                Logger.info('Current model is correpsopnding to best performance, save model {filename}\n'. \
                            format(filename=filename_best))
                state['filename_previous_best'] = filename_best

    def saveStatus(self, epoch, centerWeight_train, hashCenter_pre, MAP, result_all=None):

        np.save('../data/' + self.option.data_name + '/centers.npy', hashCenter_pre.detach().cpu().numpy())
        if MAP >= self.state['best_MAP']:
            if self.state['Database_hashpool_path'] is not None and os.path.exists(
                    self.state['Database_hashpool_path']):
                os.remove(self.state['Database_hashpool_path'])
            if self.state['Testbase_hashpool_path'] is not None and os.path.exists(
                    self.state['Testbase_hashpool_path']):
                os.remove(self.state['Testbase_hashpool_path'])
            if self.state['Trainbase_hashpool_path'] is not None and os.path.exists(
                    self.state['Trainbase_hashpool_path']):
                os.remove(self.state['Trainbase_hashpool_path'])
            self.state['Database_hashpool_path'] = utils.getDatabaseHashPoolPath(self.option, self.state)
            self.state['Testbase_hashpool_path'] = utils.getTestbaseHashPoolPath(self.option, self.state)
            self.state['Trainbase_hashpool_path'] = utils.getTrainbaseHashPoolPath(self.option, self.state)
            self.state['best_MAP'] = MAP
            self.state['best_epoch'] = epoch
            self.state['final_result'] = result_all
            np.save(utils.getWeightBestPath(self.option, self.state), centerWeight_train)

            pass
        elif epoch >= self.option.epochs - 1:
            np.save('../data/' + self.option.data_name + '/finalweight.npy', centerWeight_train)
            pass
        else:
            if os.path.exists(utils.getDatabaseHashPoolPath(self.option, self.state)):
                os.remove(utils.getDatabaseHashPoolPath(self.option, self.state))
            if os.path.exists(utils.getTestbaseHashPoolPath(self.option, self.state)):
                os.remove(utils.getTestbaseHashPoolPath(self.option, self.state))
            if os.path.exists(utils.getTrainbaseHashPoolPath(self.option, self.state)):
                os.remove(utils.getTrainbaseHashPoolPath(self.option, self.state))
        pass

    def resume(self, model_hash, model_center, optimizer_hash, optimizer_center):
        path_checkpoint = option.resume_path
        if option.resume and os.path.exists(path_checkpoint):
            checkpoint = torch.load(path_checkpoint)
            model_hash.load_state_dict(checkpoint['model_hash_dict'])
            model_center.load_state_dict(checkpoint['model_center_dict'])
            optimizer_hash.load_state_dict(checkpoint['optimizer_hash_dict'])
            optimizer_center.load_state_dict(checkpoint['optimizer_center_dict'])
            if option.use_gpu and torch.cuda.is_available():
                for state in optimizer_hash.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()
                for state in optimizer_center.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()

            return checkpoint['epoch']
        else:
            Logger.info("checkpoint file not exist!  ")
            sys.exit()
        pass


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    Logger.info("\t\tstart program\t\t")
    option= parser.parse_args()
    Logger.divider("print option")
    for k, v in vars(option).items():
        Logger.info('\t{}: {}'.format(k, v))
    state = {'start_time': start_time}
    state['use_gpu'] = option.use_gpu
    state['device'] = option.device
    state['pretrained_dir'] = "./vit_pretrain_models/ViT-B_16.npz"
    engine = Engine(option=option, state=state, config=None)
    engine.main()
    end_time = datetime.datetime.now()
    Logger.divider("END {}".format(Logger.getTimeStr(end_time)))
