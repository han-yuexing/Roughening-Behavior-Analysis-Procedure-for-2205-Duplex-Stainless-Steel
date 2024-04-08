import datetime
from distutils.version import LooseVersion
import math
import os
import os.path as osp
import shutil

import fcn
import numpy as np
import pytz
import skimage.io
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import tqdm

import torchfcn
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from CCNet_loss import CriterionDSN, CriterionOhemDSN
from scipy import ndimage
import torch.nn as nn


def predict_whole(net, image, train=False):
    N_, C_, H_, W_ = image.shape
    # image = torch.from_numpy(image)
    interp = nn.Upsample(size=(H_, W_), mode='bilinear', align_corners=True)
    prediction = net(image.cuda())  #
    if isinstance(prediction, list):
        prediction = prediction[0]
    if train == True:
        prediction = interp(prediction).cpu().detach().numpy().transpose(0, 2, 3, 1)
    else:
        prediction = interp(prediction).cpu().numpy().transpose(0, 2, 3, 1)
    return prediction


def cross_entropy2d(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    if LooseVersion(torch.__version__) < LooseVersion('0.3'):
        # ==0.2.X
        log_p = F.log_softmax(input)
    else:
        # >=0.3
        log_p = F.log_softmax(input, dim=1)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]

    loss = F.nll_loss(log_p, target, weight=weight, reduction='mean')
    # if size_average:
    #     loss /= mask.data.sum()
    return loss


def Cross_py(input, target, weight):
    criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=2, reduction='mean')
    loss = criterion(input, target)
    return loss


def Dice_loss(input, targets):
    smooth = 1
    N = targets.size()[0]
    input_positive = input[:, 1, :, :]
    input_flat = input_positive.view(N, -1)
    targets_flat = targets.view(N, -1)
    intersection = input_flat * targets_flat
    N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
    loss = 1 - N_dice_eff.sum() / N
    return loss


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=False, ignore_index=2, weight=(0.5, 0.5)):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if self.alpha is not None:
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.weight = weight

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)
        # index处理
        valid_mask = torch.where(target != self.ignore_index)
        input = input[valid_mask[0], :]
        target = target[valid_mask[0], :]

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp2())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            bal_mask = torch.where(target == 0)
            bal_loss = loss[bal_mask[0]]
            loss_sum = (self.weight[0] - self.weight[1]) * bal_loss.sum() + self.weight[1] * loss.sum()
            return loss_sum / loss.shape[0]
            # return loss.mean()
        else:
            return loss.sum()


class Trainer(object):

    def __init__(self, cuda, model, optimizer,
                 train_loader, val_loader, out, max_iter, scheduler=None,
                 size_average=True, interval_validate=None):
        self.cuda = cuda

        self.model = model
        self.optim = optimizer

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.timestamp_start = \
            datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
        self.size_average = size_average

        if interval_validate is None:
            self.interval_validate = len(self.train_loader)
        else:
            self.interval_validate = interval_validate
        if scheduler is not None:
            self.scheduler = scheduler
        else:
            self.scheduler = None

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.log_headers = [
            'epoch',
            'iteration',
            'train/loss',
            'train/acc',
            'train/acc_cls',
            'train/mean_iu',
            'train/fwavacc',
            'valid/loss',
            'valid/acc',
            'valid/acc_cls',
            'valid/mean_iu',
            'valid/fwavacc',
            'elapsed_time',
        ]
        if not osp.exists(osp.join(self.out, 'log.csv')):
            with open(osp.join(self.out, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

        self.epoch = 0
        self.iteration = 0
        self.max_iter = max_iter
        self.best_mean_iu = 0
        self.weight = torch.tensor((0.7, 0.3))
        if self.cuda:
            self.weight = self.weight.cuda()
        self.CCNet_loss = CriterionOhemDSN(thresh=0.7, min_kept=100000, weight=self.weight)

    def validate(self):
        training = self.model.training
        self.model.eval()

        n_class = len(self.val_loader.dataset.class_names)

        val_loss = 0
        visualizations = []
        label_trues, label_preds = [], []
        for batch_idx, (data, target) in tqdm.tqdm(
                enumerate(self.val_loader), total=len(self.val_loader),
                desc='Valid iteration=%d' % self.iteration, ncols=80,
                leave=False):
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            with torch.no_grad():
                score = self.model(data)
                # score_out = score[1]
                # for CCNet
                score_out = predict_whole(self.model, data)
            # seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
            # print(self.model)
            # for fcn
            # score_out = score['out']
            # score_aux = score['aux']
            # print(score_out.shape)
            # loss = Cross_py(score_out, target, self.weight)
            # loss_data = loss.data.item()
            # loss = FocalLoss(gamma=1, weight=self.weight)(score_out,target)
            # loss_data = loss
            loss = self.CCNet_loss(score, target)
            loss_data = loss
            # if np.isnan(loss_data.cpu()):
            #     raise ValueError('loss is nan while validating')
            # loss_data = Dice_loss(score, target)
            val_loss += loss_data

            imgs = data.data.cpu()
            lbl_pred = np.asarray(np.argmax(score_out, axis=3), dtype=np.uint8)
            # lbl_pred = score_out.data.max(1)[1].cpu().numpy()[:, :, :]
            # lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu()
            for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
                img, lt = self.val_loader.dataset.untransform(img, lt)
                lt = lt.reshape(lp.shape)
                label_trues.append(lt)
                label_preds.append(lp)
                if len(visualizations) < 9:
                    viz = fcn.utils.visualize_segmentation(
                        lbl_pred=lp, lbl_true=lt, img=img, n_class=n_class)
                    visualizations.append(viz)
        metrics = torchfcn.utils.label_accuracy_score(
            label_trues, label_preds, n_class)

        out = osp.join(self.out, 'visualization_viz')
        if not osp.exists(out):
            os.makedirs(out)
        out_file = osp.join(out, 'iter%012d.jpg' % self.iteration)
        skimage.io.imsave(out_file, fcn.utils.get_tile_image(visualizations))
        # tb_img = np.array(Image.open(out_file)).reshape((3, 5760, 11520))[:, :960*2, :1280*3]
        # self.writer.add_image('nine_images', tb_img, self.iteration)

        # for focal_loss
        # val_loss /= len(self.val_loader)

        with open(osp.join(self.out, 'log.csv'), 'a') as f:
            elapsed_time = (
                    datetime.datetime.now(pytz.timezone('Asia/Shanghai')) -
                    self.timestamp_start).total_seconds()
            log = [self.epoch, self.iteration] + [''] * 5 + \
                  [val_loss] + list(metrics) + [elapsed_time]
            log = map(str, log)
            f.write(','.join(log) + '\n')
        self.writer.add_scalar('mean_iu_of_validate',
                               metrics[2],
                               self.iteration / self.interval_validate)
        mean_iu = metrics[2]
        is_best = mean_iu > self.best_mean_iu
        if is_best:
            self.best_mean_iu = mean_iu
        torch.save({
            'epoch': self.epoch,
            'iteration': self.iteration,
            'arch': self.model.__class__.__name__,
            'optim_state_dict': self.optim.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'best_mean_iu': self.best_mean_iu,
        }, osp.join(self.out, 'checkpoint.pth.tar'))
        if is_best:
            shutil.copy(osp.join(self.out, 'checkpoint.pth.tar'),
                        osp.join(self.out, 'model_best.pth.tar'))

        if training:
            self.model.train()

    def train_epoch(self):
        self.writer = SummaryWriter('runs/exp2')
        self.model.train()

        n_class = len(self.train_loader.dataset.class_names)

        for batch_idx, (data, target) in tqdm.tqdm(
                enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):
            iteration = batch_idx + self.epoch * len(self.train_loader)
            # print(np.unique(target.numpy()))
            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue  # for resuming
            self.iteration = iteration

            if self.iteration % self.interval_validate == 0:
                self.validate()

            assert self.model.training

            if self.cuda:
                data, target = data.cuda(), target.cuda()

            data, target = Variable(data), Variable(target)
            # print(data.shape)
            self.optim.zero_grad()
            score = self.model(data)
            score_out = predict_whole(self.model, data, train=True)
            # score_out = score['out']
            # score_aux = score['aux']

            # loss = cross_entropy2d(score, target, weight=self.weight,
            #                        size_average=self.size_average)
            # loss = Cross_py(score_out, target, self.weight)
            # loss /= len(data)
            # loss = Cross_py(score, target, self.weight)
            # loss_data = loss.data.item()
            # loss = FocalLoss(gamma=1, weight=self.weight)(score_out,target)
            # loss /= len(data)
            # loss_data = loss
            loss = self.CCNet_loss(score, target)
            loss_data = loss
            self.writer.add_scalar('training loss',
                                   loss_data,
                                   self.iteration)
            # loss = Dice_loss(score, target)
            self.writer.add_scalar('training loss',
                                   loss_data,
                                   self.iteration)
            # if np.isnan(loss_data):
            #     raise ValueError('loss is nan while training')
            loss.backward()
            self.optim.step()
            if self.scheduler is not None:
                self.scheduler.step()

            metrics = []
            # lbl_pred = score_out.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_pred = np.asarray(np.argmax(score_out, axis=3), dtype=np.uint8)
            # lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
            # if self.iteration  == 0:
            #     print("pred", set(lbl_pred.flatten()))
            lbl_true = target.data.cpu().numpy()
            acc, acc_cls, mean_iu, fwavacc = \
                torchfcn.utils.label_accuracy_score(
                    lbl_true, lbl_pred, n_class=n_class)
            metrics.append((acc, acc_cls, mean_iu, fwavacc))
            metrics = np.mean(metrics, axis=0)
            self.writer.add_scalar('mean_iu_of_train',
                                   metrics[2],
                                   self.iteration)
            self.writer.add_scalar('learning_rate',
                                   self.optim.state_dict()['param_groups'][0]['lr'],
                                   self.iteration)
            with open(osp.join(self.out, 'log.csv'), 'a') as f:
                elapsed_time = (
                        datetime.datetime.now(pytz.timezone('Asia/Shanghai')) -
                        self.timestamp_start).total_seconds()
                log = [self.epoch, self.iteration] + [loss_data] + \
                      metrics.tolist() + [''] * 5 + [elapsed_time]
                log = map(str, log)
                f.write(','.join(log) + '\n')
            if self.iteration >= self.max_iter:
                break

    def train(self):

        max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))
        for epoch in tqdm.trange(self.epoch, max_epoch,
                                 desc='Train', ncols=80):
            self.epoch = epoch
            self.train_epoch()
            if self.iteration >= self.max_iter:
                break
