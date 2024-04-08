import argparse
import datetime
import os
import os.path as osp

import torch
import yaml

import torchfcn

from Data import SplitData, DSS
from torch.utils import data
from train import Trainer
from FCN8S import FCN8s
from torchvision.models.segmentation import fcn_resnet50
from Unet import UNet
from STLNet import STLNet
from SegNet import SegNet
from CCNet import Seg_Model



def get_parameters(model, bias=False):
    import torch.nn as nn
    modules_skipped = (
        nn.ReLU,
        nn.MaxPool2d,
        nn.Dropout2d,
        nn.Sequential,
        torchfcn.models.FCN32s,
        torchfcn.models.FCN16s,
        torchfcn.models.FCN8s,
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if bias:
                yield m.bias
            else:
                yield m.weight
        elif isinstance(m, nn.ConvTranspose2d):
            # weight is frozen because it is just a bilinear upsampling
            if bias:
                assert m.bias is None
        elif isinstance(m, modules_skipped):
            continue
        else:
            raise ValueError('Unexpected module: %s' % str(m))


here = osp.dirname(osp.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-g', '--gpu', type=int, default=0, help='gpu id')
    # configurations (same configuration as original work)
    # https://github.com/shelhamer/fcn.berkeleyvision.org
    parser.add_argument(
        '--max-iteration', type=int, default=10000000, help='max iteration'
    )
    #  parser.add_argument(
    #     '--max-epoch', type=int, default=1000, help='max iteration'
    # )
    parser.add_argument(
        '--lr', type=float, default=1.0e-14, help='learning rate',
    )
    parser.add_argument(
        '--classes', type=int, default=2, help='classes',
    )
    parser.add_argument(
        '--weight-decay', type=float, default=0.0005, help='weight decay',
    )
    parser.add_argument(
        '--momentum', type=float, default=0.9, help='momentum',
    )

    args = parser.parse_args()

    args.model = 'CCNet'

    now = datetime.datetime.now()
    args.out = osp.join(here, 'logs', now.strftime('%Y%m%d_%H%M%S.%f'))

    os.makedirs(args.out)
    with open(osp.join(args.out, 'config.yaml'), 'w') as f:
        yaml.safe_dump(args.__dict__, f, default_flow_style=False)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cuda = torch.cuda.is_available()

    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)
    # print("yes")
    # 1. dataset

    X_train_files, Y_train_files, X_val_files, Y_val_files = SplitData("./USS_Data")
    Train = DSS(1, X_train_files, Y_train_files,augmentation=False)
    Test = DSS(1, X_val_files, Y_val_files, augmentation=False)
    train_loader = data.DataLoader(Train, batch_size=1, shuffle=False)
    val_loader = data.DataLoader(Test, batch_size=1, shuffle=False)
    # print("yes dataset")

    # 2. model
    start_epoch = 0
    start_iteration = 0
    # model = torchfcn.models.FCN8s(n_class=2)
    # pretrained = 1
    # num_classes = 2
    # model = fcn_resnet50(pretrained=True, progress=True)
    # if num_classes != 21:
    #     model.classifier[4] = torch.nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
    #     model.aux_classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    # model = UNet(n_channels=3, n_classes=args.classes, bilinear=False)
    # model = SegNet(classes=2)
    model = Seg_Model(num_classes=2, pretrained_model='./resnet101-imagenet.pth', recurrence=2)
    if cuda:
        model = model.cuda()
    # print("yes model")
    # 3. optimizer

    # optim = torch.optim.SGD(
    #     [
    #         {'params': get_parameters(model, bias=False)},
    #         {'params': get_parameters(model, bias=True),
    #          'lr': args.lr * 2, 'weight_decay': 0},
    #     ],
    #     lr=args.lr,
    #     momentum=args.momentum,
    #     weight_decay=args.weight_decay)
    # params = [p for p in model.aux_classifier.parameters() if p.requires_grad]
    '''for others'''
    # optim = torch.optim.Adam(
    #     params = model.parameters(),
    #     lr=args.lr,
    #     betas=(0.9, 0.99),
    #     weight_decay=args.weight_decay)
    '''for CCNet'''
    optim = torch.optim.SGD([{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': args.lr}], 
                lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # optim = torch.optim.Adam(
    #     [
    #         {'params': get_parameters(model, bias=False)},
    #         {'params': get_parameters(model, bias=True)},
    #     ],
    #     lr=args.lr,
    #     betas=(0.9, 0.99),
    #     weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optim,step_size=500,gamma = 0.9)

    # print("yes optimizer")
    trainer = Trainer(
        cuda=cuda,
        model=model,
        optimizer=optim,
        train_loader=train_loader,
        val_loader=val_loader,
        out=args.out,
        max_iter=args.max_iteration, 
        scheduler = scheduler,
        interval_validate=1500,
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()


if __name__ == '__main__':
    main()