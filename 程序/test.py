import argparse
import os
import os.path as osp

import fcn
import numpy as np
import skimage.io
import torch
from torch.autograd import Variable
import torchfcn
import tqdm
from Data import DSS, SplitData
# from torchvision.models.segmentation.segmentation import fcn_resnet50
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from Unet import UNet
from CCNet import Seg_Model
import torch.nn as nn
import torch.nn.functional as F


# from show_result import label_to_image
# from find_contours import compute_con
# from radius import compute_radius
# from energy import compute_energy


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
        # prediction = F.softmax(interp(prediction), dim=1).cpu().detach().numpy().transpose(0,2,3,1)
        prediction = interp(prediction).cpu().detach().numpy().transpose(0, 2, 3, 1)
    return prediction


def test():
    parser = argparse.ArgumentParser()
    # parser.add_argument('model_file', help='Model path', default = './logs/CCNet_chi/model_best.pth.tar')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    # model_file = args.model_file
    model_file = './logs/CCNet_chi/model_best.pth.tar'

    files = SplitData("./USS_Data", test=True)
    print(files)
    Test = DSS(0, files)
    test_loader = torch.utils.data.DataLoader(Test, batch_size=1, shuffle=False)
    print(len(files))
    # model = torchfcn.models.FCN8s(n_class=2)
    # model = fcn_resnet50(pretrained=False, num_classes=2)
    # model = UNet(n_channels=3, n_classes=2, bilinear=False)
    model = Seg_Model(num_classes=2, recurrence=2)
    if torch.cuda.is_available():
        model = model.cuda()
    print('==> Loading %s model file: %s' %
          (model.__class__.__name__, model_file))
    model_data = torch.load(model_file)['model_state_dict']
    for k in list(model_data.keys()):
        if "aux_classifier" in k:
            del model_data[k]
    try:
        model.load_state_dict(model_data)
    except Exception:
        model.load_state_dict(model_data['model_state_dict'])
    # writer = SummaryWriter('runs/experiment_1')
    model.eval()

    print('==> Evaluating ')
    sum = 0
    for batch_idx, (data, files) in enumerate(test_loader):
        if torch.cuda.is_available():
            data = data.cuda()
        data = Variable(data)
        # x = data
        # img_grid = vutils.make_grid(x, normalize=True, scale_each= True, nrow=2)
        # for name, layer in model._modules.items():
        #     # # 为fc层预处理x
        #     # x = x.view(x.size(0), -1) if 'fc' in name else x
        #     # # print(x.size())

        #     x = layer(x)
        #     print(f'{name}')

        #     # 查看卷积层的特征图
        #     if 'layer' in name or 'conv' in name:
        #         x1 = x.transpose(0, 1)  # C，B, H, W ---> B，C, H, W
        #         img_grid = vutils.make_grid(x1, normalize=True, scale_each=True, nrow=4)
        #         writer.add_image(f'{name}_feature_maps', img_grid, global_step=0)

        # score = model(data)
        # score_out = score
        # score_out = score['out']
        with torch.no_grad():
            score_out = predict_whole(model, data)

        imgs = data.data.cpu()
        # lbl_pred = score_out.data.max(1)[1].cpu().numpy()[:, :, :]
        lbl_pred = np.asarray(np.argmax(score_out, axis=3), dtype=np.uint8)
        # for lp, f in zip(lbl_pred,  files):
        #     # ff = f[:-4].replace('Unlabel', 'Result')
        #     # ff = f.replace('/Data', '/Predict_of_label')
        #     # print(ff[:-4]+'.npy')
        #     np.save(f[:-4]+'.npy', lp)
        #     sum = sum+1
        print(files, score_out.shape)
        np.save(str(batch_idx) + '.npy', score_out)


if __name__ == '__main__':
    test()