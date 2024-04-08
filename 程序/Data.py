import os
from PIL import Image
import numpy as np
import random
from torch.utils import data
from torchvision import transforms
import torch
import random
import pandas as pd
from torch.nn import functional as F
from torch.autograd import Variable


def SplitData(root, ratio_of_test=0.1, test=False, predict=False):
    # root:数据集所在文件夹
    # 该函数将得到训练集、测试集的划分，分层抽样
    indexes = [18, 30, 48, 70, 95, 107, 138, 163, 171, 211]
    CHI1 = np.arange(0, 30)
    CHI2 = np.arange(95, 107)
    CHI = np.hstack((CHI1, CHI2)).flatten()
    if test == True:
        DataPath = 'USS/USS_Data/Unlabel'
    else:
        DataPath = '/Project_of_master/instance_label'

    Data = []
    if test == True:
        for t in os.listdir(DataPath):
            r = os.path.join(DataPath, t)
            for img in os.listdir(r):
                if img[-4:] == '.npy':
                    continue
                Data.append(os.path.join(r, img))
        return Data

    temp = [i for i in range(1, 212)]
    # random.shuffle(temp)
    for k in temp:
        if k < 100:
            name = '0' + str(k) + '_json/img.png'
        else:
            name = str(k) + '_json/img.png'
        Data.append(os.path.join(DataPath, name))
    Data = np.array(Data)
    if predict == True:
        return Data

    Label_mask = []
    Label_class = []
    for k in temp:
        if k < 100:
            name_mask = '0' + str(k) + '_json/label.png'
            name_class = '0' + str(k) + '_json/label_names.txt'
        else:
            name_mask = str(k) + '_json/label.png'
            name_class = str(k) + '_json/label_names.txt'
        Label_mask.append(os.path.join(DataPath, name_mask))
        Label_class.append(os.path.join(DataPath, name_class))
    Label_mask = np.array(Label_mask)
    Label_class = np.array(Label_class)

    valid_inds = []
    for i, _ in enumerate(indexes):
        end_ind = indexes[i]
        '''sigma phase'''
        # if end_ind == 18 or end_ind==30 or end_ind==107:
        #     continue
        '''chi phase'''
        if end_ind != 18 and end_ind != 30 and end_ind != 107:
            continue
        if i == 0:
            begin_ind = 0
        else:
            begin_ind = indexes[i - 1]
        num_layer = indexes[i] - begin_ind
        num_layer_test = int(round(ratio_of_test * num_layer))
        valid_ind = random.sample(range(begin_ind, end_ind), num_layer_test)
        valid_inds = valid_inds + valid_ind
    '''sigma phase'''
    train_inds = np.delete(np.array(range(0, 211)), valid_inds)
    # train_inds = np.delete(train_inds, CHI)
    '''chi phase'''

    # train_inds = np.delete(np.array(CHI), test_inds)
    # train_inds = train_inds[CHI]
    X_train_files = Data[train_inds]
    Mask_train_files = Label_mask[train_inds]
    class_train_files = Label_class[train_inds]
    X_valid_files = Data[valid_inds]
    Mask_valid_files = Label_mask[valid_inds]
    class_valid_files = Label_class[valid_inds]
    print(train_inds.shape, valid_inds)
    return X_train_files, Mask_train_files, class_train_files, X_valid_files, Mask_valid_files, class_valid_files


class DSS(data.Dataset):
    def __init__(self, max_num, X_files, Mask_files=None, class_files=None, transform=True, test=0):
        self.files = []
        self._transform = transform
        self.test = test
        self.max_num = max_num
        self.num_instance = torch.zeros(X_files.shape[0])
        if self.test == 0:
            for x_file, mask_file, class_file in zip(X_files, Mask_files, class_files):
                self.files.append({
                    'image': x_file,
                    'mask': mask_file,
                    'classes': class_file
                })
        else:
            self.files = X_files
        self.mean = [0.485]
        self.std = [0.229]
        self.box = (0, 0, 1280, 896)
        self.epoch = np.zeros(len(self.files))

        self.class_names = np.array([
            'background',
            'target',
            'none'
        ])

    def __getitem__(self, index):
        if self.test == 0:
            files = self.files[index]
            self.epoch[index] += 1
            # image = Image.open(files['image']).crop(self.box)  #把底下的图例刪除
            image = np.resize(np.array(Image.open(files['image'])), (480, 640))
            label_mask = np.resize(np.array(Image.open(files['mask'])), (480, 640))
            label_mask = torch.from_numpy(label_mask)
            label_classes = torch.ones([pd.read_table(files['classes'], header=None).shape[0], 1])  # 只适用于两个类别
            label_classes[0] = 0

            self.num_instance[index] = label_classes.size()[0]

            # label_classes=F.pad(label_classes, (0,0,0, self.max_num-label_classes.size()[0]), mode='constant',value=self.class_names.shape[0]-1)
            if self._transform:
                return self.transform(image, [label_mask, label_classes], index)
            else:
                return image, {'mask': label_mask, 'labels': label_classes}
        else:
            files = self.files[index]
            # image = np.array(Image.open(files).crop(self.box))
            image = np.array(Image.open(files))
            if self._transform:
                return self.transform(image), files
            else:
                return image, files

    def __len__(self):
        return len(self.files)

    def transform(self, img, lbl=None, index=0):
        # img:PIL.Image, lbl:array
        # lbl = torch.from_numpy(lbl)

        img = img.astype(np.float64)
        # img -= self.mean
        # img /= self.std
        img = torch.from_numpy(img).float()
        img = torch.unsqueeze(img, 0)
        if self.test != 0:
            return img
        lbl[0] = lbl[0].long()
        lbl[1] = lbl[1].long()
        if torch.cuda.is_available():
            img = img.cuda()
            lbl[0] = lbl[0].cuda()
            lbl[1] = lbl[1].cuda()
        return Variable(img), [{'mask': Variable(lbl[0]), 'label': Variable(lbl[1])}]

    def untransform(self, img, lbl=None):
        img = torch.squeeze(img, dim=0)
        img = img.numpy()
        img *= self.std
        img += self.mean
        img = img.astype(np.uint8)
        img = np.stack((img,) * 3, axis=-1)
        # print(img.shape)
        # img = img.transpose((1, 2, 0))
        # img = np.unsqueeze(img)
        # print(img.shape)

        if self.test == 0:
            return img, lbl
        else:
            return img


if __name__ == "__main__":
    X_train_files, Mask_train_files, class_train_files, X_valid_files, Mask_valid_files, class_valid_files = SplitData(
        "instance_label")
    # 最多是251，加上background
    Train = DSS(260, X_train_files, Mask_train_files, class_train_files)
    Valid = DSS(260, X_valid_files, Mask_valid_files, class_valid_files)
    train_loader = data.DataLoader(Train, batch_size=16, shuffle=True)
    valid_loader = data.DataLoader(Valid, batch_size=16, shuffle=True)