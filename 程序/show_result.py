import numpy as np
import os
import cv2
import pandas as pd

ret = 100
def label_to_image():
    root = "./USS_Data/Unlabel"
    for t in os.listdir(root):
        r = os.path.join(root, t)
        for f in os.listdir(r):
            if f[-4:]=='.jpg':
                continue
            ff = os.path.join(r, f)
            f_image = ff.replace('.npy', '.jpg')
            result = np.load(ff)
            image = cv2.imread(f_image)
            mask = np.where(result==1)
            front = list(zip(mask[0], mask[1]))
            f_save = ff.replace('.npy', '_label.jpg')
            for s in front:
                image[s] = [255, 0, 0]
            cv2.imwrite(f_save, image)


if __name__ == '__main__':
    label_to_image()