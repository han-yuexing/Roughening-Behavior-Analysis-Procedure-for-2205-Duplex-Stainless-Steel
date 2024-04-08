import numpy as np
import os
import cv2
import pandas as pd
ret = 100

def compute_con():
    root = "./USS_Data/Unlabel"
    for t in os.listdir(root):
        r = os.path.join(root, t)
        radius_sum = 0
        nums = 0
        for f in os.listdir(r):
            statis = []
            if f.endswith('.npy'):
                ff = os.path.join(r, f)
                result = np.load(ff).astype(np.uint8)*255
#                 gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
#                 ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
                contours, hierarchy = cv2.findContours(result, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                for c_id, cnt in enumerate(contours):
                    area = cv2.contourArea(cnt)
                    if area > 0.5*result.shape[0]*result.shape[1] or area<ret:
                        continue
                    x, y, w, h = cv2.boundingRect(cnt)
                    perimeter = cv2.arcLength(cnt, True)
                    center, radius = cv2.minEnclosingCircle(cnt)
#                     cv2.circle()
                    statis.append([area,  w, h, perimeter, radius, x, y,])
                    radius_sum+=radius
                    nums+=1
                pdf = pd.DataFrame(statis, columns=['area', 'width', 'height', 'perimeter',  'radius', 'x_lt', 'y_lt'])
                pdf.to_csv(os.path.join(os.path.join("./USS_Data/Unlabel", t), f[:-4]+'.csv'))


if __name__ == '__main__':
    compute_con()