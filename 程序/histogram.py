import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

root = "D://master/master (2)/master/Intra-phase/radius/Whole_pre"
left_Whole_pre = {'50h': 0.25, '4h': 0.125, '30min': 0.125, '2h': 0.125, '20h': 0.125, '200h': 1.25, '1h': 0.125
    , '150h': 0.875, '10h': 0.25, '100h': 0.625}
right_Whole_pre = {'50h': 10, '4h': 7, '30min': 3, '2h': 5, '20h': 9, '200h': 10, '1h': 3
    , '150h': 14, '10h': 10, '100h': 10}
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
ft = 20
left = left_Whole_pre
right = right_Whole_pre
for exc in os.listdir(root):
    if exc[:-4] not in left.keys():
        continue
    #     if exc[:-4]+'.jpg' in os.listdir('./'):
    #         continue
    r = os.path.join(root, exc)
    data = pd.read_csv(r)['avg_radius'].values
    #     data = pd.read_excel(r)['D'].values.astype(np.float64)
    b = int((right[exc[:-4]] - left[exc[:-4]]) * 8)
    hist, bins = np.histogram(data, b, range=(left[exc[:-4]], right[exc[:-4]]), density=False)
    s = sum(hist)
    x = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
    plt.bar(x, hist / s * 100, width=bins[1] - bins[0], color='red', edgecolor='black', label=exc[:-4])
    plt.ylabel('Frequency (%)', fontsize=ft, family='Times New Roman')
    plt.xlabel('Radius (' + r'$\mu$' + 'm)', fontsize=ft, family='Times New Roman')
    plt.xticks(family='Times New Roman', fontsize=ft)
    plt.yticks(family='Times New Roman', fontsize=ft)
    fontdict = {'family': 'Times New Roman',
                'size': ft}
    #     plt.legend(prop=fontdict)
    #     plt.savefig(exc[:-4]+'.jpg', bbox_inches='tight', dpi=1000)

    #     print(hist, bins)
    #     avg = 0
    #     for i in range(bins.shape[0]-1):
    #         mid = (bins[i]+bins[i+1])/2
    #         avg+=mid*hist[i]/s
    #     print(exc[:-4], avg*2, max(data),min(data))
    break
