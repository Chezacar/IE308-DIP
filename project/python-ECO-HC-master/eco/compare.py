#我们参照评价指标的定义自编的参数获取代码
import numpy as np
import os
import pandas as pd

target = pd.read_csv('./sequences/Crossing/groundtruth_rect.txt', sep='\t|,| ',
            header=None, names=['xmin', 'ymin', 'width', 'height'],
            engine='python')
result = np.loadtxt('./results/Crossing.txt')

result = np.array(result)
target = np.array(target)

def overlap(result,target): #用于获得overlap评价指标
    sz = result.shape[0]
    overlap = np.zeros((sz))
    for i in range(sz):
        save = np.zeros((4))
        save = save.astype(np.float64)
        save[0] = max(result[i,0],target[i,0])
        save[1] = max(result[i,1],target[i,1])
        save[2] = min(result[i,2],target[i,0]+target[i,2])
        save[3] = min(result[i,3],target[i,1]+target[i,3])
        overlap_square = (save[2] - save[0]) * (save[3] - save[1])
        result_square = (result[i,2] - result[i,0]) * (result[i,3] - result[i,1])
        target_square = target[i,2] * target[i,3]
        overlap[i] = overlap_square/(result_square + target_square - overlap_square)
    overlap_ave = sum(overlap) / sz
    return overlap,overlap_ave

def overlap_threshold(overlap,threshold): #用于获取overlap_threshold评价指标
    sz = overlap.shape[0]
    num = 0
    for i in range(sz):
        if overlap[i] > threshold:
            num += 1 
    overlap_th = num / sz
    return overlap_th

overlap,overlap_ave = overlap(result,target)
overlap_th = overlap_threshold(overlap,0.8)

print('overlap is {}'.format(overlap_ave))
print('overlap_threshold is {}'.format(overlap_th))