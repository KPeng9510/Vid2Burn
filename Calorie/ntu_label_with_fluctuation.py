import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import torch
#from keras.utils import Sequence, to_categorical
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import glob
from PIL import Image
import PIL
import pickle
from opts import *
from scipy.ndimage import rotate
from scipy import stats
from random import sample, randint, shuffle
import glob
import cv2
#from scipy.misc import imresize
import random
cv2.setNumThreads(0)
import time
import sys
import csv
calorie_dict = [
    175,
    175,
    162,
    193,
    189,
    395,
    382,
    357,
    352,
    294,
    254,
    237,
    298,
    368,
    366,
    347,
    351,
    240,
    249,
    325,
    320,
    332,
    274,
    356,
    192,
    405,
    434,
    178,
    157,
    192,
    221,
    222,
    207,
    248,
    356,
    175,
    234,
    264,
    214,
    288,
    260,
    369,
    392,
    196,
    192,
    187,
    227,
    359,
    336,
]
def get_median(data):

   data = sorted(data)

   size = len(data)

   if size % 2 == 0: # 判断列表长度为偶数

    median = (data[size//2]+data[size//2-1])/2

    data[0] = median

   if size % 2 == 1: # 判断列表长度为奇数

    median = data[(size-1)//2]

    data[0] = median

   return data[0]
path = 'ntu_staffs/ntu_calorie_all_skvideo.pkl'
initial_fluctuation = 50
f=open(path,'rb')
dataset=pickle.load(f)
print('Total dataset number is',len(dataset))
final_label= []
final_fps = []
category_wise_label = {}
motion_dict = {}

for i in range(49):
    category_wise_label[i]=[]
    motion_dict[i]=[]
for skeleton_path in dataset:
    label_index = int(skeleton_path.split('/')[-1].split('A')[1].split('.skeleton')[0])
    name = int(skeleton_path.split('/')[-1].split('.')[0][1:4])
    if name in [1,2,3,4,5,6,7,8]:
        print(skeleton_path)
        data = np.load(skeleton_path, allow_pickle=True).item()
        skeleton = data['skel_body0']
        final_fps.append(skeleton.shape[0])
        fps = 30
        time_length = 3600 / (skeleton.shape[0]*(1/fps))
        #print(skeleton)
        skeleton_minus = np.zeros(skeleton.shape)
        skeleton_minus[1:,:,:]=skeleton[:-1,:,:]
        skeleton_minus[0,:,:]=skeleton[0,:,:]
        skeleton = skeleton-skeleton_minus
        #print(np.max(skeleton**2))
        #sys.exit()
        head = np.expand_dims(skeleton[:,3,:],axis=1)
        forearm = np.concatenate([skeleton[:,6:8,:].sum(-2)/2, skeleton[:,10:12,:].sum(-2)/2], axis=1)
        upperarm = np.concatenate([skeleton[:, 5:7, :].sum(-2)/2, skeleton[:, 9:11, :].sum(-2)/2], axis=1)
        hand = np.concatenate([np.expand_dims(skeleton[:,12,:], axis=-2), np.expand_dims(skeleton[:,23,:], axis=-2)], axis=1)
        body = np.expand_dims(skeleton[:,1,:],1)
        thigh = np.concatenate([skeleton[:,17:19,:].sum(-2)/2, skeleton[:,13:15, :].sum(-2)/2], axis=1)
        shin = np.concatenate([skeleton[:,18:20,:].sum(-2)/2, skeleton[:,14:16, :].sum(-2)/2], axis=1)
        foot = np.concatenate([np.expand_dims(skeleton[:,20,:], axis=-2), np.expand_dims(skeleton[:,16,:], axis=-2)], axis=1)

        e_head = 0.5*68*fps*fps*(head**2).sum(-1)*0.1
        e_forearm = 0.5*68*fps*fps*(forearm**2).sum(-1)*0.03
        e_upperarm = 0.5*68*fps*fps*(upperarm**2).sum(-1)*0.04
        e_hand = 0.5*68*fps*fps*(hand**2).sum(-1)*0.025
        e_body = 0.5*68*fps*fps*(body**2).sum(-1)*0.3
        e_thigh = 0.5*68*fps*fps*(thigh**2).sum(-1)*0.1
        e_shin = 0.5*68*fps*fps*(shin**2).sum(-1)*0.07
        e_foot = 0.5*68*fps*fps*(foot**2).sum(-1)*0.035
        #print(e_head)
        label = e_head.sum()+e_forearm.sum()+e_upperarm.sum()+e_hand.sum()+e_body.sum()+e_thigh.sum()+e_shin.sum()+e_foot.sum()
        #print(np.clip(time_length*label/(4.19*1000), 63, 1000))
        #final_label.append(np.clip(time_length*label/(4.19*400)-200, 63, 1000))
        #print(label_index)
        category_wise_label[label_index-1].append(np.clip(time_length*label/(4.19*1000), 63, 400))

#for i in range(49):
#   #print(i+1, 'calorie', sum(category_wise_label[i])/len(category_wise_label[i]))
#print(sum(final_fps)/len(final_fps))
#print(max(final_label))
#print(min(final_label))
final_dict = {}
for skeleton_path in dataset:
    label_index = int(skeleton_path.split('/')[-1].split('A')[1].split('.skeleton')[0])
    name = int(skeleton_path.split('/')[-1].split('.')[0][1:4])
    motion_min = min(category_wise_label[label_index-1])
    motion_max = max(category_wise_label[label_index-1])
    fluctuate_range = (calorie_dict[label_index-1] / 1000) * initial_fluctuation

    if name in [1,2,3,4,5,6,7,8]:
        #print(skeleton_path)
        data = np.load(skeleton_path, allow_pickle=True).item()
        skeleton = data['skel_body0']
        final_fps.append(skeleton.shape[0])
        fps = 30
        time_length =3600 / (skeleton.shape[0]*(1/fps))
        #print(skeleton)
        skeleton_minus = np.zeros(skeleton.shape)
        skeleton_minus[1:,:,:]=skeleton[:-1,:,:]
        skeleton_minus[0,:,:]=skeleton[0,:,:]
        skeleton = skeleton-skeleton_minus
        #print(np.max(skeleton**2))
        #sys.exit()
        head = np.expand_dims(skeleton[:,3,:],axis=1)
        forearm = np.concatenate([skeleton[:,6:8,:].sum(-2)/2, skeleton[:,10:12,:].sum(-2)/2], axis=1)
        upperarm = np.concatenate([skeleton[:, 5:7, :].sum(-2)/2, skeleton[:, 9:11, :].sum(-2)/2], axis=1)
        hand = np.concatenate([np.expand_dims(skeleton[:,12,:], axis=-2), np.expand_dims(skeleton[:,23,:], axis=-2)], axis=1)
        body = np.expand_dims(skeleton[:,1,:],1)
        thigh = np.concatenate([skeleton[:,17:19,:].sum(-2)/2, skeleton[:,13:15, :].sum(-2)/2], axis=1)
        shin = np.concatenate([skeleton[:,18:20,:].sum(-2)/2, skeleton[:,14:16, :].sum(-2)/2], axis=1)
        foot = np.concatenate([np.expand_dims(skeleton[:,20,:], axis=-2), np.expand_dims(skeleton[:,16,:], axis=-2)], axis=1)

        e_head = 0.5*68*fps*fps*(head**2).sum(-1)*0.1
        e_forearm = 0.5*68*fps*fps*(forearm**2).sum(-1)*0.03
        e_upperarm = 0.5*68*fps*fps*(upperarm**2).sum(-1)*0.04
        e_hand = 0.5*68*fps*fps*(hand**2).sum(-1)*0.025
        e_body = 0.5*68*fps*fps*(body**2).sum(-1)*0.3
        e_thigh = 0.5*68*fps*fps*(thigh**2).sum(-1)*0.1
        e_shin = 0.5*68*fps*fps*(shin**2).sum(-1)*0.07
        e_foot = 0.5*68*fps*fps*(foot**2).sum(-1)*0.035
        #print(e_head)
        label = e_head.sum()+e_forearm.sum()+e_upperarm.sum()+e_hand.sum()+e_body.sum()+e_thigh.sum()+e_shin.sum()+e_foot.sum()
        #print(np.clip(time_length*label/(4.19*1000), 63, 1000))
        #final_label.append(np.clip(time_length*label/(4.19*400)-200, 63, 1000))
        #print(label_index)
        calorie = np.clip(time_length*label/(4.19*1000), 63, 400)
        if (motion_max-motion_min) !=0:
            final_dict[skeleton_path] = fluctuate_range*(calorie - (motion_min+motion_max)/2)/(motion_max-motion_min)+calorie_dict[label_index-1]
        else:
            final_dict[skeleton_path] = calorie_dict[label_index-1]


    #head = skeleton
    #sys.exit()
keys = [elem for elem in final_dict.keys()]
values = [elem for elem in final_dict.values()]
file = open('ntu_staffs/ntu_labels_fluctuation_max_50_pos_8' + '.csv', 'w+', newline='')
with file:
    # identifying header
    header = ['File_name', 'Calorie_label']
    writer = csv.DictWriter(file, fieldnames=header)

    # writing data row-wise into the csv file
    writer.writeheader()
    for i in range(len(keys)):
        writer.writerow({'File_name': keys[i],
                         'Calorie_label': values[i]})
#with open('calact_fluctuate.pkl', 'wb') as handle:
#    pickle.dump(dict_final, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('ntu_staffs/final_calorie_label_with_fluctuation_50_pos_8.pkl', 'wb') as handle:
    pickle.dump(final_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Fluctuaction dataset generation finished!")
