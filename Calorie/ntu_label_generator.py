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
path = 'ntu_staffs/ntu_calorie_all_skeleton.pkl'
f=open(path,'rb')
dataset=pickle.load(f)
print('Total dataset number is ',len(dataset))
final_label=[]
final_fps = []
category_wise_label = {}
for i in range(49):
    category_wise_label[i]=[]
for skeleton_path in dataset:
    label_index = int(skeleton_path.split('/')[-1].split('A')[1].split('.skeleton')[0])
    name = int(skeleton_path.split('/')[-1].split('.')[0][1:4])
    #if name in [1,2,3,4,5,6,7]:
    if True:
        print(skeleton_path)
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
        category_wise_label[label_index-1].append(np.clip(time_length*label/(4.19*1000), 63, 400))
for i in range(49):
    print(i+1, 'calorie', sum(category_wise_label[i])/len(category_wise_label[i]))
print(sum(final_fps)/len(final_fps))
#print(max(final_label))
#print(min(final_label))




    #head = skeleton
    #sys.exit()