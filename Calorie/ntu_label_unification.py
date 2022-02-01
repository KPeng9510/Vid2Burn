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
import csv
cv2.setNumThreads(0)

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
path = 'ntu_staffs/final_calorie_label_with_fluctuation_50_pos_8.pkl'
initial_fluctuation = 100
f=open(path,'rb')
dataset=pickle.load(f)
print('Total dataset number is ',len(dataset))
file_list = [elem for elem in dataset.keys()]
caloire_list = [elem for elem in dataset.values()]
dict = {}
dict_processed = {}
for index in file_list:
    #print(index)
    key_name = index.split('/')[-1].split('.')[0]
    main_key_name = key_name.split('C')[0] + key_name.split('P')[1]
    dict[main_key_name] = []
for index in range(len(caloire_list)):
    skeleton_path = file_list[index]
    camera_index = int(skeleton_path.split('/')[-1].split('C')[1].split('P')[0])
    key_name = skeleton_path.split('/')[-1].split('.')[0]
    main_key_name = key_name.split('C')[0]+key_name.split('P')[1]
    calorie = caloire_list[index]
    dict[main_key_name].append(calorie)
main_key_list = [elem for elem in dict.keys()]
unified_dict = {}
for i in main_key_list:
    unified_dict[i] = []
key = [elem for elem in dict.keys()]
calories = [elem for elem in dict.values()]
for index in range(len(key)):
    print(calories[index])
    unified_dict[key[index]]=sum(calories[index])/len(calories[index])
with open('ntu_staffs/final_ntu_camera_id_unified_50_new_pos_8.pkl', 'wb') as handle:
    pickle.dump(unified_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
keys = [elem for elem in unified_dict.keys()]
values = [elem for elem in unified_dict.values()]
file = open('ntu_staffs/ntu_labels_cam_unified_max_50_new_pos_8' + '.csv', 'w+', newline='')
with file:
    # identifying header
    header = ['File_name', 'Calorie_label']
    writer = csv.DictWriter(file, fieldnames=header)

    # writing data row-wise into the csv file
    writer.writeheader()
    for i in range(len(keys)):
        writer.writerow({'File_name': keys[i],
                         'Calorie_label': values[i]})
