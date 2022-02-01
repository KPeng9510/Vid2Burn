import os
import csv
os.environ['KERAS_BACKEND'] = 'tensorflow'
from itertools import compress
import torch
# from keras.utils import Sequence, to_categorical
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import glob
from PIL import Image
import json
import pickle
from opts import *
from scipy.ndimage import rotate
from scipy import stats
from random import sample, randint, shuffle
import glob
import cv2
# from scipy.misc import imresize
import random

cv2.setNumThreads(0)
import time
import sys






class Fluactions_label():
    def __init__(self, ):
        self.num_frames = 64
        self.C = 3
        self.H, self.W = 256, 256
        self.MET_annotations = [4.8, 3.5, 1.5, 1.3, 8, 1.3, 4, 8, 12.8, 7.8, 7, 12.3, 8, 3, 6.5, 5.8, 7.3, 5.8, 3, 3,
                                3.5, 3, 4.8, 7, 5, 7.7, 5.5, 0, 0, 0, 0, 0, 0]
        self.calorie_annotations = [344, 251, 107, 93, 573, 93, 286, 573, 916, 396, 501, 881, 572, 215, 465, 415, 523,
                                    415, 215, 215, 251, 215, 344, 501, 358, 550, 393, 209, 467, 68, 93, 170, 165]
        with open('/home/kpeng/calorie/MUSDL/Calorie/word_embedding_common_sport.pkl', 'rb') as f:
            self.world_embedding = pickle.load(f)


        self.path = '/cvhci/data/activity/Calorie/calorie_common_dataset_train_test_split/'
        self.files_train = []
        self.files_test =[]
        self.files_cross_test = []

        self.labels = []
        self.all_files = []
        self.all_labels = []

        raw_data_name = 'colorie_train_x.pkl'
        action_label_name = 'colorie_train_y.pkl'
        file = open(self.path + raw_data_name, 'rb')
        self.files_train = pickle.load(file)
        print(len(self.files_train))

        self.all_files.extend(self.files_train)
        # print(self.files)
        file = open(self.path + action_label_name, 'rb')
        self.labels = pickle.load(file)
        self.all_labels.extend(self.labels)

        raw_data_name = 'colorie_test_x.pkl'
        action_label_name = 'colorie_test_y.pkl'
        file = open(self.path + raw_data_name, 'rb')
        self.files_test = pickle.load(file)
        #print(len(self.files_test))
        invalid = self.files_test.index(
            '/cvhci/data/activity/Calorie/calorie_common_dataset/A10/person15_boxing_d4_s4.avi')
        del self.files_test[invalid]
        file = open(self.path + action_label_name, 'rb')
        self.labels = pickle.load(file)
        # if 27 in self.labels:
        #    print('True')
        # sys.exit()
        del self.labels[invalid]
        self.all_files.extend(self.files_test)
        self.all_labels.extend(self.labels)
        self.cross_path = '/cvhci/data/activity/Calorie/calorie_cross_evaluation_pkl/'
        raw_data_name = 'calorie_sport_cross_evaluation.pkl'
        file = open(self.cross_path + raw_data_name, 'rb')
        self.files_cross_test = pickle.load(file)
        #print(len(self.files_cross_test))
        self.labels = []
        for item in self.files_cross_test:
            self.labels.append(int(item.split('/')[-2].split('A')[-1]))
        self.all_files.extend(self.files_cross_test)
        self.all_labels.extend(self.labels)
        #print(len(self.all_files))
        #print(len(self.all_labels))
        # file = open(self.path+action_label_name,'rb')
        # self.labels = pickle.load(file)


        # with open(paths['split_path'] + 'train.csv', newline='') as csvfile:
        #    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        #    self.files=[row[0].split('.')[0] for row in spamreader]
        # self.files = [i.strip() for i in open(path1).readlines()]
        self.stack_size = 90
        self.num_classes_action_labels = 27
        self.stride = 1
        self.calorie_anno_list = [self.proc_label(elem, mode='CAL') for elem in range(0, 1000)]
        self.calorie_priori = np.stack(self.calorie_anno_list, axis=0)

    def proc_label(self, data, mode):
        # Scores of MTL dataset ranges from 0 to 104.5, we normalize it into 0~100
        if mode == 'CAL':
            output_dim = 1000
            label_source = data
            label_max = 1000
            scale = 5
        else:
            output_dim = 100
            label_source = data
            label_max = 10
            scale = 5
        tmp = stats.norm.pdf(np.arange(output_dim), loc=label_source * (output_dim - 1) / label_max,
                             scale=scale).astype(
            np.float32)
        key = 'soft' + mode
        data = tmp / tmp.sum()
        return data

    def generate_per_video_label(self,):
        initial_fluctuation = 100
        dict = []
        motion_dict={}
        dict_final = {}
        #all_labels =
        for index in range(33):
            mask = [elem == (index+1) for elem in self.all_labels]

            names = list(compress(self.all_files, mask))

            dict.append(names)

        for index in range(33):
            motion_dict[index]=[]
            fluctuate_range = (self.calorie_annotations[index]/1000)*initial_fluctuation
            for name in dict[index]:
                if name in self.files_train:
                    mode = 'train'
                elif name in self.files_test:
                    mode = 'test'
                else:
                    mode = 'cross_test'
                skeleton_motion = self.skeleton_reader(name, mode)
                motion_dict[index].append(skeleton_motion)
            non_zero_motions = list(compress(motion_dict[index], [elem!=0 for elem in motion_dict[index]]))
            min_motion = min(non_zero_motions)
            max_motion = max(motion_dict[index])
            for i in range(len(dict[index])):
                name = dict[index][i]
                motion = motion_dict[index][i]
                fluctuation = -fluctuate_range/2 + (motion-min_motion)/(max_motion-min_motion)*fluctuate_range
                if motion == 0:
                    fluctuation = 0
                #final_labels[index].append(self.calorie_annotations[index]+fluctuation)
                dict_final[name] =int(self.calorie_annotations[index]+fluctuation)
                print(name)
                print(dict_final[name])
        keys = [ elem for elem in dict_final.keys()]
        values = [elem for elem in dict_final.values()]
        file = open('labels_fluctuation_max_100' + '.csv', 'w+', newline='')
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
        print("Fluctuaction dataset generation finished!")


    def skeleton_reader(self, name, mode):
        file_name = name
        #print(file_name)
        if mode == 'train':
            file_name = '/export/md0/dataset/Calorie/sports_skeleton/training/'+file_name.split('/')[-2]+'/'+file_name.split('/')[-1].split('.')[-2]+'.json'
        elif mode == 'test':
            file_name = '/export/md0/dataset/Calorie/sports_skeleton/training'+file_name.split('/')[-2]+'/'+file_name.split('/')[-1].split('.')[-2]+'.json'
        else:
            file_name = '/export/md0/dataset/Calorie/sports_skeleton_cross/training/'+file_name.split('/')[-2]+'/'+file_name.split('/')[-1].split('.')[-2]+'.json'
        f = open(file_name)
        x_train = json.load(f)
        skeleton_motion = proc_skeleton(x_train)
        return skeleton_motion

def proc_skeleton(data):
    skeleton = []
    person_number = []
    for i in range(len(data)):
        #print('test dataloader')
        skeleton_data = np.concatenate([np.array(data[i]['keypoints']).reshape(-1,3)[:17,:], np.array([[0,0,0]])],axis=0)
        num_person = skeleton_data.shape[0]/18
        skeleton.append(skeleton_data)
        person_number.append(num_person)
    if len(skeleton)==0:
        #print('found')
        skeleton = 61*[np.zeros([18,3])]
        person_number = 61*[1]
    #print(len(skeleton))
    #print(skeleton)
    skeleton_data = np.stack(skeleton, axis=0)
    #print(skeleton_data.shape)
    #sys.exit()
    start = np.concatenate([skeleton_data, np.zeros([skeleton_data.shape[0], 1,3])], axis=1)

    end = np.concatenate([np.zeros([len(skeleton), 1,3]), skeleton_data], axis=1)
    diff = end - start
    diff = diff[:,1:18,:]
    diff = diff ** 2
    skeleton_movement = 0.1*np.sum(diff[:,0:4,:]/4) + 0.08*np.sum(diff[:,4:8, :]/4) + 0.11*np.sum(diff[:,8:10, :]/2) \
                        + 0.30*np.sum(diff[:,10:12, :]/2) + 0.20*np.sum(diff[:,12:14,:]/2) + 0.21*np.sum(diff[:,14:16,:]/2)
    #skeleton_movement = np.sum(np.sqrt(skeleton_movement.sum(-1)))
    #person_number = np.stack(person_number, axis=0)

    #print(skeleton_data.shape)
    return skeleton_movement

if __name__ == '__main__':
    train_dataset = Fluactions_label()
    train_dataset.generate_per_video_label()
