import csv
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import torch
#from keras.utils import Sequence, to_categorical
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import glob
from PIL import Image
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
def loadVideo(filepath, rescale=None, verbose=False, start_frame = 0, n_frames = 0):
    """
    Extracts all frames of a video and combines them to a ndarray with
    shape (frame id, height, width, channels)

    Parameters
    ----------
    filepath: str
        path to video file including the video name
        (e.g '/your/file/video.avi')
    rescale: str
        rescale input video to desired resolution (e.g. rescale='160x120')
    verbose: bool
        hide or display debug information

    Returns
    -------
    Numpy: frames
        all frames of the video with (frame id, height, width, channels)
    """
    # Opens video file with imageio.
    # Returns an 'empty' numpy with shape (1,1) if the file can not be opened.



    #print (filepath, start_frame, n_frames)
    new_dimensions = None
    if rescale:
        #Old: rescaling with ffmpeg. Does not work anymore
        #kwargs = {'size': rescale, "ffmpeg_params":['-loglevel', '-8']}
        #Dimensions for rescaling with PIL
        new_dimensions = list(map(int, rescale.split('x')))


    #print (filepath)

    cap = cv2.VideoCapture(filepath)

    if (start_frame > 0):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame);

    # Interate over frames in video
    images = []
    count = 0
    max_video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - start_frame-1

    if (n_frames > max_video_length):
        #print(filepath, start_frame, n_frames)
        #print ("n_frames >= max_video_length")
        pass

    if (n_frames > 0 and n_frames <= max_video_length):
        video_length = n_frames
    else:
        video_length = max_video_length


    while cap.isOpened():
        # Extract the frame
        ret, image = cap.read()
        if rescale:
            #Convert to PIL image for rescaling
            image = Image.fromarray(image)
            image = image.resize((224,224), resample=Image.BILINEAR)
            # Convert back to numpy array
            image = np.array(image)



        count = count + 1

        #print(np.shape(images))

        # If there are no more frames left
        #print ("len(np.shape(image)): "+str(len(np.shape(image))))
        if (count > video_length -1 or (len(np.shape(image))<2)):

            cap.release()
            # Print stats
            if (verbose):
                print("Done extracting frames.\n%d frames extracted" % count)
                print("-----")
                print(np.shape(image))
                print(np.shape(images))
            break

        images.append(image)



    #images = np.array(images)

    # Print debug information
    if verbose:
        print(np.shape(images))

        print(filepath)
        print(np.shape(images))
        print(start_frame)
        print(n_frames)
        print("---")

    return images
def data_process(tmp_data, crop_size):
    img_datas = []
    crop_x = 0
    crop_y = 0
    for j in range(len(tmp_data)):
        #print(tmp_data[j])
        img = tmp_data[j]
        if img.width > img.height:
            scale = float(256) / float(img.height)
            img = np.array(cv2.resize(np.array(img), (int(img.width * scale + 1), 256))).astype(np.float32)
        else:
            scale = float(256) / float(img.width)
            img = np.array(cv2.resize(np.array(img), (256, int(img.height * scale + 1)))).astype(np.float32)
        if j == 0:
            # crop_x = random.randint(0, int(img.shape[0] - crop_size))
            # crop_y = random.randint(0, int(img.shape[1] - crop_size))
            crop_x = int((img.shape[0] - crop_size) / 2)
            crop_y = int((img.shape[1] - crop_size) / 2)
        img = img[crop_x:crop_x + crop_size, crop_y:crop_y + crop_size, :]
        img_datas.append(img/255)


    return img_datas

class DataLoader_video_train(Dataset):
    def __init__(self, mode, batch_size = 1):
        self.num_frames = 64
        self.C=3
        self.H, self.W =256,256
        self.MET_annotations = [4.8, 3.5, 1.5, 1.3, 8, 1.3, 4, 8, 12.8, 7.8, 7, 12.3, 8, 3, 6.5, 5.8, 7.3, 5.8, 3, 3,
                                3.5, 3, 4.8, 7, 5, 7.7, 5.5, 0, 0, 0,0,0,0]
        self.calorie_annotations = [344, 251, 107, 93, 573, 93, 286, 573, 916, 396, 501, 881, 572, 215, 465, 415, 523,
                                    415, 215, 215, 251, 215, 344, 501, 358, 550, 393, 209, 467, 68, 93, 170, 165]
        with open('/home/kpeng/calorie/MUSDL/Calorie/word_embedding_common_sport.pkl', 'rb') as f:
            self.world_embedding = pickle.load(f)
        with open('calact_fluctuate.pkl', 'rb') as fi:
            self.dict = pickle.load(fi)
        self.batch_size = batch_size
        self.path = '/cvhci/data/activity/Calorie/calorie_common_dataset_train_test_split/'
        self.files = []
        self.mode = mode
        self.labels = []
        if mode == 'train':
            raw_data_name = 'colorie_train_x.pkl'
            action_label_name = 'colorie_train_y.pkl'
            file = open(self.path+raw_data_name, 'rb')
            self.files = pickle.load(file)
            #print(self.files)
            file = open(self.path+action_label_name,'rb')
            self.labels = pickle.load(file)
            #print(self.dict)
            self.calorie_labels = [self.dict[elem] for elem in self.files]
        elif mode == 'test':
            raw_data_name = 'colorie_test_x.pkl'
            action_label_name = 'colorie_test_y.pkl'
            file = open(self.path+raw_data_name, 'rb')
            self.files = pickle.load(file)
            invalid = self.files.index('/cvhci/data/activity/Calorie/calorie_common_dataset/A10/person15_boxing_d4_s4.avi')
            del self.files[invalid]
            file = open(self.path+action_label_name,'rb')
            self.labels = pickle.load(file)
            #if 27 in self.labels:
            #    print('True')
            #sys.exit()
            del self.labels[invalid]
            self.calorie_labels = [self.dict[elem] for elem in self.files]
        elif mode == 'cross_test':
            self.cross_path = '/cvhci/data/activity/Calorie/calorie_cross_evaluation_pkl/'
            raw_data_name = 'calorie_sport_cross_evaluations.pkl'
            file = open(self.cross_path+raw_data_name, 'rb')
            self.files = pickle.load(file)
            self.labels = []
            for item in self.files:
                self.labels.append(int(item.split('/')[-2].split('A')[-1]))
            #file = open(self.path+action_label_name,'rb')
            #self.labels = pickle.load(file)

            self.calorie_labels = [self.dict[elem] for elem in self.files]
        else:
            print('No train/test mode selected, problem occured.')
            sys.exit()

        #with open(paths['split_path'] + 'train.csv', newline='') as csvfile:
        #    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        #    self.files=[row[0].split('.')[0] for row in spamreader]
        #self.files = [i.strip() for i in open(path1).readlines()]
        self.stack_size = 90
        self.num_classes_action_labels = 27
        self.stride = 1
        self.calorie_anno_list = [self.proc_label(elem, mode='CAL') for elem in range(0,1000)]
        self.calorie_priori = np.stack(self.calorie_anno_list, axis=0)
        print(len(self.calorie_labels))
        #print(max(self.calorie_labels))

    def proc_label(self, data, mode):
        # Scores of MTL dataset ranges from 0 to 104.5, we normalize it into 0~100
        if mode == 'CAL':
            output_dim=1000
            label_source=data
            label_max=1000
            scale=5
        else:
            output_dim=100
            label_source=data
            label_max=10
            scale=15
        #print(label_source)
        tmp = stats.norm.pdf(np.arange(output_dim), loc=label_source* (output_dim-1) / label_max, scale=scale).astype(
            np.float32)
        key='soft'+mode
        data = tmp / tmp.sum()
        #
        # Each judge choose a score from [0, 0.5, ..., 9.5, 10], we normalize it into 0~20
        #tmp = [stats.norm.pdf(np.arange(1000), loc=judge_score * (1000-1) / judge_max, scale=self.args.std).astype(np.float32)
        #        for judge_score in data['CAL']]
        #tmp = np.stack(tmp)
        #data['soft_calorie_gt'] = tmp / tmp.sum(axis=-1, keepdims=True)  # 7x21
        return data
    def __len__(self):
        return int(len(self.files))
    
    def __getitem__(self, idx):
        data={}
        file_name = self.files[idx]
        #print(file_name)
        #file_name = '/export/local_data/datasets/CalSports/Calorie/calorie_common_dataset/' + file_name.split('/')[-2]+'/'+file_name.split('/')[-1]
        #print(file_name)
        #x_train = self._get_video(file_name, self.mode)
        #x_train = np.array(x_train, np.float32)
        #x_train /= 127.5
        #x_train -= 1
        y_train = self.labels[idx]
        #y_train = to_categorical(y_train, num_classes = self.num_classes)
        #data['video']=x_train
        data['class_name']=y_train-1
        #print(self.MET_annotations[int(y_train-1)])
        data['real_cal']=self.calorie_labels[idx]
        data['filename'] = file_name.split('/')[-2]+'/'+file_name.split('/')[-1]
        #data['MET']=self.proc_label(self.MET_annotations[int(y_train-1)], mode='MET')
        data['CAL'] = self.proc_label(self.calorie_labels[idx], mode='CAL')
        #print(data['CAL'])
        #data['original_cal'] = self.calorie_labels[idx]

        #data['CAL']=self.proc_label(self.calorie_annotations[int(y_train-1)], mode='CAL')
        data['word_emb'] = self.world_embedding
        data['calorie_prior'] = self.calorie_priori
        #print(data['calorie_prior'])
        data['word_emb_in'] =  self.world_embedding[int(y_train-1)]


        return data

    def _get_video(self, vid_name, mode):
        images = loadVideo(vid_name)
        arr = torch.zeros(self.num_frames, self.C, self.H, self.W)
        for index, image in enumerate(images):
            images[index] = Image.fromarray(image.astype('uint8'), 'RGB')

        #images.sort()
        files = []
        if len(images) > (self.stack_size * self.stride):
            start = randint(0, len(images) - self.stack_size * self.stride)
            files.extend([images[i] for i in range(start, (start + self.stack_size * self.stride), self.stride)])
        elif len(images) < self.stack_size:
            files.extend(images)
            while len(files) < self.stack_size:
                files.extend(images)
            files = files[:self.stack_size]
        else:
            start = randint(0, len(images) - self.stack_size)
            files.extend([images[i] for i in range(start, (start + self.stack_size))])

        #files.sort()
        files = data_process(files, 224)
        #if mode =='train':
            #pass
            #files = [elem + gaussian_noise(elem) for elem in files]
            #angle = random.randint(-45,45)
            #index = random.randint(0,3)
            #print(index)
            #print(files[0].shape)
            #if np.random.rand() < 0.1:
            #    files = [vertical_flip(elem, True) for elem in files]
            #files = [scale_augmentation(elem) for elem in files]
            #mode = ['right', 'left', 'down', 'up']
            #files = [translate(elem, 10, mode[index]) for elem in files]
            #files = [rotate_img(elem, angle) for elem in files]

        arr = np.transpose(np.stack(files, axis=0),[0,3,1,2])

        #for i in files:
        #    if os.path.isfile(i):
        #        arr.append(cv2.resize(cv2.imread(i), (224, 224)))
        #    else:
        #        arr.append(arr[-1])
        #print(arr)
        return arr
def scale_augmentation(image, scale_range=(256,480), crop_size=224):
    scale_size = np.random.randint(*scale_range)
    image = cv2.resize(image, (scale_size, scale_size))
    image = random_crop(image, crop_size)
    return image
def random_crop(image, crop_size):
    crop_size = check_size(crop_size)
    h, w, _ = image.shape
    top = np.random.randint(0, h - crop_size[0])
    left = np.random.randint(0, w - crop_size[1])
    bottom = top + crop_size[0]
    right = left + crop_size[1]
    image = image[top:bottom, left:right, :]
    return image
def check_size(size):
    if type(size) == int:
        size = (size, size)
    if type(size) != tuple:
        raise TypeError('size is int or tuple')
    return size
def horizontal_flip(image, done=True):
    if done:
        image = image[:, ::-1, :]
    return image

def vertical_flip(image, done=True):
    if done:
        image = image[::-1, :, :]
    return image


def translate(img, shift=10, direction='right', roll=True):
    assert direction in ['right', 'left', 'down', 'up'], 'Directions should be top|up|left|right'
    img = img.copy()
    if direction == 'right':
        right_slice = img[:, -shift:].copy()
        img[:, shift:] = img[:, :-shift]
        if roll:
            img[:,:shift] = np.fliplr(right_slice)
    if direction == 'left':
        left_slice = img[:, :shift].copy()
        img[:, :-shift] = img[:, shift:]
        if roll:
            img[:, -shift:] = left_slice
    if direction == 'down':
        down_slice = img[-shift:, :].copy()
        img[shift:, :] = img[:-shift,:]
        if roll:
            img[:shift, :] = down_slice
    if direction == 'up':
        upper_slice = img[:shift, :].copy()
        img[:-shift, :] = img[shift:, :]
        if roll:
            img[-shift:,:] = upper_slice
    return img
def rotate_img(img, angle, bg_patch=(224, 224)):
    #print(img.shape)
    assert len(img.shape) <= 3, "Incorrect image shape"
    rgb = len(img.shape) == 3
    if rgb:
        bg_color = np.mean(img[:bg_patch[0], :bg_patch[1], :], axis=(0, 1))
    else:
        bg_color = np.mean(img[:bg_patch[0], :bg_patch[1]])
    img = rotate(img, angle, reshape=False)
    mask = [img <= 0, np.any(img <= 0, axis=-1)][rgb]
    img[mask] = bg_color
    return img
def gaussian_noise(img, mean=0, sigma=0.03):
    img = img.copy()
    noise = np.random.normal(mean, sigma, img.shape)
    mask_overflow_upper = img+noise >= 1.0
    mask_overflow_lower = img+noise < 0
    noise[mask_overflow_upper] = 1.0
    noise[mask_overflow_lower] = 0
    img += noise
    return img
if __name__ == '__main__':
    dataset = []
    if 1:

        dataset.append(DataLoader_video_train(mode='train',batch_size=1))
        dataset.append(DataLoader_video_train(mode='test',batch_size=1))
        dataset.append(DataLoader_video_train(mode='cross_test',batch_size=1))
        po = torch.zeros(20)
        file = open('caloriediv3'+'.csv', 'w+', newline='')
        header = ['gt', 'file_name']
        writer = csv.DictWriter(file, fieldnames=header)
        index = 0
        dict = {}
        for i in range(33):
            dict[str(i)]=[]
        print(dict.keys())
        '''
        with file:
            for dst in dataset:
                for data in dst:
                    index=int(np.floor(np.argmax(data['CAL'])/100))
                    po[index]+=1

                    # identifying header
                    #header = ['gt', 'file_name']
                    #writer = csv.DictWriter(file, fieldnames=header)

                    # writing data row-wise into the csv file
                    writer.writeheader()
                    list.append(np.argmax(data['CAL']).item())
                    if 1:
                        writer.writerow({'gt': np.argmax(data['CAL']).item(),'file_name': data['filename']})
        print(np.mean(list))
        '''
        #with open('div.pickle', 'wb') as handle:
        #    pickle.dump(list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #dataset.append(DataLoader_video_train(mode='train',batch_size=1))
    #dataset.append(DataLoader_video_train(mode='test',batch_size=1))
    #dataset.append(DataLoader_video_train(mode='cross_test',batch_size=1))
    po = torch.zeros(33)
    for dset in dataset:
        for step, data in enumerate(dset):
            dict[str(data['class_name'])].append(np.argmax(data['CAL']))
    for i in range(33):
        print(i, ':', np.mean(dict[str(i)]))
        print(i, ':', np.std(dict[str(i)]))

    #for step, data in enumerate(train_dataset):
    #    print(data['class_name'])
    #    #print(self.calorie_annotations[data['class_name']-1])
    #    # print(train_dataset[100])
    #    #print(len(data['video'][3]['keypoints']))
    #    #print(data['MET'])
    #    print(data['original_cal'])
    #    #sys.exit()
