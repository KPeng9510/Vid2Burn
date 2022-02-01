import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
import torch
# from keras.utils import Sequence, to_categorical
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import glob
from PIL import Image
import pickle
from opts import *
from scipy import stats
from random import sample, randint, shuffle
import glob
import cv2
# cv2.setNumThreads(0)
import time
import sys


def loadVideo(filepath, rescale=None, verbose=False, start_frame=0, n_frames=0):
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

    # print (filepath, start_frame, n_frames)
    new_dimensions = None
    if rescale:
        # Old: rescaling with ffmpeg. Does not work anymore
        # kwargs = {'size': rescale, "ffmpeg_params":['-loglevel', '-8']}
        # Dimensions for rescaling with PIL
        new_dimensions = list(map(int, rescale.split('x')))

    # print (filepath)

    cap = cv2.VideoCapture(filepath)

    if (start_frame > 0):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame);

    # Interate over frames in video
    images = []
    count = 0
    max_video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - start_frame - 1

    if (n_frames > max_video_length):
        # print(filepath, start_frame, n_frames)
        # print ("n_frames >= max_video_length")
        pass

    if (n_frames > 0 and n_frames <= max_video_length):
        video_length = n_frames
    else:
        video_length = max_video_length

    while cap.isOpened():
        # Extract the frame
        ret, image = cap.read()
        if rescale:
            # Convert to PIL image for rescaling
            image = Image.fromarray(image)
            image = image.resize((224, 224), resample=Image.BILINEAR)
            # Convert back to numpy array
            image = np.array(image)

        count = count + 1

        # print(np.shape(images))

        # If there are no more frames left
        # print ("len(np.shape(image)): "+str(len(np.shape(image))))
        if (count > video_length - 1 or (len(np.shape(image)) < 2)):

            cap.release()
            # Print stats
            if (verbose):
                print("Done extracting frames.\n%d frames extracted" % count)
                print("-----")
                print(np.shape(image))
                print(np.shape(images))
            break

        images.append(image)

    # images = np.array(images)

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
        # print(tmp_data[j])
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
        img_datas.append(img / 255)

    return img_datas


class DataLoader_video_train(Dataset):
    def __init__(self, mode, batch_size=1):
        self.num_frames = 64
        self.C = 3
        self.H, self.W = 256, 256
        self.MET_annotations = [4.8, 3.5, 1.5, 1.3, 8, 1.3, 4, 8, 12.8, 7.8, 7, 12.3, 8, 3, 6.5, 5.8, 7.3, 5.8, 3, 3,
                                3.5, 3, 4.8, 7, 5, 7.7, 5.5, 0, 0, 0, 0, 0, 0]
        self.calorie_annotations = [344, 251, 107, 93, 573, 93, 286, 573, 916, 396, 501, 881, 572, 215, 465, 415, 523,
                                    415, 215, 215, 251, 215, 344, 501, 358, 550, 393, 209, 467, 68, 93, 170, 165]

        self.batch_size = batch_size
        self.path = '/cvhci/data/activity/Calorie/calorie_common_dataset_train_test_split/'
        self.files = []
        self.labels = []
        if mode == 'train':
            raw_data_name = 'colorie_train_x.pkl'
            action_label_name = 'colorie_train_y.pkl'
            file = open(self.path + raw_data_name, 'rb')
            self.files = pickle.load(file)
            # print(self.files)
            file = open(self.path + action_label_name, 'rb')
            self.labels = pickle.load(file)
        elif mode == 'test':
            raw_data_name = 'colorie_test_x.pkl'
            action_label_name = 'colorie_test_y.pkl'
            file = open(self.path + raw_data_name, 'rb')
            self.files = pickle.load(file)
            invalid = self.files.index(
                '/cvhci/data/activity/Calorie/calorie_common_dataset/A10/person15_boxing_d4_s4.avi')
            del self.files[invalid]
            file = open(self.path + action_label_name, 'rb')
            self.labels = pickle.load(file)
            # if 27 in self.labels:
            #    print('True')
            # sys.exit()
            del self.labels[invalid]
        elif mode == 'cross_test':
            self.cross_path = '/cvhci/data/activity/Calorie/calorie_cross_evaluation_pkl/'
            raw_data_name = 'calorie_sport_cross_evaluation.pkl'
            file = open(self.cross_path + raw_data_name, 'rb')
            self.files = pickle.load(file)
            self.labels = []
            for item in self.files:
                self.labels.append(int(item.split('/')[-2].split('A')[-1]))
            # file = open(self.path+action_label_name,'rb')
            # self.labels = pickle.load(file)
        else:
            print('No train/test mode selected, problem occured.')
            sys.exit()

        # with open(paths['split_path'] + 'train.csv', newline='') as csvfile:
        #    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        #    self.files=[row[0].split('.')[0] for row in spamreader]
        # self.files = [i.strip() for i in open(path1).readlines()]
        self.stack_size = 90
        self.num_classes_action_labels = 27
        self.stride = 1

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
        #
        # Each judge choose a score from [0, 0.5, ..., 9.5, 10], we normalize it into 0~20
        # tmp = [stats.norm.pdf(np.arange(1000), loc=judge_score * (1000-1) / judge_max, scale=self.args.std).astype(np.float32)
        #        for judge_score in data['CAL']]
        # tmp = np.stack(tmp)
        # data['soft_calorie_gt'] = tmp / tmp.sum(axis=-1, keepdims=True)  # 7x21
        return data

    def __len__(self):
        return int(len(self.files))

    def __getitem__(self, idx):
        data = {}
        path = '/export/local_data/datasets/CalSports/Calorie/'
        file_name = self.files[idx]
        file_name = path + file_name.split('/')[-3]+'/'+file_name.split('/')[-2]+'/'+file_name.split('/')[-1]
        x_train = self._get_video(file_name)
        # x_train = np.array(x_train, np.float32)
        # x_train /= 127.5
        # x_train -= 1
        y_train = self.labels[idx]
        # y_train = to_categorical(y_train, num_classes = self.num_classes)
        data['video'] = x_train
        data['class_name'] = y_train - 1
        # print(self.MET_annotations[int(y_train-1)])
        data['MET'] = self.proc_label(self.MET_annotations[int(y_train - 1)], mode='MET')
        data['CAL'] = self.proc_label(self.calorie_annotations[int(y_train - 1)], mode='CAL')
        return data

    def _get_video(self, vid_name):
        images = loadVideo(vid_name)
        arr = torch.zeros(self.num_frames, self.C, self.H, self.W)
        for index, image in enumerate(images):
            images[index] = Image.fromarray(image.astype('uint8'), 'RGB')

        # images.sort()
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

        # files.sort()
        files = data_process(files, 224)
        arr = np.transpose(np.stack(files, axis=0), [0, 3, 1, 2])
        # for i in files:
        #    if os.path.isfile(i):
        #        arr.append(cv2.resize(cv2.imread(i), (224, 224)))
        #    else:
        #        arr.append(arr[-1])
        # print(arr)
        return arr


if __name__ == '__main__':
    train_dataset = DataLoader_video_train(mode='train', batch_size=1)
    print(len(train_dataset))
    for step, data in enumerate(train_dataset):
        # for i in range(len(train_dataset)):
        print(data['class_name'])
        # print(train_dataset[100])
        print(data['video'].shape)
        print(data['MET'])
        print(data['CAL'])
        sys.exit()
