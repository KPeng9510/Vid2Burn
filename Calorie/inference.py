import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import torch
#from keras.utils import Sequence, to_categorical
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import glob
import csv
from PIL import Image
import pickle
from opts import *
from scipy.ndimage import rotate
from scipy import stats
from random import sample, randint, shuffle
import glob
import cv2
import random
cv2.setNumThreads(0)
import time
import sys

import os
import sys
#from typing import Dict
#import json
#import urllib
from torchvision.transforms import Compose, Lambda

#from pytorchvideo.data.encoded_video import EncodedVideo

sys.path.append('../')

#from models.STAM import STAM_224
from models.s3d import S3D
from models.p3d import *
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, precision_recall_curve
from utils import *
from opts import *
from scipy import stats
import gensim
from tqdm import tqdm
from Calorie.calorie_sport_loader import DataLoader_video_train as dvt
from models.i3d import InceptionI3d
#from models.model.resnet import generate_model
from models.model.resnet21d import generate_model
from models.evaluator import Evaluator, Evaluator_cal
from config import get_parser
import sklearn
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
        with open('/home/kpeng/calorie/MUSDL/Calorie/word_embedding_common_sport.pkl', 'rb') as f:
            self.world_embedding = pickle.load(f)

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
            raw_data_name = 'calorie_sport_cross_evaluation_2.pkl'
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
        file_name = self.files[idx]
        # print(file_name)
        #file_name = '/export/md0/dataset/Calorie/calorie_common_dataset/' + file_name.split('/')[-2] + '/' + \
        #            file_name.split('/')[-1]
        x_train = self._get_video(file_name)
        # x_train = np.array(x_train, np.float32)
        # x_train /= 127.5
        # x_train -= 1
        y_train = self.labels[idx]
        # y_train = to_categorical(y_train, num_classes = self.num_classes)
        data['video'] = x_train
        data['class_name'] = y_train - 1
        #print(self.MET_annotations[int(y_train-1)])
        data['real_cal']=self.MET_annotations[int(y_train-1)]
        data['filename'] = file_name.split('/')[-2]+'/'+file_name.split('/')[-1]
        # print(self.MET_annotations[int(y_train-1)])
        data['MET'] = self.proc_label(self.MET_annotations[int(y_train - 1)], mode='MET')
        data['CAL'] = self.proc_label(self.calorie_annotations[int(y_train - 1)], mode='CAL')
        data['word_emb'] = self.world_embedding
        data['calorie_prior'] = self.calorie_priori
        data['word_emb_in'] = self.world_embedding[int(y_train - 1)]

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
        # files = [elem + gaussian_noise(elem) for elem in files]
        # angle = random.randint(1, 360)
        # files = [rotate_img(elem, angle) for elem in files]
        arr = np.transpose(np.stack(files, axis=0), [0, 3, 1, 2])

        # for i in files:
        #    if os.path.isfile(i):
        #        arr.append(cv2.resize(cv2.imread(i), (224, 224)))
        #    else:
        #        arr.append(arr[-1])
        # print(arr)
        return arr


def rotate_img(img, angle, bg_patch=(224, 224)):
    # print(img.shape)
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
    mask_overflow_upper = img + noise >= 1.0
    mask_overflow_lower = img + noise < 0
    noise[mask_overflow_upper] = 1.0
    noise[mask_overflow_lower] = 0
    img += noise
    return img
def get_models(args):
    """
    Get the i3d backbone and STthe evaluator with parameters moved to GPU.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    i3d = InceptionI3d().cuda()
    #i3d = S3D(400)#generate_model(18).cuda()  #model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True).cuda() #generate_model(18).cuda() #torch.hub.load('facebookresearch/pytorchvideo', 'x3d_s', pretrained=True).cuda() #
    # load the weight file and copy the parameters
    """
    if os.path.isfile(args.i3d_pretrained_path):
        print ('loading weight file')
        weight_dict = torch.load(args.i3d_pretrained_path)
        model_dict = i3d.state_dict()
        for name, param in weight_dict.items():
            if 'module' in name:
                name = '.'.join(name.split('.')[1:])
            if name in model_dict:
                if param.size() == model_dict[name].size():
                    model_dict[name].copy_(param)
                else:
                    print (' size? ' + name, param.size(), model_dict[name].size())
            else:
                print (' name? ' + name)

        print (' loaded')
    else:
        print ('weight file?')
    """
    ##i3d.load_state_dict(torch.load(args.i3d_pretrained_path))
    #i3d.fc = torch.nn.Identity()
    i3d = i3d.cuda()
    #i3d.fc = nn.Linear(512,1024).cuda()
    evaluator_calorie = Evaluator(output_dim=1000, model_type='USDL').cuda()
    #evaluator_met = Evaluator(output_dim=100, model_type='USDL').cuda()
    evaluator_action = Evaluator(output_dim=27, model_type='USDL').cuda()
    #if args.type == 'USDL':
    #    evaluator = Evaluator(output_dim=output_dim['USDL'], model_type='USDL').cuda()
    #else:
    #    evaluator = Evaluator(output_dim=output_dim['MUSDL'], model_type='MUSDL', num_judges=num_judges).cuda()

    if len(args.gpu.split(',')) > 1:
        i3d = nn.DataParallel(i3d)
        evaluator_action = nn.DataParallel(evaluator_action)
        evaluator_calorie = nn.DataParallel(evaluator_calorie)
        #evaluator_met = nn.DataParallel(evaluator_met)
    return i3d, evaluator_action, evaluator_calorie #, evaluator_met

file_name_list = ['A02/SchoolRulesHowTheyHelpUs_walk_f_cm_np1_fr_med_2.avi', 'A12/v_JumpRope_g15_c04.avi','A20/v_TaiChi_g10_c04.avi','A26/v_JumpingJack_g01_c07.avi', 'A27/v_BodyWeightSquats_g07_c04.avi', 'A28/r83UnE5VLC8_000165_000175.mp4', 'A29/GkYrazsi4pw_000058_000068.mp4', 'A13/v_PushUps_g04_c04.avi']
#file_name_list=['A15/v_Basketball_g25_c07.avi', 'A26/v_JumpingJack_g10_c03.avi', 'A13/v_PushUps_g25_c03.avi', 'A15/v_Basketball_g03_c05.avi', 'A20/v_TaiChi_g10_c04.avi']
#file_name_list = [ 'A28/oQ8_Qur79TU_000027_000037.mp4', 'A17/v_TennisSwing_g19_c03.avi', 'A22/v_WalkingWithDog_g09_c01.avi', 'A10/person17_boxing_d1_s4.avi']
file_name_list=['A26/v_JumpingJack_g03_c03.avi']
if __name__ == '__main__':
    args = get_parser().parse_args()
    i3d, evaluator_action, evaluator_calorie = get_models(args=args)
    checkpoint = torch.load(args.checkpoint_path)
    i3d.load_state_dict(checkpoint['i3d'])
    evaluator_calorie.load_state_dict(checkpoint['evaluator_calorie'])
    evaluator_action.load_state_dict(checkpoint['evaluator_action'])
    #aggregate.load_state_dict(checkpoint['aggregate'])
    # epoch = checkpoint['epoch']
    #optimizer.load_state_dict(checkpoint['optimizer'])
    rho_best = checkpoint['rho_best']
    #rho_cross_best = checkpoint['rho_cross_best']
    dataloaders={}
    dataloaders['test'] = torch.utils.data.DataLoader(dvt(mode='test'),
                                                      batch_size=1,
                                                      num_workers=args.num_workers,
                                                      shuffle=False,
                                                      pin_memory=True,
                                                      worker_init_fn=worker_init_fn)
    dataloaders['cross_test'] = torch.utils.data.DataLoader(dvt(mode='cross_test'),
                                                            batch_size=1,
                                                            num_workers=args.num_workers,
                                                            shuffle=False,
                                                            pin_memory=True,
                                                            worker_init_fn=worker_init_fn)
    for mode in ['test', 'cross_test']:
        #train_dataset = dataloaders[mode]
        file_list = []
        pred_list = []
        action_label_list = []
        calorie_label = []
        for data in tqdm(dataloaders[mode]):
            #print(data.keys())
            calorie_label.append(torch.argmax(data['CAL']))
            file_list.append(data['filename'])
            action_label_list.extend(data['class_name'])
            videos = data['video'].cuda()
            emb_in = data['word_emb_in'].cuda()
            videos.transpose_(1, 2)  # N, C, T, H, W
            batch_size, C, frames, H, W = videos.shape
            clip_feats = torch.empty(batch_size, 10, args.num_feature_encoder).cuda()
            """
            for i in range(5):

                input = videos[:, :, 16 * i:16 * i + 32, :, :]
                if input.size()[2]<32:
                    input = videos[:, :, -32:, :, :]
                #input_sf = transform(input)
                clip_feats[:, i] = i3d(input) #.squeeze(2)
            input = videos[:, :, -32:, :, :]
            #input_sf = transform(input)
            clip_feats[:, 5] = i3d(input)  # .squeeze(2)
            """
            feature = []
            for i in range(9):
                input = videos[:, :, 10 * i:10 * i + 16, :, :]
                # if input.size()[2]<16:
                #    input = videos[:, :, -16:, :, :]
                # input_sf = transform(input)

                clip_feats[:, i], map = i3d(input)
                feature.append(map)
            input = videos[:, :, -16:, :, :]
            # input_sf = transform(input)
            clip_feats[:, 9], map = i3d(input)
            feature.append(map)
            map = torch.cat(feature, dim=2)
            #print(map.size())
            #sys.exit()
            #print(data['filename'])
            if data['filename'][0] in file_name_list:
            #    #print(data['filename'])
                np.save(data['filename'][0].split('/')[-1].split('.')[0]+'file.npy',map.cpu().detach().numpy())
                np.save(data['filename'][0].split('/')[-1].split('.')[0]+'video.npy', videos.cpu().detach().numpy())
            #    #
            
            
            # aggregate_feature = aggregate(clip_feats)
            #
            # probs_action, rep = evaluator_action(clip_feats.mean(1))
            probs_calorie,_ = evaluator_calorie(clip_feats.mean(1))

            #print(probs_calorie)
            # probs_met = evaluator_met(clip_feats.mean(1))

            # probs_calorie = evaluator_calorie(clip_feats.mean(1))
            #np.savetxt('probs_calorie.txt', probs_calorie.detach().cpu().numpy())
            preds_calorie = probs_calorie.argmax(-1)
            pred_list.append(preds_calorie.cpu().detach().numpy())
            ##sys.exit()
        #sys.exit()
        file = open('newtest_result_fluctuate_I3D'+mode+'.csv', 'w+', newline='')
        with file:
            # identifying header
            header = ['File_name', 'Calorie_prediction', 'gt']
            writer = csv.DictWriter(file, fieldnames=header)

            # writing data row-wise into the csv file
            writer.writeheader()
            for i in range(len(pred_list)):
                writer.writerow({'File_name': file_list[i],
                                 'Calorie_prediction': pred_list[i], 'gt':calorie_label[i]})




