import os
import sys
#from typing import Dict
#import json
#import urllib
from torchvision.transforms import Compose, Lambda
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel


from torch.utils.data.distributed import DistributedSampler
import torch
#from pytorchvideo.data.encoded_video import EncodedVideo

sys.path.append('../')
#from models.STAM import STAM_224
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, precision_recall_curve
from utils import *
from opts import *
from scipy import stats
from tqdm import tqdm
from Calorie.calorie_tpl_sport_loader import DataLoader_video_train
from models.i3d import InceptionI3d
#from models.model.resnet import generate_model
from models.model.resnet21d import generate_model
from models.evaluator import Evaluator
from config import get_parser
import sklearn
from sklearn.neighbors import KNeighborsClassifier
torch.autograd.set_detect_anomaly(True)

from sklearn.metrics import accuracy_score

#from models.tsn import TSN
"""
def do_knn_and_accuracies(self, embedding, label):
    # print(embeddings_and_labels)
    query_embeddings = embeddings_and_labels["val"][0]
    query_labels = embeddings_and_labels["val"][1]

    knn_indices, knn_distances = utils.stat_utils.get_knn(reference_embeddings, query_embeddings, 1, False)
    knn = KNeighborsClassifier()
    knn.fit(embedding, label)
    predicted = knn.predict(testing)


    knn_labels = reference_labels[knn_indices][:, 0]

    accuracy = accuracy_score(knn_labels, query_labels)
    print(accuracy)
    with open(self.embedding_filename + "_last", 'wb') as f:
        print("Dumping embeddings for new max_acc to file", self.embedding_filename + "_last")
        pickle.dump([query_embeddings, query_labels, reference_embeddings, reference_labels, accuracy], f)
    accuracies["accuracy"] = accuracy
    keyname = self.accuracies_keyname("mean_average_precision_at_r")  # accuracy as keyname not working
    accuracies[keyname] = accuracy
"""
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
side_size = 256
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 256
num_frames = 32
sampling_rate = 4
frames_per_second = 30
slowfast_alpha = 4
num_clips = 10
num_crops = 3
"""
class LinearClassifier(nn.Module):
    def __init__(self, class_num):

        self.classifier = nn.Linear(256,class_num)
    def forward(self, embedding):
        return self.classifier(embedding)"""
class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self):
        super(TripletLoss, self).__init__()
        self.margin = 0.2

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()

class lstm_aggregate(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1024,1024,2,bidirectional=False)
        #self.fc = nn.Linear(10,1)
    def forward(self, clip):
        feature = self.lstm(clip)
        #feature = self.fc(feature[0].permute(0,2,1)).permute(0,2,1)
        #print(feature.size())
        feature = feature[0].mean(-2)
        return feature.squeeze()

class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """

    def __init__(self):
        super().__init__()

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        #print(frames.size())
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            2,
            torch.linspace(

                0, frames.size()[2] - 1, frames.size()[2] // slowfast_alpha
            ).long().cuda(),
        )
        frame_list = [slow_pathway, fast_pathway]
        #print(slow_pathway.size())
        #print(fast_pathway.size())
        return frame_list


transform = PackPathway()
def compute_AAE(probs, label, label_action):
    pred = probs.argmax(-1)
    #print(pred)
    true_label = label.argmax(-1)
    #print(true_label)
    aae=[]
    label_action = np.stack(label_action)
    for i in range(np.min(label_action), np.max(label_action)+1):
        #print(label_action)
        mask = label_action == i
        #print(mask.sum())
        aae.append(sklearn.metrics.mean_absolute_error(true_label[mask], pred[mask])/1000)
    loss = sklearn.metrics.mean_absolute_error(true_label, pred)/1000
    return loss,aae
def logloss(y_true, y_pred, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -(y_true * np.log(y_pred)).sum(axis=1).mean()
def compute_NLL(probs, label):
    #pred = probs.argmax(-1)
    #print(pred)
    #true_label = label.argmax(-1)
    #print(true_label)
    #print(probs)
    #print(label)
    loss = logloss(label, probs)
    return loss
def test_scikit_ap(cat_labels, cat_preds):
  ''' Calculate average precision per emotion category using sklearn library.
  :param cat_preds: Categorical emotion predictions.
  :param cat_labels: Categorical emotion labels.
  :param ind2cat: Dictionary converting integer index to categorical emotion.
  :return: Numpy array containing average precision per emotion category.
  '''
  ##print(cat_labels)
  #print(cat_preds)
  ap = np.zeros(27, dtype=np.float32)
  cat_labels = np.array(cat_labels)
  #print(np.max(cat_labels))
  #sys.exit()
  cat_preds = np.array(cat_preds)
  for i in range(27):
    #print(cat_labels.shape)
    mask = cat_labels == i
    #print(i)
    #print(mask.shape)
    ap[i] = sklearn.metrics.accuracy_score(cat_labels[mask].astype(np.float), cat_preds[mask].astype(np.float))
    #print ('Category A%16s %.5f' %(str(i+1), ap[i]))
  #print ('Mean AP %.5f' %(ap.mean()))
  return ap, ap.mean()

def get_models(args):
    """
    Get the i3d backbone and STthe evaluator with parameters moved to GPU.
    """
    """
    if args.gpu_num is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        #if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
        args.local_rank = args.local_rank * args.ngpus_per_node + args.gpu_num
        dist.init_process_group(backend='nccl', 
                                world_size=1, rank=int(args.local_rank))


    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    i3d = InceptionI3d()
    #i3d = generate_model(18).cuda()  #model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True).cuda() #generate_model(18).cuda() #torch.hub.load('facebookresearch/pytorchvideo', 'x3d_s', pretrained=True).cuda() #
    #i3d.fc = torch.nn.
    i3d.load_state_dict(torch.load(args.i3d_pretrained_path))
    #i3d.fc = nn.Linear(512,1024).cuda()
    evaluator_calorie = Evaluator(output_dim=1000, model_type='USDL')
    #evaluator_met = Evaluator(output_dim=100, model_type='USDL').cuda()
    evaluator_action = Evaluator(output_dim=27, model_type='USDL')
    #if args.type == 'USDL':
    #    evaluator = Evaluator(output_dim=output_dim['USDL'], model_type='USDL').cuda()
    #else:
    #    evaluator = Evaluator(output_dim=output_dim['MUSDL'], model_type='MUSDL', num_judges=num_judges).cuda()
    """



    i3d = InceptionI3d().cuda()

    # i3d = generate_model(18).cuda()  #model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True).cuda() #generate_model(18).cuda() #torch.hub.load('facebookresearch/pytorchvideo', 'x3d_s', pretrained=True).cuda() #
    # i3d.fc = torch.nn.
    #i3d.load_state_dict(torch.load(args.i3d_pretrained_path))
    #i3d = torch.nn.SyncBatchNorm.convert_sync_batchnorm(i3d, process_group)
    # i3d.fc = nn.Linear(512,1024).cuda()
    evaluator_calorie = Evaluator(output_dim=1000, model_type='USDL').cuda()
    # evaluator_met = Evaluator(output_dim=100, model_type='USDL').cuda()
    evaluator_action = Evaluator(output_dim=27, model_type='USDL').cuda()
    """
    if args.distributed:
        #torch.cuda.set_device(args.gpu)
        #i3d.cuda(args.gpu)
        #evaluator_action.cuda(args.gpu)
        #evaluator_calorie.cuda(args.gpu)
        #args.train_batch_size = int(args.train_batch_size / args.ngpus_per_node)
        #args.test_batch_size = int(args.test_batch_size / args.ngpus_per_node)
        #args.workers = int((args.num_workers + args.ngpus_per_node - 1) / args.ngpus_per_node)
        i3d = DistributedDataParallel(i3d,device_ids=[local_rank],
                                                      output_device=local_rank,find_unused_parameters=True)
        evaluator_action = DistributedDataParallel(evaluator_action,device_ids=[local_rank],
                                                      output_device=local_rank,find_unused_parameters=True)
        evaluator_calorie = DistributedDataParallel(evaluator_calorie,device_ids=[local_rank],
                                                      output_device=local_rank,find_unused_parameters=True)
        #evaluator_met = nn.DataParallel(evaluator_met)"""
    return i3d, evaluator_action, evaluator_calorie #, evaluator_met


def compute_score(probs,  mode):
    #if model_type == 'USDL':
    if mode == 'CAL':
        pred = probs #.argmax(dim=-1) #(1000 / (output_dim-1))
        #print(pred)
    else:
        pred = probs.argmax(dim=-1)
    #else:
    #    # calculate expectation & denormalize & sort
    #    judge_scores_pred = torch.stack([prob.argmax(dim=-1) * judge_max / (output_dim-1)
    #                                     for prob in probs], dim=1).sort()[0]  # N, 7#
    #
    #    # keep the median 3 scores to get final score according to the rule of diving
    #    pred = torch.sum(judge_scores_pred[:, 2:5], dim=1) * data['difficulty'].cuda()
    return pred


def compute_loss(model_type, criterion, probs, label, mode):
    #if model_type == 'USDL':
    if mode == 'CAL':
        label=label['cal1_proc']
    else:
        label=label['act1']
    loss = criterion(torch.log(probs), label.cuda())
    #else:
    #    loss = sum([criterion(torch.log(probs[i]), label[:, i].cuda()) for i in range(num_judges)])
    return loss


def get_dataloaders(args):
    dataloaders = {}
    #sampler1 = DistributedSampler(DataLoader_video_train(mode='train'))  # 这个sampler会自动分配数据到各个gpu上
    #sampler2 = DistributedSampler(DataLoader_video_train(mode='test'))  # 这个sampler会自动分配数据到各个gpu上
    #sampler3 = DistributedSampler(DataLoader_video_train(mode='cross_test'))  # 这个sampler会自动分配数据到各个gpu上

    dataloaders['train'] = torch.utils.data.DataLoader(DataLoader_video_train(mode='train'),
                                                       batch_size=int(args.train_batch_size),
                                                       num_workers=args.num_workers,
                                                       shuffle=True,
                                                       pin_memory=True,
                                                       worker_init_fn=worker_init_fn)

    dataloaders['test'] = torch.utils.data.DataLoader(DataLoader_video_train(mode='test'),
                                                      batch_size=int(args.test_batch_size),
                                                      num_workers=args.num_workers,
                                                      shuffle=False,

                                                      pin_memory=True,
                                                      worker_init_fn=worker_init_fn)
    dataloaders['cross_test'] = torch.utils.data.DataLoader(DataLoader_video_train(mode='cross_test'),
                                                      batch_size=int(args.test_batch_size),
                                                      num_workers=args.num_workers,
                                                      shuffle=False,

                                                      pin_memory=True,
                                                      worker_init_fn=worker_init_fn)
    return dataloaders


def main(dataloaders, i3d, evaluator_action, evaluator_calorie, base_logger, args):
    # print configuration
    print('=' * 40)
    for k, v in vars(args).items():
        print(f'{k}: {v}')
    print('=' * 40)
    #aggregate = lstm_aggregate().cuda()
    criterion = nn.KLDivLoss()
    tpl_loss = TripletLoss()
    #classifier_act = LinearClassifier(6)
    #classifier_cal = LinearClassifier()
    criterion_action = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([*i3d.parameters()] + [*evaluator_action.parameters()]+[*evaluator_calorie.parameters()],
                                 lr=args.lr, weight_decay=args.weight_decay)
    epoch_best = 0
    rho_best = 0
    rho_cross_best = 0
    if(args.load_checkpoint == True):
        checkpoint = torch.load(args.checkpoint_path)
        i3d.load_state_dict(checkpoint['i3d'])
        evaluator_calorie.load_state_dict(checkpoint['evaluator_calorie'])
        evaluator_action.load_state_dict(checkpoint['evaluator_action'])
        #aggregate.load_state_dict(checkpoint['aggregate'])
        #epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        rho_best=checkpoint['rho_best']
        rho_cross_best=checkpoint['rho_cross_best']
    for epoch in range(args.num_epochs):
        log_and_print(base_logger, f'Epoch: {epoch}  Current Best: {rho_best} at epoch {epoch_best}')

        for split in ['train', 'test', 'cross_test']:
            true_scores_calorie = []
            #true_scores_met = []
            true_action = []
            pred_scores_calorie = []
            #pred_scores_met = []
            pred_scores_action = []
            test_label = []
            if split == 'train':
                i3d.train()
                evaluator_calorie.train()
                #aggregate.train()
                evaluator_action.train()
                torch.set_grad_enabled(True)
            else:
                i3d.eval()
                evaluator_action.eval()
                evaluator_calorie.eval()
                #aggregate.eval()
                torch.set_grad_enabled(False)
            if split in ['train', 'test']:
                for data in tqdm(dataloaders[split]):
                    action = [data['act1'].cuda(), data['act2'].cuda(), data['act3'].cuda()]
                    embedding_cal = []
                    embedding_act = []
                    calorie = [data['cal1_proc'].cuda(), data['cal2_proc'].cuda(), data['cal3_proc'].cuda()]
                    video = [data['img1'].cuda(), data['img2'].cuda(), data['img3'].cuda()]
                    loss = 0
                    for i in range(3):
                        videos = video[i]

                        #videos = data['video'].cuda()
                        videos.transpose_(1, 2)  # N, C, T, H, W

                        batch_size, C, frames, H, W = videos.shape
                        #print(videos.shape)
                        """
                        clip_feats = torch.empty(batch_size,6,args.num_feature_encoder).cuda()
    
                        for i in range(5):
    
                            input = videos[:, :, 16 * i:16 * i + 32, :, :]
                            if input.size()[2]<32:
                                input = videos[:, :, -32:, :, :]
                            input_sf = transform(input)
                            clip_feats[:, i] = i3d(input_sf) #.squeeze(2)
                        input = videos[:, :, -32:, :, :]
                        input_sf = transform(input)
                        clip_feats[:, 5] = i3d(input_sf) #.squeeze(2)
                        """
                        clip_feats = torch.empty(batch_size,10,args.num_feature_encoder).cuda()

                        for j in range(9):

                            input = videos[:, :, 10 * j:10 * j + 16, :, :].cuda()
                            #if input.size()[2]<16:
                            #    input = videos[:, :, -16:, :, :]
                            #input_sf = transform(input)
                            clip_feats[:, j] = i3d(input).squeeze(2)
                        input = videos[:, :, -16:, :, :]
                        #input_sf = transform(input)
                        clip_feats[:, 9] = i3d(input).squeeze(2)
                        #aggregate_feature = clip_feats.mean(1)
                        #aggregate_feature = aggregate(clip_feats)

                        probs_action, representation_act = evaluator_action(clip_feats.mean(1))
                        probs_calorie, representation_cal = evaluator_calorie(clip_feats.mean(1))
                        embedding_act.append(representation_act)
                        embedding_cal.append(representation_cal)

                        if i == 0:
                            true_scores_calorie.append(calorie[i].cpu().detach().numpy())
                            # true_scores_met.extend(data['MET'].numpy())
                            true_action.extend(action[i].cpu().detach().numpy())
                            #print('test')
                            preds_action = compute_score(probs_action, 'Action')
                            preds_calorie = compute_score(probs_calorie, 'CAL')
                            if split == 'test':
                                pred_scores_calorie.append([x.cpu().detach().numpy() for x in preds_calorie])
                                pred_scores_action.extend([x.detach() for x in preds_action])
                            if split == 'train':
                                loss_action = compute_loss(args.type, criterion_action, probs_action, data,'ACT')
                                loss_calorie = compute_loss(args.type, criterion, probs_calorie+0.0001, data, 'CAL')

                                loss_all = 0.5 * loss_action + 0.5*loss_calorie
                    if split == 'train':
                        loss_1 =loss_all+ 0.002*tpl_loss(embedding_act[0], embedding_act[1], embedding_act[2])
                        #print(tpl_loss(embedding_act[0], embedding_act[1], embedding_act[2]))
                        loss =loss_1+ 0.002*tpl_loss(embedding_cal[0], embedding_cal[1], embedding_cal[2])
                        #print(tpl_loss(embedding_cal[0], embedding_cal[1], embedding_cal[2]))
                        #print(loss_action)
                        #print(loss_calorie)
                        #torch.autograd.set_detect_anomaly(True)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        #with open(args.embedding_filename_train + "_last", 'wb') as f:
                        #    print("Dumping embeddings for new max_acc to file", self.embedding_filename + "_last")
                        #    pickle.dump([embedding_act, embedding_cal, reference_embeddings, reference_labels, accuracy],
                        #                f)
                    #if split == 'train':
                    #    break
                if(split=='test'):
                    #print(pred_scores_calorie)
                    pred_scores_calorie = np.concatenate(pred_scores_calorie, axis=0)
                    true_scores_calorie = np.concatenate(true_scores_calorie, axis=0)
                    rho_calorie = np.zeros(pred_scores_calorie.shape[0])
                    p_calorie = np.zeros(pred_scores_calorie.shape[0])
                    for i in range(pred_scores_calorie.shape[0]):
                        rho_calorie[i], p_calorie[i] = stats.spearmanr(pred_scores_calorie[i], true_scores_calorie[i])

                    rho_calorie_cat = rho_calorie
                    rho_calorie = rho_calorie.mean()
                    test_label.extend([i.item() for i in true_action])
                    accuracy_score_categoties, accuracy_score_all = test_scikit_ap(test_label, pred_scores_action)
                    aae_eval,aae_cat = compute_AAE(pred_scores_calorie, true_scores_calorie, true_action)
                    nll_eval = compute_NLL(pred_scores_calorie, true_scores_calorie)
                    for i in range(len(aae_cat)):
                        log_and_print(base_logger, f'{split} calorie SPC error for A{i}: {rho_calorie_cat[i]}')
                        log_and_print(base_logger, f'{split} calorie AAE error for A{i}: {aae_cat[i]}')
                    log_and_print(base_logger, f'{split} correlation_calorie: {rho_calorie}')
                    log_and_print(base_logger, f'{split} AAE error correlation_calorie: {aae_eval}')
                    log_and_print(base_logger, f'{split} calorie NLL: {nll_eval}')
                    log_and_print(base_logger, f'{split} action accuracy: {accuracy_score_all}')
                    log_and_print(base_logger, 'Mean AP: %.5f' %(accuracy_score_all))
                    for i in range(27):

                        log_and_print(base_logger, 'Category A%s : %.5f' % (str(i + 1), accuracy_score_categoties[i]))
                    rho = rho_calorie
                    if rho > rho_best:
                        rho_best = rho
                        epoch_best = epoch
                        log_and_print(base_logger, '-----New best found!-----')
                        if args.save:
                            torch.save({'epoch': epoch,
                                        'i3d': i3d.state_dict(),
                                        'evaluator_calorie': evaluator_calorie.state_dict(),
                                       'evaluator_action': evaluator_action.state_dict(),

                                        'optimizer': optimizer.state_dict(),
                                        'rho_best': rho_best,
                                        'rho_cross_best':rho_cross_best,
                                        'aae': aae_eval,
                                        'nll': nll_eval}, args.save_path+f'/ckpts/{args.save_model_name}.pt')
            else:
                embedding_act = []
                embedding_cal = []
                for data in tqdm(dataloaders[split]):

                    true_scores_calorie.append(data['cal1_proc'])
                    true_action.extend(data['act1'])
                    videos = data['img1'].cuda()
                    videos.transpose_(1, 2)  # N, C, T, H, W
                    batch_size, C, frames, H, W = videos.shape
                    clip_feats = torch.empty(batch_size, 10, args.num_feature_encoder).cuda()
                    """
                    for i in range(5):

                        input = videos[:, :, 16 * i:16 * i + 32, :, :]
                        if input.size()[2]<32:
                            input = videos[:, :, -32:, :, :]
                        input_sf = transform(input)
                        clip_feats[:, i] = i3d(input_sf) #.squeeze(2)
                    input = videos[:, :, -32:, :, :]
                    input_sf = transform(input)
                    clip_feats[:, 5] = i3d(input_sf)  # .squeeze(2)"""
                    for i in range(9):

                        input = videos[:, :, 10 * i:10 * i + 16, :, :]
                        #if input.size()[2]<16:
                        #    input = videos[:, :, -16:, :, :]
                        #input_sf = transform(input)
                        clip_feats[:, i] = i3d(input).squeeze(2)
                    input = videos[:, :, -16:, :, :]
                    #input_sf = transform(input)
                    clip_feats[:, 9] = i3d(input).squeeze(2)
                    #
                    #aggregate_feature = aggregate(clip_feats)
                    #
                    probs_action, embedd_act = evaluator_action(clip_feats.mean(1))
                    probs_calorie, embedd_cal = evaluator_calorie(clip_feats.mean(1))
                    embedding_act.append(embedd_act)
                    embedding_cal.append(embedd_cal)
                    #probs_met = evaluator_met(clip_feats.mean(1))

                    #probs_calorie = evaluator_calorie(clip_feats.mean(1))
                    preds_calorie = compute_score(probs_calorie, 'CAL')
                    pred_scores_calorie.append([i.cpu().detach().numpy() for i in preds_calorie])
                pred_scores_calorie = np.concatenate(pred_scores_calorie, axis=0)
                true_scores_calorie = np.concatenate(true_scores_calorie, axis=0)
                rho_calorie = np.zeros(pred_scores_calorie.shape[0])
                p_calorie = np.zeros(pred_scores_calorie.shape[0])
                for i in range(pred_scores_calorie.shape[0]):
                    rho_calorie[i], p_calorie[i] = stats.spearmanr(pred_scores_calorie[i], true_scores_calorie[i])
                rho_calorie_cat = rho_calorie
                rho_calorie = rho_calorie.mean()
                aae_eval,aae_cat = compute_AAE(pred_scores_calorie, true_scores_calorie, true_action)
                nll_eval = compute_NLL(pred_scores_calorie, true_scores_calorie)
                test_label.extend([i.item() for i in true_action])
                log_and_print(base_logger, f'{split} avg cross-evaluation correlation_calorie: {rho_calorie}')
                log_and_print(base_logger, f'{split} AAE error correlation_calorie: {aae_eval}')
                log_and_print(base_logger, f'{split} calorie NLL: {nll_eval}')
                for i in range(len(aae_cat)):
                    log_and_print(base_logger, f'{split} calorie SPC error for A{i}: {rho_calorie_cat[i]}')
                    log_and_print(base_logger, f'{split} calorie AAE error for A{i}: {aae_cat[i]}')
                if rho_calorie > rho_cross_best:
                    rho_cross_best = rho_calorie
                    epoch_best = epoch
                    log_and_print(base_logger, '-----New best found for cross-entropy!-----')
                    if args.save:
                        torch.save({'epoch': epoch,
                                    'i3d': i3d.state_dict(),
                                    'evaluator_calorie': evaluator_calorie.state_dict(),
                                    'evaluator_action': evaluator_action.state_dict(),

                                    'optimizer': optimizer.state_dict(),
                                    'rho_best': rho_best,
                                    'rho_cross_best':rho_cross_best,
                                    'aae': aae_eval,
                                    'nll': nll_eval}, args.save_path + f'/ckpts/{args.save_model_name}best_cross.pt')

if __name__ == '__main__':

    args = get_parser().parse_args()
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    if not os.path.exists(args.save_path+'/exp'):
        os.mkdir(args.save_path+'/exp')
    if not os.path.exists(args.save_path+'/ckpts'):
        os.mkdir(args.save_path+'/ckpts')
    save_log = args.save_path+'/exp'
    init_seed(args)

    base_logger = get_logger(save_log+f'/{args.type}.log', args.log_info)
    i3d, evaluator_action, evaluator_calorie = get_models(args)
    dataloaders = get_dataloaders(args)

    main(dataloaders, i3d, evaluator_action,evaluator_calorie, base_logger, args)


