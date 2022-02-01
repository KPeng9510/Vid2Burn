

import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--log_info',
                        type=str,
                        help='info that will be displayed when logging',
                        default='Exp1')

    parser.add_argument('--std',
                        type=float,
                        help='standard deviation for gaussian distribution learning',
                        default=5)

    parser.add_argument('--distributed',
                        type=bool,
                        help='standard deviation for gaussian distribution learning',
                        default=False)
    parser.add_argument('--ngpus_per_node',
                        type=int,
                        help='standard deviation for gaussian distribution learning',
                        default=1)
    parser.add_argument('--gpu_num',
                        type=float,
                        help='standard deviation for gaussian distribution learning',
                        default=2)
    parser.add_argument('--save',
                        action='store_true',
                        help='if set true, save the best model',
                        default=True)
    parser.add_argument('--load_checkpoint',
                        action='store_true',
                        help='if set true, save the best model',
                        default=False)
    parser.add_argument('--type',
                        type=str,
                        help='type of the model: USDL or MUSDL',
                        choices=['USDL', 'MUSDL'],
                        default='USDL')

    parser.add_argument('--lr',
                        type=float,
                        help='learning rate',
                        default=1e-4)
    parser.add_argument('--local_rank',
                        type=int,
                        help='learning rate',
                        default=0)
    parser.add_argument('--weight_decay',
                        type=float,
                        help='L2 weight decay',
                        default=1e-5)
    parser.add_argument('--weight_calorie',
                        type=float,
                        help='Loss weight calorie',
                        default=0.5)
    parser.add_argument('--weight_met',
                        type=float,
                        help='loss weight MET',
                        default=0)
    parser.add_argument('--weight_action',
                        type=float,
                        help='Loss weight action',
                        default=0.5)
    parser.add_argument('--weight_causal',
                        type=float,
                        help='Loss weight action',
                        default=0.002)
    parser.add_argument('--weight_bias',
                        type=float,
                        help='Loss weight action',
                        default=0.004)
    parser.add_argument('--weight_output_bias',
                        type=float,
                        help='Loss weight action',
                        default=0.003)
    parser.add_argument('--temporal_aug',
                        type=int,
                        help='the maximum of random temporal shift, ranges from 0 to 6',
                        default=6)

    parser.add_argument('--seed',
                        type=int,
                        help='manual seed',
                        default=1)

    parser.add_argument('--num_workers',
                        type=int,
                        help='number of subprocesses for dataloader',
                        default=4)

    parser.add_argument('--gpu',
                        type=str,
                        help='id of gpu device(s) to be used',
                        default='1')

    parser.add_argument('--train_batch_size',
                        type=int,
                        help='batch size for training phase',
                        default=2)

    parser.add_argument('--test_batch_size',
                        type=int,
                        help='batch size for test phase',
                        default=2)

    parser.add_argument('--num_epochs',
                        type=int,
                        help='number of training epochs',
                        default=40)
    parser.add_argument('--i3d_pretrained_path',
                        type=str,
                        help='number of training epochs',
                        default='/home/kpeng/calorie/MUSDL/pretrained/r3d18_200ep.pth')
    parser.add_argument('--save_path',
                        type=str,
                        help='number of training epochs',
                        default='/home/kpeng/calorie/model_saved/r3dtest')
    parser.add_argument('--checkpoint_path',
                        type=str,
                        help='number of training epochs',
                        default='/home/kpeng/calorie/modelweights/train_versuch/main_I3D_STS/ckpts/training_baseline_R3D_test1best_cross.pt')
    parser.add_argument('--save_model_name',
                        type=str,
                        help='number of training epochs',
                        default='training_baseline_R3D_test1')
    parser.add_argument('--num_feature_encoder',
                        type=int,
                        help='number of training epochs',
                        default=1039)
    return parser
