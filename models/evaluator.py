import torch.nn as nn
from opts import *
import torch
from thop import profile

class causal_aggregate(torch.nn.Module):
    def __init__(self, concept_dimension):
        super(causal_aggregate, self).__init__()
        self.one_matrix = torch.ones([concept_dimension, concept_dimension])
        self.adjacency = nn.Parameter(torch.triu(self.one_matrix))
        self.adjacency.requires_grad = True
        self.softmax = nn.Softmax(dim=1)
    def forward(self, clip):
        # input clip size: batchsize,10,1024
        #print(self.adjacency.requires_grad)
        adjacency = torch.triu(self.softmax(self.adjacency))
        #print(clip.size())
        #print(adjacency.size())
        return clip@adjacency, adjacency

def compute_causal_constraint(adj, concept_dimension):
    unit = torch.eye(concept_dimension).cuda()
    return torch.trace(torch.matrix_power(unit + adj@adj, 1))


class MLP_block(nn.Module):

    def __init__(self, output_dim):
        super(MLP_block, self).__init__()
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        #self.layer_naive = nn.Linear(1024,1024)
        self.layer_representation = nn.Linear(256, 768)
        self.layer1 = nn.Linear(1039, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, output_dim)

    def forward(self, x):
        #rint(x.size())
        #x = self.activation(self.layer_naive(x))
        #x = self.layer_representation(x)
        #print(x)
        x = self.activation(self.layer1(x))
        representation = self.layer_representation(x)  # self.layer_representation(x)
        x = self.activation(self.layer2(x))
        output = self.softmax(self.layer3(x))
        return output, representation


class Evaluator(nn.Module):

    def __init__(self, output_dim, model_type='USDL', num_judges=None):
        super(Evaluator, self).__init__()

        self.model_type = model_type

        if model_type == 'USDL':
            self.evaluator = MLP_block(output_dim=output_dim)
        else:
            assert num_judges is not None, 'num_judges is required in MUSDL'
            self.evaluator = nn.ModuleList([MLP_block(output_dim=output_dim) for _ in range(num_judges)])

    def forward(self, feats_avg):  # data: NCTHW

        if self.model_type == 'USDL':
            probs, representation= self.evaluator(feats_avg)  # Nxoutput_dim
        else:
            probs = [evaluator(feats_avg) for evaluator in self.evaluator]  # len=num_judges
        return probs, representation


class MLP_block_cal(nn.Module):

    def __init__(self, output_dim):
        super(MLP_block_cal, self).__init__()
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        #self.causal_graph = causal_aggregate(concept_dimension=256)
        #self.layer_naive = nn.Linear(1024,1024)
        #self.dropout = nn.Dropout(p=0.1, inplace=False)
        self.layer_representation = nn.Linear(256, 768)
        self.layer1 = nn.Linear(1024, 256)
        self.layer2 = nn.Linear(1024, 128)
        self.layer3 = nn.Linear(128, output_dim)

    def forward(self, x, representation_act):
        #print(x.size())
        #x = self.activation(self.layer_naive(x))
        #x = self.layer_representation(x)
        #print(calorie.shape)
        #x = torch.cat([x,calorie.cuda()], dim=-1)
        #print(x.size())
        x = self.activation(self.layer1(x))
        #print(x.size())
        #print(x.size())
        representation = self.layer_representation(x)
        #print(representation_act.size())
        #representation_act, adj = self.causal_graph(representation_act)
        #print(representation_act.size())
        x = torch.cat([x, representation_act], dim=-1)
        #print(x.size())
        #loss = compute_causal_constraint(adj, 256)
        x = self.activation(self.layer2(x))
        output = self.softmax(self.layer3(x))
        loss  =0
        return output, representation, loss


class Evaluator_cal(nn.Module):

    def __init__(self, output_dim, model_type='USDL', num_judges=None):
        super(Evaluator_cal, self).__init__()

        self.model_type = model_type

        if model_type == 'USDL':
            self.evaluator = MLP_block_cal(output_dim=output_dim)

        else:
            assert num_judges is not None, 'num_judges is required in MUSDL'
            self.evaluator = nn.ModuleList([MLP_block_cal(output_dim=output_dim) for _ in range(num_judges)])

    def forward(self, feats_avg, calorie):  # data: NCTHW

        if self.model_type == 'USDL':
            probs, representation, loss = self.evaluator(feats_avg, calorie)  # Nxoutput_dim
        else:
            probs = [evaluator(feats_avg) for evaluator in self.evaluator]  # len=num_judges
        return probs, representation




class MLP_block_rep(nn.Module):

    def __init__(self, output_dim):
        super(MLP_block_rep, self).__init__()
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, output_dim)
        self.layer1 = nn.Linear(1039, 256)
    def forward(self, x):
        #print(x.size())
        #x = self.activation(self.layer_naive(x))
        #x = self.layer_representation(x)
        #print(calorie.shape)
        #x = torch.cat([x,calorie.cuda()], dim=-1)
        #print(x.size())
        x = self.activation(self.layer1(x))
        #print(x.size())
        #print(x.size())
        #representation = self.layer_representation(x)
        #print(representation_act.size())
        #representation_act, adj = self.causal_graph(representation_act)
        #print(representation_act.size())
        #x = torch.cat([x, representation_act], dim=-1)
        #print(x.size())
        #loss = compute_causal_constraint(adj, 256)
        x = self.activation(self.layer2(x))
        output = self.softmax(self.layer3(x))
        loss  =0
        representation = 0
        return output, representation, loss


class Evaluator_reg(nn.Module):

    def __init__(self, output_dim, model_type='USDL', num_judges=None):
        super(Evaluator_reg, self).__init__()

        self.model_type = model_type

        if model_type == 'USDL':
            self.evaluator = MLP_block_rep(output_dim=1)

        else:
            assert num_judges is not None, 'num_judges is required in MUSDL'
            self.evaluator = nn.ModuleList([MLP_block_cal(output_dim=output_dim) for _ in range(num_judges)])

    def forward(self, feats_avg):  # data: NCTHW

        if self.model_type == 'USDL':
            probs, representation, loss = self.evaluator(feats_avg)  # Nxoutput_dim
        else:
            probs = [evaluator(feats_avg) for evaluator in self.evaluator]  # len=num_judges
        return probs, representation
