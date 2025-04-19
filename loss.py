import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import pdb
from scipy.optimize import minimize


def EntropyM(input_,mask):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon) 
    entropy = torch.sum(entropy, dim=1)
    entropy = entropy * mask.float()
    return entropy 


def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

def CDAN(input_list, ad_net, entropy=None, coeff=None, random_layer=None):
    softmax_output = input_list[1].detach()
    feature = input_list[0]
    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))       
    batch_size = softmax_output.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0+torch.exp(-entropy)
        source_mask = torch.ones_like(entropy)
        source_mask[feature.size(0)//2:] = 0
        source_weight = entropy*source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[0:feature.size(0)//2] = 0
        target_weight = entropy*target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()
        return torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / torch.sum(weight).detach().item()
    else:
        return nn.BCELoss()(ad_out, dc_target) 

def DANN(features, ad_net):
    ad_out = ad_net(features)
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    return nn.BCELoss()(ad_out, dc_target)


class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):

        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss
        return loss

# focal neighbor loss
class CrossEntropyFeatureAugWeight(nn.Module):

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduction=True):
        super(CrossEntropyFeatureAugWeight, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets, weight):
        log_probs = self.logsoftmax(inputs)
        if self.use_gpu: targets = targets.cuda()
        loss = (- targets * log_probs.double()).sum(dim=1)
        loss  = loss * weight
        if self.reduction:
            return loss.mean()
        else:
            return loss
        return loss



class CrossEntropyOn(nn.Module):

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduction=False):
        super(CrossEntropyOn, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):

        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)
        # loss = loss * weight.float()
        if self.reduction:
            return loss.mean()
        else:
            return loss
        return loss


   
class CrossEntropy(nn.Module):

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduction=True):
        super(CrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets,weight):

        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)
        loss = loss * weight.float()
        if self.reduction:
            return loss.mean()
        else:
            return loss
        return loss

class CombinedLoss(nn.Module):
    def __init__(self, num_classes, beta=0.5, epsilon=0.1, use_gpu=True, reduction=True):
        super(CombinedLoss, self).__init__()
        self.beta = beta
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, labeled_inputs, labeled_targets, unlabeled_inputs, pseudo_labels):
        ce_loss = self.compute_ce_loss(labeled_inputs, labeled_targets)
        au_loss = self.compute_au_loss(unlabeled_inputs, pseudo_labels)
        combined_loss = ce_loss + au_loss
        return combined_loss

    def compute_ce_loss(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        ce_loss = -torch.sum(targets * log_probs, dim=1)
        return ce_loss.mean()

    def compute_au_loss(self, inputs, pseudo_labels, weights):
        eta = self.compute_eta(inputs)
        indicator = (eta >= self.beta).float()
        phi_loss = self.compute_phi(inputs, pseudo_labels, weights)
        au_loss = torch.mean(indicator * phi_loss)
        return au_loss

    def compute_eta(self, inputs):
        probabilities = torch.softmax(inputs, dim=1)
        max_prob, _ = torch.max(probabilities, dim=1)
        return max_prob

    def compute_phi(self, inputs, pseudo_labels, weights):
        log_probs = self.logsoftmax(inputs)
        selected_log_probs = torch.gather(log_probs, 1, pseudo_labels.unsqueeze(1))
        weighted_log_probs = selected_log_probs * weights.unsqueeze(1)
        return -weighted_log_probs.squeeze()