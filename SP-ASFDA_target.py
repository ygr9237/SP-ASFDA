import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.transforms import Normalize
from sklearn.metrics.pairwise import cosine_similarity
import scipy.stats as stats

import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from loss import  CrossEntropyOn,CrossEntropyFeatureAugWeight,CombinedLoss
import torch.nn.functional as F

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def image_train(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])
def data_load_t(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args._true_test_dset_path).readlines()
    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i

        new_tar = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'   
                    new_tar.append(line)
        txt_tar = new_tar.copy()

    dsets["test"] = ImageList_idx(txt_tar, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, num_workers=args.worker, drop_last=False)
    return dset_loaders


def data_load(args): 
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i

        new_tar = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'   
                    new_tar.append(line)
        txt_tar = new_tar.copy()
        txt_test = txt_tar.copy()

    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, num_workers=args.worker, drop_last=False)

    return dset_loaders


def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy*100, mean_ent

def train_target(args):
    dset_loaders = data_load(args)
    
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()  

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    modelpath = args.output_dir_src + '/source_F.pt'   
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_B.pt'   
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_C.pt'    
    netC.load_state_dict(torch.load(modelpath))
    

    param_group = []
    for k, v in netF.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False
    for k, v in netB.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)
   
    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0

    while iter_num < max_iter:

        try:
            inputs_test, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = iter_test.next()

        if inputs_test.size(0) == 1:
            continue

        if iter_num % interval_iter == 0:
            netF.eval()
            netB.eval()
            netC.eval()

            if iter_num /interval_iter == 0:
                mem_label,loc_label,true_label= obtain_label(dset_loaders['test'], netF, netB, netC, args)
            else:
                mem_label = obtain_label_rectify(dset_loaders['test'], netF, netB, netC, args,loc_label,true_label)
            mem_label = torch.from_numpy(np.array(mem_label)).cuda()
            netF.train()
            netB.train()
            netC.train()

        
        inputs_test = inputs_test.cuda()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        features_test = netB(netF(inputs_test))
        outputs_test = netC(features_test)
        
        pred = mem_label[tar_idx]
        classifier_loss = CombinedLoss(num_classes=args.class_num)(outputs_test, pred.cuda()).float()
        
        if args.ent:
            softmax_out = nn.Softmax(dim=1)(outputs_test)
            entropy_loss = torch.mean(loss.Entropy(softmax_out))
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                entropy_loss -= gentropy_loss
            im_loss = entropy_loss * args.ent_par
            classifier_loss = classifier_loss + im_loss 

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()
        best_acc = 0
        best_F = netF
        best_B = netB
        best_C = netC


        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            if args.dset=='VISDA-C':
                acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te) + '\n' + acc_list
            else:
                acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC, False)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te)

            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')
            if best_acc <= acc_s_te:
                best_acc = acc_s_te
                best_F = netF
                best_B = netB
                best_C = netC
            netF.train()
            netB.train()
            netC.train()     
    if args.issave:
        torch.save(best_F.state_dict(), osp.join(args.output_dir, "target_F_.pt"))
        torch.save(best_B.state_dict(), osp.join(args.output_dir, "target_B_.pt"))
        torch.save(best_C.state_dict(), osp.join(args.output_dir, "target_C_.pt"))
    return netF, netB, netC

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def obtain_label(loader, netF, netB, netC, args):
    h_dict = {}
    loc_dict = {}
    fea_sel_dict = {}
    label_sel_dict = {}

    for cls in range(args.class_num):
        h_dict[cls] = []
        loc_dict[cls] = []
        fea_sel_dict[cls] = []
        label_sel_dict[cls] = []

    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        sel_path = iter_test.dataset.imgs
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            feas_uniform = F.normalize(feas)
            outputs = netC(feas)
            if start_test:
                all_fea = feas_uniform.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas_uniform.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)  
    con, predict = torch.max(all_output, 1)
    accuracy_ini = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])


    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count > args.threshold)
    labelset = labelset[0]
    

    dd = cdist(all_fea, initc[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]
    for round in range(50):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

        confidence = 1 - (dd.min(axis=1) / (dd.min(axis=1) + dd.max(axis=1)))
        if confidence > args.confidence_threshold:
            break
    
    accuracy = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    if(len(labelset) < args.class_num):
        print("missing classes")

    loc_label = []
    true_label = []
    index = 0
    index_v = 0
    closest_samples_indices = np.argsort(dd, axis=0)[:args.K_evg_loc]
    closest_samples = all_fea[closest_samples_indices]

    for index, i in enumerate(closest_samples):
        all_output[i] = all_label[i]
        loc_label.append(i)
        true_label.append(int(all_label[i]))
        pred_label[i] = all_label[i]
        index += 1
        index_v += 1

    print("needed labeled")
    print(len(loc_label))

    unlabeled_indices = np.setdiff1d(np.arange(len(all_fea)), loc_label)
    unlabeled_samples = all_fea[unlabeled_indices]
    similarities = cosine_similarity(unlabeled_samples, all_fea[loc_label])
    for idx, unlabeled_idx in enumerate(unlabeled_indices):
        unlabeled_similarities = similarities[idx]
        nearest_neighbors_indices = np.argsort(unlabeled_similarities)[-args.K_num_loc:]

        weighted_labels = all_label[loc_label][nearest_neighbors_indices] * unlabeled_similarities[
            nearest_neighbors_indices]
        updated_label = np.argmax(np.bincount(weighted_labels.astype(int)))
        entropy = stats.entropy(np.bincount(weighted_labels.astype(int)))
        max_probability = np.max(np.bincount(weighted_labels.astype(int))) / args.K_num_loc
        eta = ((1 - entropy) + max_probability) / 2

        if eta > 0.8:
            pred_label[unlabeled_idx] = updated_label


    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)

    log_str = 'Accuracy = {:.2f}% -> {:.2f}% -> {:.2f}%'.format(accuracy_ini * 100, accuracy * 100, acc * 100)
    print(log_str +'\n')
    args.out_file.write(log_str+'\n')
    args.out_file.flush()

    pred_true = []

    for index in range(len(pred_label)):
        label = np.zeros(args.class_num)
        if index in loc_label:
            label[true_label[loc_label.index(index)]] = 1.0
        else:
            label[pred_label[index]] = 1.0
        pred_true.append(label)

    return pred_true,loc_label,true_label

def obtain_label_rectify(loader, netF, netB, netC, args,loc_label,true_label):
    h_dict = {}
    loc_dict = {}
    fea_sel_dict = {}
    label_sel_dict = {}
    for cls in range(args.class_num):
        h_dict[cls] = []
        loc_dict[cls] = []
        fea_sel_dict[cls] = []
        label_sel_dict[cls] = []

    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            feas_uniform = F.normalize(feas)
            outputs = netC(feas)
            if start_test:
                all_fea = feas_uniform.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas_uniform.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    con, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
        
    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    
    labelset = np.where(cls_count>args.threshold)
    labelset = labelset[0]
    dd = cdist(all_fea, initc[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]
    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    unlabeled_indices = np.setdiff1d(np.arange(len(all_fea)), loc_label)
    unlabeled_samples = all_fea[unlabeled_indices]
    similarities = cosine_similarity(unlabeled_samples, all_fea[loc_label])
    for idx, unlabeled_idx in enumerate(unlabeled_indices):
        unlabeled_similarities = similarities[idx]
        nearest_neighbors_indices = np.argsort(unlabeled_similarities)[-args.K_num_loc:]

        weighted_labels = all_label[loc_label][nearest_neighbors_indices] * unlabeled_similarities[
            nearest_neighbors_indices]
        updated_label = np.argmax(np.bincount(weighted_labels.astype(int)))
        entropy = stats.entropy(np.bincount(weighted_labels.astype(int)))
        max_probability = np.max(np.bincount(weighted_labels.astype(int))) / args.K_num_loc
        eta = ((1 - entropy) + max_probability) / 2

        if eta > 0.8:
            pred_label[unlabeled_idx] = updated_label

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format( accuracy * 100,acc * 100)
    print(log_str+'\n')
    args.out_file.write(log_str+'\n')
    args.out_file.flush()
    print(len(loc_label))
    pred_true = []
    for index in range(len(pred_label)):
        label = np.zeros(args.class_num)
        if index in loc_label:
            label[true_label[loc_label.index(index)]] = 1.0
        else:
            label[pred_label[index]] = 1.0
        pred_true.append(label)
    return pred_true

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='asfda')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=2, help="target")
    parser.add_argument('--max_epoch', type=int, default=20, help="max iterations")
    parser.add_argument('--interval', type=int, default=20)
    parser.add_argument('--KK', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home', choices=['VISDA-C', 'office', 'office-home', 'office-caltech'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="alexnet, vgg16, resnet50, res101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--SSN', type=int, default=200)
    parser.add_argument('--e_n', type=int, default=5)
    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--K_evg_loc', type=int, default=10)
    parser.add_argument('--K_num_loc', type=int, default=5)
    parser.add_argument('--threshold', type=int, default=10)
    parser.add_argument('--confidence_threshold', type=float, default=0.5)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])  
    parser.add_argument('--output', type=str, default='./object/ASFDA_Report')
    parser.add_argument('--output_src', type=str, default='./object/meckps50')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    parser.add_argument('--issave', type=bool, default=False)
    args = parser.parse_args()

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'Real_world']
        args.class_num = 65 
    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    if args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.class_num = 12
    if args.dset == 'office-caltech':
        names = ['amazon', 'caltech', 'dslr', 'webcam']
        args.class_num = 10
        
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    print(args.gpu_id)

    folder = './data/'
    args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
    args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
    args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'


    args.output_dir_src = osp.join(args.output_src, args.da, args.dset, names[args.s][0].upper())
    args.output_dir = osp.join(args.output, args.da, args.dset, names[args.s][0].upper()+names[args.t][0].upper())
    args.name = names[args.s][0].upper()+names[args.t][0].upper()

    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)


    args.savename = 'ASFDA_'+ str(args.ratio) + '_beta' +str(args.beta)+ '_kk_'+ str(args.KK)+'alpha_'+str(args.alpha)
    args.out_file = open(osp.join(args.output_dir, args.savename + '.txt'), 'w')
    args.out_file.write(print_args(args)+'\n')
    args.out_file.flush()
    train_target(args)