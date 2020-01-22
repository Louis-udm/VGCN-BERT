# -*- coding: utf-8 -*-

# # # #
# train_vgcn_bert.py
# @author Zhibin.LU
# @created 2019-09-01T19:41:49.613Z-04:00
# @last-modified 2020-01-21T20:39:57.444Z-05:00
# @website: https://louis-udm.github.io
# @description 
# # # #


# %%
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import argparse
import gc

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler
from torch.utils import data


# from tqdm import tqdm, trange
import collections
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertModel, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam #, warmup_linear

def set_work_dir(local_path="stage/VGCN-BERT", server_path="myprojects/VGCN-BERT"):
    if os.path.exists(os.path.join(os.getenv("HOME"), local_path)):
        os.chdir(os.path.join(os.getenv("HOME"), local_path))
    elif os.path.exists(os.path.join(os.getenv("HOME"), server_path)):
        os.chdir(os.path.join(os.getenv("HOME"), server_path))
    else:
        raise Exception('Set work path error!')

set_work_dir()
print('Current host:',os.uname()[1],'Current dir:', os.getcwd())

import random
random.seed(44)
np.random.seed(44)
torch.manual_seed(44)

cuda_yes = torch.cuda.is_available()
if cuda_yes:
    torch.cuda.manual_seed_all(44)
device = torch.device("cuda:0" if cuda_yes else "cpu")

#%%
'''
Configuration
'''

parser = argparse.ArgumentParser()
parser.add_argument('--ds', type=str, default='mr')
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--load', type=int, default=0)
parser.add_argument('--sw', type=int, default='0')
parser.add_argument('--dim', type=int, default='16')
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--l2', type=float, default=0.01)
parser.add_argument('--model', type=str, default='VGCN_Bert')
args = parser.parse_args()
cfg_ds = args.ds
n10fold=args.fold
cfg_model_type=args.model
cfg_stop_words=True if args.sw==1 else False
will_train_mode_from_checkpoint=True if args.load==1 else False
gcn_embedding_dim=args.dim
learning_rate0=args.lr
l2_decay=args.l2

dataset_list={'sst', 'hate', 'mr', 'cola', 'arhate'}
# hate: 10k, mr: 6753, sst: 7792, r8: 5211

if cfg_ds=='R8':
    cfg_stop_words=True

will_train_mode=True
total_train_epochs = 9 
dropout_rate = 0.2  #0.5 # Dropout rate (1 - keep probability).
if cfg_ds=='hate':
    batch_size = 12 #12
    learning_rate0 = 4e-6 #2e-5
    l2_decay = 2e-4 #data3
# elif cfg_ds=='mr':
elif cfg_ds=='sst':
    batch_size = 16 #12   
    learning_rate0 = 1e-5 #2e-5  
    # l2_decay = 0.001
    l2_decay = 0.01 #default
elif cfg_ds=='cola':
    batch_size = 16 #12
    learning_rate0 = 8e-6 #2e-5  
    l2_decay = 0.01 
elif cfg_ds=='arhate':
    total_train_epochs = 9
    batch_size = 16 #12
    # learning_rate0 = 1.1e-5 #2e-5  
    learning_rate0 = 1e-5 #2e-5  
    # l2_decay = 2e-5
    l2_decay = 0.001
    # l2_decay = 0.018
elif cfg_ds=='mr':
    batch_size = 16 #12   
    learning_rate0 = 8e-6 #2e-5  
    l2_decay = 0.01
    # l2_decay = 1e-4 #default

MAX_SEQ_LENGTH = 200+gcn_embedding_dim #512 #256 #128 #300
gradient_accumulation_steps = 1
bert_model_scale = 'bert-base-uncased'
do_lower_case = True
warmup_proportion = 0.1

if cfg_stop_words:
    data_dir='data_gcn'
else:
    data_dir='data_gcn_allwords'
output_dir = './output/'

perform_metrics_str=['weighted avg','f1-score']

# cfg_add_linear_mapping_term=False
# cfg_vocab_adj='all'
cfg_vocab_adj='pmi'
# cfg_vocab_adj='tf'
cfg_adj_npmi_threshold=0.2
cfg_adj_tf_threshold=0
classifier_act_func=nn.ReLU()

resample_train_set=False # if mse and resample, then do resample
do_softmax_before_mse=True
# HATE_OFFENSIVE and LARGETWEETS
if cfg_ds in ('hate','dahate'):
    cfg_loss_criterion = 'mse'
else:
    cfg_loss_criterion='cle'
if n10fold>0:
    model_file_save=cfg_model_type+str(gcn_embedding_dim)+'_model_'+cfg_ds+'_'+cfg_loss_criterion+'_'+"sw"+str(int(cfg_stop_words))+'_'+str(n10fold)+'.pt'
else:
    model_file_save=cfg_model_type+str(gcn_embedding_dim)+'_model_'+cfg_ds+'_'+cfg_loss_criterion+'_'+"sw"+str(int(cfg_stop_words))+'.pt'

model_file_evaluate=model_file_save


print(cfg_model_type+' Start at:', time.asctime())
print('\n----- Configure -----',
    '\n  cfg_ds:',cfg_ds,
    '\n  stop_words:',cfg_stop_words,
    # '\n  Vocab GCN_hidden_dim: 768 -> 1152 -> 768',
    '\n  Vocab GCN_hidden_dim: vocab_size -> 128 -> '+str(gcn_embedding_dim),
    # '\n  Learning_rate0_FT/gcn_lr0/fc_lr0:',learning_rate0,lr0_gacn,lr0_classifier,'weight_decay_FT/gcn_l2/fc_l2:',weight_decay_finetune,weight_decay_gacn,weight_decay_classifier,
    # '\n  Learning_rate0_FT/gcn_lr0/fc_lr0:',learning_rate0,'weight_decay_FT/gcn_l2/fc_l2:',l2_decay,
    '\n  Learning_rate0:',learning_rate0,'weight_decay:',l2_decay,
    '\n  Loss_criterion',cfg_loss_criterion,'softmax_before_mse',do_softmax_before_mse,
    '\n  Dropout:',dropout_rate, 'Run_adj:',cfg_vocab_adj, 'gcn_act_func: Relu',
    '\n  MAX_SEQ_LENGTH:',MAX_SEQ_LENGTH,#'valid_data_taux',valid_data_taux,
    '\n  perform_metrics_str:',perform_metrics_str,
    '\n  model_file_evaluate:',model_file_evaluate)



#%%
'''
Prepare data set
Load gacn adjacent matrix
'''
print('\n----- Prepare data set -----')
print('Load/shuffle/seperate',cfg_ds,'data set, and gacn adjacent matrix')

import pickle as pkl
import scipy.sparse as sp

objects=[]
if n10fold<=0:
    names = [ 'labels','train_y','train_y_prob', 'valid_y','valid_y_prob','test_y','test_y_prob', 'shuffled_clean_docs','vocab_adj_tf','vocab_adj_pmi','vocab_map'] 
    for i in range(len(names)):
        datafile="./"+data_dir+"/data_%s.%s"%(cfg_ds,names[i])
        with open(datafile, 'rb') as f:
            objects.append(pkl.load(f, encoding='latin1'))
    lables_list,train_y, train_y_prob,valid_y,valid_y_prob,test_y,test_y_prob, shuffled_clean_docs,gcn_vocab_adj_tf,gcn_vocab_adj,gcn_vocab_map=tuple(objects)

else:
    names = [ 'labels', 'gen_y','gen_y_prob','vocab_map']
    for i in range(len(names)):
        datafile="./"+data_dir+"/data_%s.%s"%(cfg_ds,names[i])
        with open(datafile, 'rb') as f:
            objects.append(pkl.load(f, encoding='latin1'))
    names_fold = [ 'train_y','train_y_prob', 'valid_y','valid_y_prob','test_y','test_y_prob', 'shuffled_clean_docs','vocab_adj_tf','vocab_adj'] 
    for i in range(len(names_fold)):
        datafile="./"+data_dir+"/data_%s.fold%d.%s"%(cfg_ds,n10fold,names_fold[i])
        with open(datafile, 'rb') as f:
            objects.append(pkl.load(f, encoding='latin1'))
    lables_list, gen_y, gen_y_prob, gcn_vocab_map, train_y, train_y_prob,valid_y,valid_y_prob,test_y,test_y_prob, shuffled_clean_docs, gcn_vocab_adj_tf, gcn_vocab_adj=tuple(objects)

label2idx=lables_list[0]
idx2label=lables_list[1]

y=np.hstack((train_y,valid_y,test_y))
y_prob=np.vstack((train_y_prob,valid_y_prob,test_y_prob))

examples=[]
for i,ts in enumerate(shuffled_clean_docs):
    ex=InputExample(i, ts.strip(), confidence=y_prob[i],label=y[i])
    examples.append(ex)

num_classes=len(label2idx)
gcn_vocab_size=len(gcn_vocab_map)
train_size = len(train_y)
valid_size = len(valid_y)
test_size = len(test_y)

indexs = np.arange(0, len(examples))
train_examples = [examples[i] for i in indexs[:train_size]]
valid_examples = [examples[i] for i in indexs[train_size:train_size+valid_size]]
test_examples = [examples[i] for i in indexs[train_size+valid_size:train_size+valid_size+test_size]]

if cfg_adj_tf_threshold>0:
    gcn_vocab_adj_tf.data *= (gcn_vocab_adj_tf.data > cfg_adj_tf_threshold)
    gcn_vocab_adj_tf.eliminate_zeros()
if cfg_adj_npmi_threshold>0:
    gcn_vocab_adj.data *= (gcn_vocab_adj.data > cfg_adj_npmi_threshold)
    # 去掉自环
    # gcn_vocab_adj.setdiag(0.0)
    gcn_vocab_adj.eliminate_zeros()

# 去掉自环
# gcn_vocab_adj_tf.setdiag(0.0)
# gcn_vocab_adj.setdiag(0.0)


if cfg_vocab_adj=='pmi':
    gcn_vocab_adj_list=[gcn_vocab_adj]
elif cfg_vocab_adj=='tf':
    gcn_vocab_adj_list=[gcn_vocab_adj_tf]
elif cfg_vocab_adj=='all':
    gcn_vocab_adj_list=[gcn_vocab_adj_tf,gcn_vocab_adj]

norm_gcn_vocab_adj_list=[]
for i in range(len(gcn_vocab_adj_list)):
    adj=gcn_vocab_adj_list[i] #.tocsr() #(lr是用非norm时的1/10)
    print('Zero ratio(?>66%%) for vocab adj %dth: %.8f'%(i, 100*(1-adj.count_nonzero()/(adj.shape[0]*adj.shape[1]))))
    adj=normalize_adj(adj)
    norm_gcn_vocab_adj_list.append(sparse_scipy2torch(adj.tocoo()).to(device))
gcn_adj_list=norm_gcn_vocab_adj_list
# gcn_adj_list=[sparse_scipy2torch(sp.identity(gcn_vocab_adj.shape[0],dtype=np.float32).tocoo()).to(device)]



del gcn_vocab_adj_tf,gcn_vocab_adj,gcn_vocab_adj_list
gc.collect()


def get_class_count_and_weight(y,n_classes):
    classes_count=[]
    weight=[]
    for i in range(n_classes):
        count=np.sum(y==i)
        classes_count.append(count)
        weight.append(len(y)/(n_classes*count))
    return classes_count,weight

train_classes_num, train_classes_weight = get_class_count_and_weight(train_y,len(label2idx))
loss_weight=torch.tensor(train_classes_weight).to(device)

tokenizer = BertTokenizer.from_pretrained(bert_model_scale, do_lower_case=do_lower_case)

#%%

def get_pytorch_dataloader(examples, tokenizer, batch_size, shuffle_choice, classes_weight=None, total_resample_size=-1):
    dataset = CorpusDataset(examples, tokenizer)
    if shuffle_choice==0: # shuffle==False
        return DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=CorpusDataset.pad)
    elif shuffle_choice==1: # shuffle==True
        return DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=4,
                                collate_fn=CorpusDataset.pad)
    elif shuffle_choice==2: #weighted resampled
        assert classes_weight is not None
        assert total_resample_size>0
        # weights = [13.4 if label == 0 else 4.6 if label == 2 else 1.0 for _,_,_,label in data]
        weights = [classes_weight[0] if label == 0 else classes_weight[1] if label == 1 else classes_weight[2] for _,_,_,_,label in dataset]
        sampler = WeightedRandomSampler(weights, num_samples=total_resample_size, replacement=True)
        return DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                sampler=sampler,
                                num_workers=4,
                                collate_fn=CorpusDataset.pad)


train_dataloader = get_pytorch_dataloader(train_examples, tokenizer, batch_size, shuffle_choice=0 )
valid_dataloader = get_pytorch_dataloader(valid_examples, tokenizer, batch_size, shuffle_choice=0 )
test_dataloader = get_pytorch_dataloader(test_examples, tokenizer, batch_size, shuffle_choice=0 )

# total_train_steps = int(len(train_examples) / batch_size / gradient_accumulation_steps * total_train_epochs)
total_train_steps = int(len(train_dataloader) / gradient_accumulation_steps * total_train_epochs)

print("***** Running training *****")
print('  Train_classes count:', train_classes_num)
print('  Num examples for train =',len(train_examples),', after weight sample:',len(train_dataloader)*batch_size)
print("  Num examples for validate = %d"% len(valid_examples))
print("  Batch size = %d"% batch_size)
print("  Num steps = %d"% total_train_steps)

#%%
'''
train model
'''

def evaluate(model, gcn_adj_list,predict_dataloader, batch_size, epoch_th, dataset_name):
    # print("***** Running prediction *****")
    model.eval()
    predict_out = []
    all_label_ids = []
    ev_loss=0
    total = 0
    correct = 0
    start = time.time()
    with torch.no_grad():
        for batch in predict_dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, y_prob, label_ids, gcn_swop_eye = batch
            # the parameter label_ids is None, model return the prediction score
            logits = model(gcn_adj_list,gcn_swop_eye,input_ids,  segment_ids, input_mask)

            if cfg_loss_criterion=='mse':
                if do_softmax_before_mse:
                    logits=F.softmax(logits,-1)
                # mse的时候相当于对输出概率同原始概率做mse的拟合
                loss = F.mse_loss(logits, y_prob)
            else:
                # cross entropy loss中已经包含softmax
                if loss_weight is None:
                    loss = F.cross_entropy(logits.view(-1, num_classes), label_ids)
                else:
                    loss = F.cross_entropy(logits.view(-1, num_classes), label_ids)
            ev_loss+=loss.item()

            _, predicted = torch.max(logits, -1)
            predict_out.extend(predicted.tolist())
            all_label_ids.extend(label_ids.tolist())
            # eval_accuracy = accuracy(logits.detach().cpu().numpy(), label_ids.to('cpu').numpy())
            eval_accuracy=predicted.eq(label_ids).sum().item()
            total += len(label_ids)
            correct += eval_accuracy

        f1_metrics=f1_score(np.array(all_label_ids).reshape(-1),
            np.array(predict_out).reshape(-1), average='weighted')
        # f1_metrics = classification_report(np.array(all_label_ids).reshape(-1),
        #     np.array(predict_out).reshape(-1),output_dict=True)[perform_metrics_str[0]][perform_metrics_str[1]]
        print("Report:\n"+classification_report(np.array(all_label_ids).reshape(-1),
            np.array(predict_out).reshape(-1),digits=4))
        # print(perform_metrics, f1_metrics)

    ev_acc = correct/total
    end = time.time()
    print('Epoch : %d, %s: %.3f Acc : %.3f on %s, Spend:%.3f minutes for evaluation'
        % (epoch_th, ' '.join(perform_metrics_str), 100*f1_metrics, 100.*ev_acc, dataset_name,(end-start)/60.0))
    print('--------------------------------------------------------------')
    return ev_loss, ev_acc, f1_metrics

from model_vgcn_bert import VGCN_Bert

#%%
if will_train_mode:
    print("----- Running training -----")
    if will_train_mode_from_checkpoint and os.path.exists(os.path.join(output_dir, model_file_save)):
        checkpoint = torch.load(os.path.join(output_dir, model_file_save), map_location='cpu')
        if 'step' in checkpoint:
            prev_save_step=checkpoint['step']
            start_epoch = checkpoint['epoch']
        else:
            prev_save_step=-1
            start_epoch = checkpoint['epoch']+1
        valid_acc_prev = checkpoint['valid_acc']
        perform_metrics_prev = checkpoint['perform_metrics']
        model = VGCN_Bert.from_pretrained(bert_model_scale, state_dict=checkpoint['model_state'], gcn_adj_dim=gcn_vocab_size, gcn_adj_num=len(gcn_adj_list), gcn_embedding_dim=gcn_embedding_dim, num_labels=len(label2idx))
        pretrained_dict=checkpoint['model_state']
        net_state_dict = model.state_dict()
        pretrained_dict_selected = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}
        net_state_dict.update(pretrained_dict_selected)
        model.load_state_dict(net_state_dict)
        print('Loaded the pretrain model:',model_file_save,', epoch:',checkpoint['epoch'],'step:',prev_save_step,'valid acc:',
            checkpoint['valid_acc'],' '.join(perform_metrics_str)+'_valid:', checkpoint['perform_metrics'])

    else:
        start_epoch = 0
        valid_acc_prev = 0
        perform_metrics_prev = 0
        model = VGCN_Bert.from_pretrained(bert_model_scale, gcn_adj_dim=gcn_vocab_size, gcn_adj_num=len(gcn_adj_list),gcn_embedding_dim=gcn_embedding_dim, num_labels=len(label2idx))
        prev_save_step=-1

    model.to(device)

    param_optimizer = list(model.named_parameters())
    # optimizer = BertAdam(optimizer_grouped_parameters, lr=learning_rate0, warmup=warmup_proportion, t_total=total_train_steps)
    optimizer = BertAdam(model.parameters(), lr=learning_rate0, warmup=warmup_proportion, t_total=total_train_steps, weight_decay=l2_decay)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate0)

    # train using only BertForTokenClassification
    train_start = time.time()
    global_step_th = int(len(train_examples) / batch_size / gradient_accumulation_steps * start_epoch)
    # for epoch in trange(start_epoch, total_train_epochs, desc="Epoch"):
    
    all_loss_list={'train':[],'valid':[],'test':[]}
    all_f1_list={'train':[],'valid':[],'test':[]}
    for epoch in range(start_epoch, total_train_epochs):
        tr_loss = 0
        ep_train_start = time.time()
        model.train()
        optimizer.zero_grad()
        # for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        for step, batch in enumerate(train_dataloader):
            if prev_save_step >-1:
                if step<=prev_save_step: continue
            if prev_save_step >-1: 
                prev_save_step=-1
            batch = tuple(t.to(device) for t in batch)
            # input_ids, input_mask, segment_ids, y_prob, label_ids, tag_ids = batch
            input_ids, input_mask, segment_ids, y_prob, label_ids, gcn_swop_eye = batch
            
            logits = model(gcn_adj_list, gcn_swop_eye, input_ids, segment_ids, input_mask)

            if cfg_loss_criterion=='mse':
                if do_softmax_before_mse:
                    logits=F.softmax(logits,-1)
                # mse的时候相当于对输出概率同原始概率做mse的拟合
                loss = F.mse_loss(logits, y_prob)
            else:
                # cross entropy loss中已经包含softmax
                if loss_weight is None:
                    loss = F.cross_entropy(logits, label_ids)
                else:
                    loss = F.cross_entropy(logits.view(-1, num_classes), label_ids, loss_weight)

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            # import pudb;pu.db
            loss.backward()

            tr_loss += loss.item()
            # gradient_accumulation_steps=1时，每个mini batch都会更新
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step_th += 1
            if step % 40 == 0:
                print("Epoch:{}-{}/{}, Train {}: {} ".format(epoch, step, len(train_dataloader), cfg_loss_criterion,loss.item()))
                if step>39:
                    to_save={'epoch': epoch,'step':step, 'model_state': model.state_dict(),
                                'valid_acc': -1, 'lower_case': do_lower_case,
                                'perform_metrics':-1}
                    # torch.save(to_save, os.path.join(output_dir, model_file_save))

        print('--------------------------------------------------------------')
        # _,_,tr_f1 = evaluate(model, gcn_adj_list, train_dataloader, batch_size, epoch, 'Train_set')
        valid_loss,valid_acc,perform_metrics = evaluate(model, gcn_adj_list, valid_dataloader, batch_size, epoch, 'Valid_set')
        test_loss,_,test_f1 = evaluate(model, gcn_adj_list, test_dataloader, batch_size, epoch, 'Test_set')
        all_loss_list['train'].append(tr_loss)
        all_loss_list['valid'].append(valid_loss)
        all_loss_list['test'].append(test_loss)
        # all_f1_list['train'].append(tr_f1)
        all_f1_list['valid'].append(perform_metrics)
        all_f1_list['test'].append(test_f1)
        print("Epoch:{} completed, Total Train Loss:{}, Valid Loss:{}, Spend {}m ".format(epoch, tr_loss, valid_loss, (time.time() - train_start)/60.0))
        # Save a checkpoint
        # if valid_acc > valid_acc_prev:
        if perform_metrics > perform_metrics_prev:
            # model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            to_save={'epoch': epoch, 'model_state': model.state_dict(),
                        'valid_acc': valid_acc, 'lower_case': do_lower_case,
                        'perform_metrics':perform_metrics}
            torch.save(to_save, os.path.join(output_dir, model_file_save+'.good'))
            # valid_acc_prev = valid_acc
            perform_metrics_prev = perform_metrics
            test_f1_when_valid_best=test_f1
            # train_f1_when_valid_best=tr_f1
            valid_f1_best_epoch=epoch

    print('\n**Optimization Finished!,Total spend:',(time.time() - train_start)/60.0)
    print("**Valid weighted F1: %.3f at %d epoch."%(100*perform_metrics_prev,valid_f1_best_epoch))
    print("**Test weighted F1 when valid best: %.3f"%(100*test_f1_when_valid_best))
