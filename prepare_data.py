# -*- coding: utf-8 -*-

# # # #
# prepare_data.py
# @author Zhibin.LU
# @created 2019-05-22T09:28:29.705Z-04:00
# @last-modified 2020-04-15T14:51:35.359Z-04:00
# @website: https://louis-udm.github.io
# @description: pre-process data and prepare the vocabulary graph
# # # #

#%%
import random
import numpy as np
import scipy.sparse as sp
import pickle as pkl
import re
import time
import argparse
import os
import sys
from sklearn.utils import shuffle

from nltk.corpus import stopwords
import nltk

random.seed(44)
np.random.seed(44)

# import torch
# cuda_yes = torch.cuda.is_available()
# device = torch.device("cuda:0" if cuda_yes else "cpu")
# torch.manual_seed(44)
# if cuda_yes:
#     torch.cuda.manual_seed_all(44)

#%%
'''
Config:
'''
parser = argparse.ArgumentParser()
parser.add_argument('--ds', type=str, default='cola')
parser.add_argument('--sw', type=int, default=0)
args = parser.parse_args()
cfg_ds = args.ds
cfg_del_stop_words=True if args.sw==1 else False

dataset_list={'sst', 'cola'}

if cfg_ds not in dataset_list:
    sys.exit("Dataset choice error!")

will_dump_objects=True
dump_dir='data/dump_data'
if not os.path.exists(dump_dir):
    os.mkdir(dump_dir)

if cfg_del_stop_words:
    freq_min_for_word_choice=5 
    # freq_min_for_word_choice=10 #best
else:
    freq_min_for_word_choice=1 # for bert+

valid_data_taux = 0.05 
test_data_taux = 0.10

# word co-occurence with context windows
window_size = 20
if cfg_ds in ('mr','sst','cola'):
    window_size = 1000 # use whole sentence

tfidf_mode='only_tf'  
# tfidf_mode='all_tfidf' 

# 在clean doc时是否使用bert_tokenizer分词, data3时不用更好
cfg_use_bert_tokenizer_at_clean=True

bert_model_scale='bert-base-uncased'
# bert_model_scale='bert-large-uncased'
bert_lower_case=True

print('---data prepare configure---')
print('Data set: ',cfg_ds,'freq_min_for_word_choice',freq_min_for_word_choice,'window_size',window_size)
print('del_stop_words',cfg_del_stop_words,'use_bert_tokenizer_at_clean',cfg_use_bert_tokenizer_at_clean)
print('tfidf-mode',tfidf_mode,'bert_model_scale',bert_model_scale,'bert_lower_case',bert_lower_case)
print('\n')

#%%
'''
Get the tweets,y,confidence etc from data file
'''
print('Get the tweets,y,confidence etc from data file...')
start=time.time()

import pandas as pd

def del_http_user_tokenize(tweet):
    # delete [ \t\n\r\f\v]
    space_pattern = r'\s+'
    url_regex = (r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        r'[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = r'@[\w\-]+'
    tweet = re.sub(space_pattern, ' ', tweet)
    tweet = re.sub(url_regex, '', tweet)
    tweet = re.sub(mention_regex, '', tweet)
    return tweet



#%%
if cfg_ds=='sst':
    from get_sst_data import DataReader
    train, valid, test = DataReader("data/SST-2/train.txt","./data/SST-2/dev.txt","./data/SST-2/test.txt").read()
    random.shuffle(train)
    random.shuffle(valid)
    random.shuffle(test)
    train_size=len(train)
    valid_size=len(valid)
    test_size=len(test)
    print('SST-2, train_szie:%d, valid_size:%d, test_size:%d'%(train_size,valid_size,test_size))
    trainset = {}
    validset = {}
    testset = {}
    for data, dataset in [(train, trainset), (valid, validset), (test, testset)]:
        label = []
        all_text = []
        for line in data:
            label.append(line[0])
            all_text.append(line[1])
        dataset["label"] = label
        dataset["data"] = all_text
    
    label2idx = {label:i for i,label in enumerate(testset['label'])}
    idx2label = {i:label for i,label in enumerate(testset['label'])}
    corpus=trainset['data']+validset['data']+testset['data']
    y=np.array(trainset['label']+validset['label']+testset['label'])
    corpus_size=len(corpus)
    y_prob=np.eye(corpus_size,len(label2idx))[y]

elif cfg_ds=='cola':
    label2idx = {'0':0, '1':1}
    idx2label = {0:'0', 1:'1'}
    train_valid_df = pd.read_csv('data/CoLA/train.tsv', encoding='utf-8', header=None, sep='\t')
    train_valid_df = shuffle(train_valid_df)
    # use dev set as test set, because we can not get the ground true label of the real test set.
    test_df = pd.read_csv('data/CoLA/dev.tsv', encoding='utf-8', header=None, sep='\t')
    test_df = shuffle(test_df)
    train_valid_size=train_valid_df[1].count()
    valid_size=int(train_valid_size*valid_data_taux)
    train_size=train_valid_size-valid_size
    test_size=test_df[1].count()
    print('CoLA train_valid Total:',train_valid_size,'test Total:',test_size)
    df=pd.concat((train_valid_df,test_df))
    corpus = df[3]
    y = df[1].values  #y.as_matrix()
    # 获得confidence
    y_prob=np.eye(len(y), len(label2idx))[y]
    corpus_size=len(y)
    
#%%
doc_content_list=[]
for t in corpus:
    doc_content_list.append(del_http_user_tokenize(t))
max_len_seq=0
max_len_seq_idx=-1
min_len_seq=1000
min_len_seq_idx=-1
sen_len_list=[]
for i,seq in enumerate(doc_content_list):
    seq=seq.split()
    sen_len_list.append(len(seq))
    if len(seq)<min_len_seq:
        min_len_seq=len(seq)
        min_len_seq_idx=i
    if len(seq)>max_len_seq:
        max_len_seq=len(seq)
        max_len_seq_idx=i
print('Statistics for original text: max_len%d,id%d, min_len%d,id%d, avg_len%.2f' \
    %(max_len_seq, max_len_seq_idx,min_len_seq, min_len_seq_idx,np.array(sen_len_list).mean()))


#%%
'''
Remove stop words from tweets
'''
print('Remove stop words from tweets...')

if cfg_del_stop_words:
    from nltk.corpus import stopwords
    nltk.download('stopwords')
    stop_words = stopwords.words('english')
    stop_words=set(stop_words)
else:
    stop_words={}
print('Stop_words:',stop_words)

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    # string = " ".join(re.split("[^a-zA-Z]", string.lower())).strip()
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

tmp_word_freq = {}  # to remove rare words
new_doc_content_list = []

# use bert_tokenizer for split the sentence
if cfg_use_bert_tokenizer_at_clean:
    print('Use bert_tokenizer for seperate words to bert vocab')
    from pytorch_pretrained_bert import BertTokenizer
    bert_tokenizer = BertTokenizer.from_pretrained(bert_model_scale, do_lower_case=bert_lower_case)

for doc_content in doc_content_list:
    new_doc = clean_str(doc_content)
    if cfg_use_bert_tokenizer_at_clean:
        sub_words = bert_tokenizer.tokenize(new_doc)
        sub_doc=' '.join(sub_words).strip()
        new_doc=sub_doc
    new_doc_content_list.append(new_doc)
    for word in new_doc.split():
        if word in tmp_word_freq:
            tmp_word_freq[word] += 1
        else:
            tmp_word_freq[word] = 1

doc_content_list=new_doc_content_list

# for normal dataset
clean_docs = []
count_void_doc=0
for i,doc_content in enumerate(doc_content_list):
    words = doc_content.split()
    doc_words = []
    for word in words:
        # if tmp_word_freq[word] >= freq_min_for_word_choice:
        if cfg_ds in ('mr','sst','cola'):
            doc_words.append(word)
        elif word not in stop_words and tmp_word_freq[word] >= freq_min_for_word_choice:
            doc_words.append(word)
    doc_str = ' '.join(doc_words).strip()
    if doc_str == '':
        count_void_doc+=1
        # doc_str = '[unk]'
        # doc_str = 'normal'
        # doc_str = doc_content
        print('No.',i, 'is a empty doc after treat, replaced by \'%s\'. original:%s'%(doc_str,doc_content))
    clean_docs.append(doc_str)


print('Total',count_void_doc,' docs are empty.')


#%%

min_len = 10000
min_len_id = -1
max_len = 0
max_len_id = -1
aver_len = 0

for i,line in enumerate(clean_docs):
    temp = line.strip().split()
    aver_len = aver_len + len(temp)
    if len(temp) < min_len:
        min_len = len(temp)
        min_len_id=i
    if len(temp) > max_len:
        max_len = len(temp)
        max_len_id=i

aver_len = 1.0 * aver_len / len(clean_docs)
print('After tokenizer:')
print('Min_len : ' + str(min_len)+' id: '+str(min_len_id))
print('Max_len : ' + str(max_len)+' id: '+str(max_len_id))
print('Average_len : ' + str(aver_len))


#%%
'''
Build graph
'''
print('Build graph...')

if cfg_ds in ('mr', 'sst','cola'):
    shuffled_clean_docs=clean_docs
    train_docs=shuffled_clean_docs[:train_size]
    valid_docs=shuffled_clean_docs[train_size:train_size+valid_size]
    train_valid_docs=shuffled_clean_docs[:train_size+valid_size]
    train_y = y[:train_size]
    valid_y = y[train_size:train_size+valid_size]
    test_y = y[train_size+valid_size:]
    train_y_prob = y_prob[:train_size]
    valid_y_prob = y_prob[train_size:train_size+valid_size]
    test_y_prob = y_prob[train_size+valid_size:]

#%%

# build vocab using whole corpus(train+valid+test+genelization)
word_set = set()
for doc_words in shuffled_clean_docs:
    words = doc_words.split()
    for word in words:
        word_set.add(word)
        # if word in word_freq:
        #     word_freq[word] += 1
        # else:
        #     word_freq[word] = 1

vocab = list(word_set)
vocab_size = len(vocab)

vocab_map = {}
for i in range(vocab_size):
    vocab_map[vocab[i]] = i


# build vocab_train_valid
word_set_train_valid = set()
for doc_words in train_valid_docs:
    words = doc_words.split()
    for word in words:
        word_set_train_valid.add(word)
vocab_train_valid = list(word_set_train_valid)
vocab_train_valid_size = len(vocab_train_valid)
    
#%%
# a map for word -> doc_list
if tfidf_mode=='all_tf_train_valid_idf':
    for_idf_docs = train_valid_docs
else:
    for_idf_docs = shuffled_clean_docs 
word_doc_list = {}
for i in range(len(for_idf_docs)):
    doc_words = for_idf_docs[i]
    words = doc_words.split()
    appeared = set()
    for word in words:
        if word in appeared:
            continue
        if word in word_doc_list:
            doc_list = word_doc_list[word]
            doc_list.append(i)
            word_doc_list[word] = doc_list
        else:
            word_doc_list[word] = [i]
        appeared.add(word)

word_doc_freq = {}
for word, doc_list in word_doc_list.items():
    word_doc_freq[word] = len(doc_list)

#%%
'''
Doc word heterogeneous graph
and Vocabulary graph
'''
print('Calculate First isomerous adj and First isomorphic vocab adj, get word-word PMI values')

adj_y=np.hstack( ( train_y, np.zeros(vocab_size), valid_y, test_y ))
adj_y_prob=np.vstack(( train_y_prob, np.zeros((vocab_size,len(label2idx)),dtype=np.float32), valid_y_prob, test_y_prob ))

windows = []
for doc_words in train_valid_docs:
    words = doc_words.split()
    length = len(words)
    if length <= window_size:
        windows.append(words)
    else:
        for j in range(length - window_size + 1):
            window = words[j: j + window_size]
            windows.append(window)

print('Train_valid size:',len(train_valid_docs),'Window number:',len(windows))

word_window_freq = {}
for window in windows:
    appeared = set()
    for i in range(len(window)):
        if window[i] in appeared:
            continue
        if window[i] in word_window_freq:
            word_window_freq[window[i]] += 1
        else:
            word_window_freq[window[i]] = 1
        appeared.add(window[i])

word_pair_count = {}
for window in windows:
    appeared = set()
    for i in range(1, len(window)):
        for j in range(0, i):
            word_i = window[i]
            word_i_id = vocab_map[word_i]
            word_j = window[j]
            word_j_id = vocab_map[word_j]
            if word_i_id == word_j_id:
                continue
            word_pair_str = str(word_i_id) + ',' + str(word_j_id)
            if word_pair_str in appeared:
                continue
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1
            appeared.add(word_pair_str)
            # two orders
            word_pair_str = str(word_j_id) + ',' + str(word_i_id)
            if word_pair_str in appeared:
                continue
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1
            appeared.add(word_pair_str)


#%%
from math import log

row = []
col = []
weight = []
tfidf_row = []
tfidf_col = []
tfidf_weight = []
vocab_adj_row=[]
vocab_adj_col=[]
vocab_adj_weight=[]

num_window = len(windows)
tmp_max_npmi=0
tmp_min_npmi=0
tmp_max_pmi=0
tmp_min_pmi=0
for key in word_pair_count:
    temp = key.split(',')
    i = int(temp[0])
    j = int(temp[1])
    count = word_pair_count[key]
    word_freq_i = word_window_freq[vocab[i]]
    word_freq_j = word_window_freq[vocab[j]]
    pmi = log((1.0 * count / num_window) / (1.0 * word_freq_i * word_freq_j/(num_window * num_window)))
    # 使用normalized pmi:
    npmi = log(1.0 * word_freq_i * word_freq_j/(num_window * num_window))/log(1.0 * count / num_window) -1
    if npmi>tmp_max_npmi: tmp_max_npmi=npmi
    if npmi<tmp_min_npmi: tmp_min_npmi=npmi
    if pmi>tmp_max_pmi: tmp_max_pmi=pmi
    if pmi<tmp_min_pmi: tmp_min_pmi=pmi
    if pmi>0:
        row.append(train_size + i)
        col.append(train_size + j)
        weight.append(pmi)
    if npmi>0:
        vocab_adj_row.append(i)
        vocab_adj_col.append(j)
        vocab_adj_weight.append(npmi)
print('max_pmi:',tmp_max_pmi,'min_pmi:',tmp_min_pmi)
print('max_npmi:',tmp_max_npmi,'min_npmi:',tmp_min_npmi)


#%%
print('Calculate doc-word tf-idf weight')

n_docs = len(shuffled_clean_docs)
doc_word_freq = {}
for doc_id in range(n_docs):
    doc_words = shuffled_clean_docs[doc_id]
    words = doc_words.split()
    for word in words:
        word_id = vocab_map[word]
        doc_word_str = str(doc_id) + ',' + str(word_id)
        if doc_word_str in doc_word_freq:
            doc_word_freq[doc_word_str] += 1
        else:
            doc_word_freq[doc_word_str] = 1

for i in range(n_docs):
    doc_words = shuffled_clean_docs[i]
    words = doc_words.split()
    doc_word_set = set()
    tfidf_vec = []
    for word in words:
        if word in doc_word_set:
            continue
        j = vocab_map[word]
        key = str(i) + ',' + str(j)
        tf = doc_word_freq[key]
        tfidf_row.append(i)
        if i < train_size: 
            row.append(i)
        else:
            row.append(i + vocab_size) 
        tfidf_col.append(j)
        col.append(train_size + j)
        # smooth
        idf = log((1.0 + n_docs) / (1.0+word_doc_freq[vocab[j]])) +1.0
        # weight.append(tf * idf)
        if tfidf_mode=='only_tf':
            tfidf_vec.append(tf)
        else:
            tfidf_vec.append(tf * idf)
        doc_word_set.add(word)
    if len(tfidf_vec)>0:
        weight.extend(tfidf_vec)
        tfidf_weight.extend(tfidf_vec)
    
#%%
'''
Assemble adjacency matrix and dump to files
'''
node_size = vocab_size + corpus_size

adj_list = []
adj_list.append(sp.csr_matrix((weight, (row, col)), shape=(node_size, node_size), dtype=np.float32))
for i,adj in enumerate(adj_list):
    adj_list[i] = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) 
    adj_list[i].setdiag(1.0)

vocab_adj=sp.csr_matrix((vocab_adj_weight, (vocab_adj_row, vocab_adj_col)), shape=(vocab_size, vocab_size), dtype=np.float32)
vocab_adj.setdiag(1.0)

#%%
print('Calculate isomorphic vocab adjacency matrix using doc\'s tf-idf...')
tfidf_all=sp.csr_matrix((tfidf_weight, (tfidf_row, tfidf_col)), shape=(corpus_size, vocab_size), dtype=np.float32)
tfidf_train=tfidf_all[:train_size]
tfidf_valid=tfidf_all[train_size:train_size+valid_size]
tfidf_test=tfidf_all[train_size+valid_size:]
tfidf_X_list=[tfidf_train,tfidf_valid,tfidf_test]
vocab_tfidf=tfidf_all.T.tolil()
for i in range(vocab_size):
    norm=np.linalg.norm(vocab_tfidf.data[i])
    if norm>0:
        vocab_tfidf.data[i]/=norm
vocab_adj_tf=vocab_tfidf.dot(vocab_tfidf.T)

# check
print('Check adjacent matrix...')
for k in range(len(adj_list)):
    count=0
    for i in range(adj_list[k].shape[0]):
        if adj_list[k][i,i]<=0:
            count+=1
            print('No.%d adj, abnomal diagonal found, No.%d'%(k,i))
    if count>0:
        print('No.%d adj, totoal %d zero diagonal found.'%(k,count))

# dump objects
if will_dump_objects:
    print('Dump objects...')

    with open(dump_dir+"/data_%s.labels"%cfg_ds, 'wb') as f:
        pkl.dump([label2idx,idx2label], f)

    with open(dump_dir+"/data_%s.vocab_map"%cfg_ds, 'wb') as f:
        pkl.dump(vocab_map, f)

    with open(dump_dir+"/data_%s.vocab"%cfg_ds, 'wb') as f:
        pkl.dump(vocab, f)

    with open(dump_dir+"/data_%s.adj_list"%cfg_ds, 'wb') as f:
        pkl.dump(adj_list, f)
    with open(dump_dir+"/data_%s.y"%cfg_ds, 'wb') as f:
        pkl.dump(y, f)
    with open(dump_dir+"/data_%s.y_prob"%cfg_ds, 'wb') as f:
        pkl.dump(y_prob, f)
    with open(dump_dir+"/data_%s.train_y"%cfg_ds, 'wb') as f:
        pkl.dump(train_y, f)
    with open(dump_dir+"/data_%s.train_y_prob"%cfg_ds, 'wb') as f:
        pkl.dump(train_y_prob, f)
    with open(dump_dir+"/data_%s.valid_y"%cfg_ds, 'wb') as f:
        pkl.dump(valid_y, f)
    with open(dump_dir+"/data_%s.valid_y_prob"%cfg_ds, 'wb') as f:
        pkl.dump(valid_y_prob, f)
    with open(dump_dir+"/data_%s.test_y"%cfg_ds, 'wb') as f:
        pkl.dump(test_y, f)
    with open(dump_dir+"/data_%s.test_y_prob"%cfg_ds, 'wb') as f:
        pkl.dump(test_y_prob, f)
    with open(dump_dir+"/data_%s.tfidf_list"%cfg_ds, 'wb') as f:
        pkl.dump(tfidf_X_list, f)
    with open(dump_dir+"/data_%s.vocab_adj_pmi"%(cfg_ds), 'wb') as f:
        pkl.dump(vocab_adj, f)
    with open(dump_dir+"/data_%s.vocab_adj_tf"%(cfg_ds), 'wb') as f:
        pkl.dump(vocab_adj_tf, f)
    with open(dump_dir+"/data_%s.shuffled_clean_docs"%cfg_ds, 'wb') as f:
        pkl.dump(shuffled_clean_docs, f)
        
print('Data prepared, spend %.2f s'%(time.time()-start))
