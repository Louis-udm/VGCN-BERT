# -*- coding: utf-8 -*-

# @author Zhibin.LU
# @website: https://github.com/Louis-udm

import re

import numpy as np
import scipy.sparse as sp
import torch
from nltk.tokenize import TweetTokenizer
from torch.utils import data
from torch.utils.data import (
    DataLoader,
    Dataset,
    RandomSampler,
    SequentialSampler,
    TensorDataset,
    WeightedRandomSampler,
)
from torch.utils.data.distributed import DistributedSampler

"""
General functions
"""


def del_http_user_tokenize(tweet):
    # delete [ \t\n\r\f\v]
    space_pattern = r"\s+"
    url_regex = (
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|"
        r"[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    )
    mention_regex = r"@[\w\-]+"
    tweet = re.sub(space_pattern, " ", tweet)
    tweet = re.sub(url_regex, "", tweet)
    tweet = re.sub(mention_regex, "", tweet)
    return tweet


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    # string = " ".join(re.split("[^a-zA-Z]", string.lower())).strip()
    string = re.sub(r"\'s", " 's", string)
    string = re.sub(r"\'ve", " 've", string)
    string = re.sub(r"n\'t", " n't", string)
    string = re.sub(r"\'re", " 're", string)
    string = re.sub(r"\'d", " 'd", string)
    string = re.sub(r"\'ll", " 'll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def clean_tweet_tokenize(string):
    tknzr = TweetTokenizer(
        reduce_len=True, preserve_case=False, strip_handles=False
    )
    tokens = tknzr.tokenize(string.lower())
    return " ".join(tokens).strip()


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    # adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))  # D-degree matrix
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


def sparse_scipy2torch(coo_sparse):
    # coo_sparse=coo_sparse.tocoo()
    i = torch.LongTensor(np.vstack((coo_sparse.row, coo_sparse.col)))
    v = torch.from_numpy(coo_sparse.data)
    return torch.sparse.FloatTensor(i, v, torch.Size(coo_sparse.shape))


def get_class_count_and_weight(y, n_classes):
    classes_count = []
    weight = []
    for i in range(n_classes):
        count = np.sum(y == i)
        classes_count.append(count)
        weight.append(len(y) / (n_classes * count))
    return classes_count, weight


"""
Functions and Classes for read and organize data set
"""


class InputExample(object):
    """
    A single training/test example for sentence classifier.
    """

    def __init__(self, guid, text_a, text_b=None, confidence=None, label=None):
        """
        Constructs a InputExample.

        Args:
            guid: Unique id for the example(a sentence or a pair of sentences).
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        # string of the sentence,example: [EU, rejects, German, call, to, boycott, British, lamb .]
        self.text_a = text_a
        self.text_b = text_b
        # the label(class) for the sentence
        self.confidence = confidence
        self.label = label


class InputFeatures(object):
    """
    A single set of features of data.
    result of convert_examples_to_features(InputExample)

    please refer to bert.modeling
    """

    def __init__(
        self,
        guid,
        tokens,
        input_ids,
        gcn_vocab_ids,
        input_mask,
        segment_ids,
        confidence,
        label_id,
    ):
        self.guid = guid
        self.tokens = tokens
        self.input_ids = input_ids
        self.gcn_vocab_ids = gcn_vocab_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.confidence = confidence
        self.label_id = label_id


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """
    Truncates a sequence pair in place to the maximum length.
    """

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def example2feature(
    example, tokenizer, gcn_vocab_map, max_seq_len, gcn_embedding_dim
):

    # tokens_a = tokenizer.tokenize(example.text_a)
    # do not need use bert.tokenizer again, because be used at prepare_data.py
    tokens_a = example.text_a.split()
    assert example.text_b == None
    # Account for [CLS] and [SEP] with "- 2" ,# -1 for gcn_words_convoled
    if len(tokens_a) > max_seq_len - 1 - gcn_embedding_dim:
        print(
            "GUID: %d, Sentence is too long: %d"
            % (example.guid, len(tokens_a))
        )
        tokens_a = tokens_a[: (max_seq_len - 1 - gcn_embedding_dim)]

    gcn_vocab_ids = []
    for w in tokens_a:
        gcn_vocab_ids.append(gcn_vocab_map[w])

    tokens = (
        ["[CLS]"] + tokens_a + ["[SEP]" for i in range(gcn_embedding_dim + 1)]
    )
    segment_ids = [0] * len(tokens)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    feat = InputFeatures(
        guid=example.guid,
        tokens=tokens,
        input_ids=input_ids,
        gcn_vocab_ids=gcn_vocab_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        # label_id=label2idx[example.label]
        confidence=example.confidence,
        label_id=example.label,
    )
    return feat


class CorpusDataset(Dataset):
    def __init__(
        self,
        examples,
        tokenizer,
        gcn_vocab_map,
        max_seq_len,
        gcn_embedding_dim,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.gcn_embedding_dim = gcn_embedding_dim
        self.gcn_vocab_map = gcn_vocab_map

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        feat = example2feature(
            self.examples[idx],
            self.tokenizer,
            self.gcn_vocab_map,
            self.max_seq_len,
            self.gcn_embedding_dim,
        )
        return (
            feat.input_ids,
            feat.input_mask,
            feat.segment_ids,
            feat.confidence,
            feat.label_id,
            feat.gcn_vocab_ids,
        )

    # @classmethod
    # def pad(cls,batch):
    def pad(self, batch):
        gcn_vocab_size = len(self.gcn_vocab_map)
        seqlen_list = [len(sample[0]) for sample in batch]
        maxlen = np.array(seqlen_list).max()

        f_collect = lambda x: [sample[x] for sample in batch]
        f_pad = lambda x, seqlen: [
            sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch
        ]
        # filliing with -1, for indicate the position of this pad is not in gcn_vocab_list. then for generate the transform order tensor and delete this column.
        # first -1 correspond[CLS]
        f_pad2 = lambda x, seqlen: [
            [-1] + sample[x] + [-1] * (seqlen - len(sample[x]) - 1)
            for sample in batch
        ]

        batch_input_ids = torch.tensor(f_pad(0, maxlen), dtype=torch.long)
        batch_input_mask = torch.tensor(f_pad(1, maxlen), dtype=torch.long)
        batch_segment_ids = torch.tensor(f_pad(2, maxlen), dtype=torch.long)
        batch_confidences = torch.tensor(f_collect(3), dtype=torch.float)
        batch_label_ids = torch.tensor(f_collect(4), dtype=torch.long)
        batch_gcn_vocab_ids_paded = np.array(f_pad2(5, maxlen)).reshape(-1)
        # generate eye matrix according to gcn_vocab_size+1, the 1 is for f_pad2 filling -1, then change to the row with all 0 value.
        batch_gcn_swop_eye = torch.eye(gcn_vocab_size + 1)[
            batch_gcn_vocab_ids_paded
        ][:, :-1]
        # This tensor is for transform batch_embedding_tensor to gcn_vocab order
        # -1 is seq_len. usage: batch_gcn_swop_eye.matmul(batch_seq_embedding)
        batch_gcn_swop_eye = batch_gcn_swop_eye.view(
            len(batch), -1, gcn_vocab_size
        ).transpose(1, 2)

        return (
            batch_input_ids,
            batch_input_mask,
            batch_segment_ids,
            batch_confidences,
            batch_label_ids,
            batch_gcn_swop_eye,
        )
