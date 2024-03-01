# -*- coding: utf-8 -*-

# @author Zhibin.LU
# @website: https://github.com/Louis-udm

"""Train vgcn_bert model"""

import argparse
import gc
import os
import pickle as pkl
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# use pytorch_pretrained_bert.modeling for huggingface transformers 0.6.2
from pytorch_pretrained_bert.optimization import BertAdam  # , warmup_linear
from pytorch_pretrained_bert.tokenization import BertTokenizer

# from tqdm import tqdm, trange
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader

from vgcn_bert.env_config import env_config
from vgcn_bert.models.vgcn_bert import VGCNBertModel
from vgcn_bert.utils import *

# from transformers import BertTokenizer,AdamW


random.seed(env_config.GLOBAL_SEED)
np.random.seed(env_config.GLOBAL_SEED)
torch.manual_seed(env_config.GLOBAL_SEED)

cuda_yes = torch.cuda.is_available()
if cuda_yes:
    torch.cuda.manual_seed_all(env_config.GLOBAL_SEED)
device = torch.device("cuda:0" if cuda_yes else "cpu")


"""
Configuration
"""

parser = argparse.ArgumentParser()
parser.add_argument("--ds", type=str, default="cola")
parser.add_argument("--load", type=int, default=0)
parser.add_argument("--sw", type=int, default="0")
parser.add_argument("--dim", type=int, default="16")
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--l2", type=float, default=0.01)
parser.add_argument("--model", type=str, default="VGCN_BERT")
parser.add_argument("--validate_program", action="store_true")
args = parser.parse_args()
args.ds = args.ds
cfg_model_type = args.model
cfg_stop_words = True if args.sw == 1 else False
will_train_mode_from_checkpoint = True if args.load == 1 else False
gcn_embedding_dim = args.dim
learning_rate0 = args.lr
l2_decay = args.l2

dataset_list = {"sst", "cola"}
# hate: 10k, mr: 6753, sst: 7792, r8: 5211

total_train_epochs = 9
dropout_rate = 0.2  # 0.5 # Dropout rate (1 - keep probability).
if args.ds == "sst":
    batch_size = 16  # 12
    learning_rate0 = 1e-5  # 2e-5
    # l2_decay = 0.001
    l2_decay = 0.01  # default
elif args.ds == "cola":
    batch_size = 16  # 12
    learning_rate0 = 8e-6  # 2e-5
    l2_decay = 0.001

MAX_SEQ_LENGTH = 200 + gcn_embedding_dim
gradient_accumulation_steps = 1

# bert_model_scale='bert-large-uncased'
bert_model_scale = "bert-base-uncased"
if env_config.TRANSFORMERS_OFFLINE == 1:
    bert_model_scale = os.path.join(
        env_config.HUGGING_LOCAL_MODEL_FILES_PATH,
        f"hf-maintainers_{bert_model_scale}",
    )

do_lower_case = True
warmup_proportion = 0.1

data_dir = f"data/preprocessed/{args.ds}"
output_dir = "output/"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

perform_metrics_str = ["weighted avg", "f1-score"]

# cfg_add_linear_mapping_term=False
cfg_vocab_adj = "pmi"
# cfg_vocab_adj='all'
# cfg_vocab_adj='tf'
cfg_adj_npmi_threshold = 0.2
cfg_adj_tf_threshold = 0
classifier_act_func = nn.ReLU()

resample_train_set = False  # if mse and resample, then do resample
do_softmax_before_mse = True
cfg_loss_criterion = "cle"
model_file_4save = (
    f"{cfg_model_type}{gcn_embedding_dim}_model_{args.ds}_{cfg_loss_criterion}"
    f"_sw{int(cfg_stop_words)}.pt"
)

if args.validate_program:
    total_train_epochs = 1

print(cfg_model_type + " Start at:", time.asctime())
print(
    "\n----- Configure -----",
    f"\n  args.ds: {args.ds}",
    f"\n  stop_words: {cfg_stop_words}",
    # '\n  Vocab GCN_hidden_dim: 768 -> 1152 -> 768',
    f"\n  Vocab GCN_hidden_dim: vocab_size -> 128 -> {str(gcn_embedding_dim)}",
    f"\n  Learning_rate0: {learning_rate0}" f"\n  weight_decay: {l2_decay}",
    f"\n  Loss_criterion {cfg_loss_criterion}"
    f"\n  softmax_before_mse: {do_softmax_before_mse}",
    f"\n  Dropout: {dropout_rate}"
    f"\n  Run_adj: {cfg_vocab_adj}"
    f"\n  gcn_act_func: Relu",
    f"\n  MAX_SEQ_LENGTH: {MAX_SEQ_LENGTH}",  #'valid_data_taux',valid_data_taux
    f"\n  perform_metrics_str: {perform_metrics_str}",
    f"\n  model_file_4save: {model_file_4save}",
    f"\n  validate_program: {args.validate_program}",
)


"""
Prepare data set
Load vocabulary adjacent matrix
"""
print("\n----- Prepare data set -----")
print(
    f"  Load/shuffle/seperate {args.ds} dataset, and vocabulary graph adjacent matrix"
)

objects = []
names = [
    "labels",
    "train_y",
    "train_y_prob",
    "valid_y",
    "valid_y_prob",
    "test_y",
    "test_y_prob",
    "shuffled_clean_docs",
    "vocab_adj_tf",
    "vocab_adj_pmi",
    "vocab_map",
]
for i in range(len(names)):
    datafile = "./" + data_dir + "/data_%s.%s" % (args.ds, names[i])
    with open(datafile, "rb") as f:
        objects.append(pkl.load(f, encoding="latin1"))
(
    lables_list,
    train_y,
    train_y_prob,
    valid_y,
    valid_y_prob,
    test_y,
    test_y_prob,
    shuffled_clean_docs,
    gcn_vocab_adj_tf,
    gcn_vocab_adj,
    gcn_vocab_map,
) = tuple(objects)

label2idx = lables_list[0]
idx2label = lables_list[1]

y = np.hstack((train_y, valid_y, test_y))
y_prob = np.vstack((train_y_prob, valid_y_prob, test_y_prob))

examples = []
for i, ts in enumerate(shuffled_clean_docs):
    ex = InputExample(i, ts.strip(), confidence=y_prob[i], label=y[i])
    examples.append(ex)

num_classes = len(label2idx)
gcn_vocab_size = len(gcn_vocab_map)
train_size = len(train_y)
valid_size = len(valid_y)
test_size = len(test_y)

indexs = np.arange(0, len(examples))
train_examples = [examples[i] for i in indexs[:train_size]]
valid_examples = [
    examples[i] for i in indexs[train_size : train_size + valid_size]
]
test_examples = [
    examples[i]
    for i in indexs[
        train_size + valid_size : train_size + valid_size + test_size
    ]
]

if cfg_adj_tf_threshold > 0:
    gcn_vocab_adj_tf.data *= gcn_vocab_adj_tf.data > cfg_adj_tf_threshold
    gcn_vocab_adj_tf.eliminate_zeros()
if cfg_adj_npmi_threshold > 0:
    gcn_vocab_adj.data *= gcn_vocab_adj.data > cfg_adj_npmi_threshold
    gcn_vocab_adj.eliminate_zeros()

if cfg_vocab_adj == "pmi":
    gcn_vocab_adj_list = [gcn_vocab_adj]
elif cfg_vocab_adj == "tf":
    gcn_vocab_adj_list = [gcn_vocab_adj_tf]
elif cfg_vocab_adj == "all":
    gcn_vocab_adj_list = [gcn_vocab_adj_tf, gcn_vocab_adj]

norm_gcn_vocab_adj_list = []
for i in range(len(gcn_vocab_adj_list)):
    adj = gcn_vocab_adj_list[i]  # .tocsr() #(lr是用非norm时的1/10)
    print(
        "  Zero ratio(?>66%%) for vocab adj %dth: %.8f"
        % (i, 100 * (1 - adj.count_nonzero() / (adj.shape[0] * adj.shape[1])))
    )
    adj = normalize_adj(adj)
    norm_gcn_vocab_adj_list.append(sparse_scipy2torch(adj.tocoo()).to(device))
gcn_adj_list = norm_gcn_vocab_adj_list


del gcn_vocab_adj_tf, gcn_vocab_adj, gcn_vocab_adj_list
gc.collect()

train_classes_num, train_classes_weight = get_class_count_and_weight(
    train_y, len(label2idx)
)
loss_weight = torch.tensor(train_classes_weight, dtype=torch.float).to(device)

tokenizer = BertTokenizer.from_pretrained(
    bert_model_scale, do_lower_case=do_lower_case
)


def get_pytorch_dataloader(
    examples,
    tokenizer,
    batch_size,
    shuffle_choice,
    classes_weight=None,
    total_resample_size=-1,
):
    ds = CorpusDataset(
        examples, tokenizer, gcn_vocab_map, MAX_SEQ_LENGTH, gcn_embedding_dim
    )
    if shuffle_choice == 0:  # shuffle==False
        return DataLoader(
            dataset=ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=ds.pad,
        )
    elif shuffle_choice == 1:  # shuffle==True
        return DataLoader(
            dataset=ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=ds.pad,
        )
    elif shuffle_choice == 2:  # weighted resampled
        assert classes_weight is not None
        assert total_resample_size > 0
        weights = [
            classes_weight[0]
            if label == 0
            else classes_weight[1]
            if label == 1
            else classes_weight[2]
            for _, _, _, _, label in dataset
        ]
        sampler = WeightedRandomSampler(
            weights, num_samples=total_resample_size, replacement=True
        )
        return DataLoader(
            dataset=ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=4,
            collate_fn=ds.pad,
        )


# ds size=1 for validating the program
if args.validate_program:
    train_examples = [train_examples[0]]
    valid_examples = [valid_examples[0]]
    test_examples = [test_examples[0]]

train_dataloader = get_pytorch_dataloader(
    train_examples, tokenizer, batch_size, shuffle_choice=0
)
valid_dataloader = get_pytorch_dataloader(
    valid_examples, tokenizer, batch_size, shuffle_choice=0
)
test_dataloader = get_pytorch_dataloader(
    test_examples, tokenizer, batch_size, shuffle_choice=0
)


# total_train_steps = int(len(train_examples) / batch_size / gradient_accumulation_steps * total_train_epochs)
total_train_steps = int(
    len(train_dataloader) / gradient_accumulation_steps * total_train_epochs
)

print("  Train_classes count:", train_classes_num)
print(
    f"  Num examples for train = {len(train_examples)}",
    f", after weight sample: {len(train_dataloader) * batch_size}",
)
print("  Num examples for validate = %d" % len(valid_examples))
print("  Batch size = %d" % batch_size)
print("  Num steps = %d" % total_train_steps)


"""
Train vgcn_bert model
"""


def predict(model, examples, tokenizer, batch_size):
    dataloader = get_pytorch_dataloader(
        examples, tokenizer, batch_size, shuffle_choice=0
    )
    predict_out = []
    confidence_out = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch = tuple(t.to(device) for t in batch)
            (
                input_ids,
                input_mask,
                segment_ids,
                _,
                label_ids,
                gcn_swop_eye,
            ) = batch
            score_out = model(
                gcn_adj_list, gcn_swop_eye, input_ids, segment_ids, input_mask
            )
            if cfg_loss_criterion == "mse" and do_softmax_before_mse:
                score_out = torch.nn.functional.softmax(score_out, dim=-1)
            predict_out.extend(score_out.max(1)[1].tolist())
            confidence_out.extend(score_out.max(1)[0].tolist())

    return np.array(predict_out).reshape(-1), np.array(confidence_out).reshape(
        -1
    )


def evaluate(
    model, gcn_adj_list, predict_dataloader, batch_size, epoch_th, dataset_name
):
    # print("***** Running prediction *****")
    model.eval()
    predict_out = []
    all_label_ids = []
    ev_loss = 0
    total = 0
    correct = 0
    start = time.time()
    with torch.no_grad():
        for batch in predict_dataloader:
            batch = tuple(t.to(device) for t in batch)
            (
                input_ids,
                input_mask,
                segment_ids,
                y_prob,
                label_ids,
                gcn_swop_eye,
            ) = batch
            # the parameter label_ids is None, model return the prediction score
            logits = model(
                gcn_adj_list, gcn_swop_eye, input_ids, segment_ids, input_mask
            )

            if cfg_loss_criterion == "mse":
                if do_softmax_before_mse:
                    logits = F.softmax(logits, -1)
                loss = F.mse_loss(logits, y_prob)
            else:
                if loss_weight is None:
                    loss = F.cross_entropy(
                        logits.view(-1, num_classes), label_ids
                    )
                else:
                    loss = F.cross_entropy(
                        logits.view(-1, num_classes), label_ids
                    )
            ev_loss += loss.item()

            _, predicted = torch.max(logits, -1)
            predict_out.extend(predicted.tolist())
            all_label_ids.extend(label_ids.tolist())
            eval_accuracy = predicted.eq(label_ids).sum().item()
            total += len(label_ids)
            correct += eval_accuracy

        f1_metrics = f1_score(
            np.array(all_label_ids).reshape(-1),
            np.array(predict_out).reshape(-1),
            average="weighted",
        )
        print(
            "Report:\n"
            + classification_report(
                np.array(all_label_ids).reshape(-1),
                np.array(predict_out).reshape(-1),
                digits=4,
            )
        )

    ev_acc = correct / total
    end = time.time()
    print(
        "Epoch : %d, %s: %.3f Acc : %.3f on %s, Spend:%.3f minutes for evaluation"
        % (
            epoch_th,
            " ".join(perform_metrics_str),
            100 * f1_metrics,
            100.0 * ev_acc,
            dataset_name,
            (end - start) / 60.0,
        )
    )
    print("--------------------------------------------------------------")
    return ev_loss, ev_acc, f1_metrics


print("\n----- Running training -----")
if will_train_mode_from_checkpoint and os.path.exists(
    os.path.join(output_dir, model_file_4save)
):
    checkpoint = torch.load(
        os.path.join(output_dir, model_file_4save), map_location="cpu"
    )
    if "step" in checkpoint:
        prev_save_step = checkpoint["step"]
        start_epoch = checkpoint["epoch"]
    else:
        prev_save_step = -1
        start_epoch = checkpoint["epoch"] + 1
    valid_acc_prev = checkpoint["valid_acc"]
    perform_metrics_prev = checkpoint["perform_metrics"]
    model = VGCNBertModel.from_pretrained(
        bert_model_scale,
        state_dict=checkpoint["model_state"],
        gcn_adj_dim=gcn_vocab_size,
        gcn_adj_num=len(gcn_adj_list),
        gcn_embedding_dim=gcn_embedding_dim,
        num_labels=len(label2idx),
    )
    pretrained_dict = checkpoint["model_state"]
    net_state_dict = model.state_dict()
    pretrained_dict_selected = {
        k: v for k, v in pretrained_dict.items() if k in net_state_dict
    }
    net_state_dict.update(pretrained_dict_selected)
    model.load_state_dict(net_state_dict)
    print(
        f"Loaded the pretrain model: {model_file_4save}",
        f", epoch: {checkpoint['epoch']}",
        f"step: {prev_save_step}",
        f"valid acc: {checkpoint['valid_acc']}",
        f"{' '.join(perform_metrics_str)}_valid: {checkpoint['perform_metrics']}",
    )

else:
    start_epoch = 0
    valid_acc_prev = 0
    perform_metrics_prev = 0
    model = VGCNBertModel.from_pretrained(
        bert_model_scale,
        gcn_adj_dim=gcn_vocab_size,
        gcn_adj_num=len(gcn_adj_list),
        gcn_embedding_dim=gcn_embedding_dim,
        num_labels=len(label2idx),
    )
    prev_save_step = -1

model.to(device)

optimizer = BertAdam(
    model.parameters(),
    lr=learning_rate0,
    warmup=warmup_proportion,
    t_total=total_train_steps,
    weight_decay=l2_decay,
)

train_start = time.time()
global_step_th = int(
    len(train_examples)
    / batch_size
    / gradient_accumulation_steps
    * start_epoch
)

all_loss_list = {"train": [], "valid": [], "test": []}
all_f1_list = {"train": [], "valid": [], "test": []}
for epoch in range(start_epoch, total_train_epochs):
    tr_loss = 0
    ep_train_start = time.time()
    model.train()
    optimizer.zero_grad()
    # for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
    for step, batch in enumerate(train_dataloader):
        if prev_save_step > -1:
            if step <= prev_save_step:
                continue
        if prev_save_step > -1:
            prev_save_step = -1
        batch = tuple(t.to(device) for t in batch)
        (
            input_ids,
            input_mask,
            segment_ids,
            y_prob,
            label_ids,
            gcn_swop_eye,
        ) = batch

        logits = model(
            gcn_adj_list, gcn_swop_eye, input_ids, segment_ids, input_mask
        )

        if cfg_loss_criterion == "mse":
            if do_softmax_before_mse:
                logits = F.softmax(logits, -1)
            loss = F.mse_loss(logits, y_prob)
        else:
            if loss_weight is None:
                loss = F.cross_entropy(logits, label_ids)
            else:
                loss = F.cross_entropy(
                    logits.view(-1, num_classes), label_ids, loss_weight
                )

        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
        loss.backward()

        tr_loss += loss.item()
        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step_th += 1
        if step % 40 == 0:
            print(
                "Epoch:{}-{}/{}, Train {} Loss: {}, Cumulated time: {}m ".format(
                    epoch,
                    step,
                    len(train_dataloader),
                    cfg_loss_criterion,
                    loss.item(),
                    (time.time() - train_start) / 60.0,
                )
            )

    print("--------------------------------------------------------------")
    valid_loss, valid_acc, perform_metrics = evaluate(
        model, gcn_adj_list, valid_dataloader, batch_size, epoch, "Valid_set"
    )
    test_loss, _, test_f1 = evaluate(
        model, gcn_adj_list, test_dataloader, batch_size, epoch, "Test_set"
    )
    all_loss_list["train"].append(tr_loss)
    all_loss_list["valid"].append(valid_loss)
    all_loss_list["test"].append(test_loss)
    all_f1_list["valid"].append(perform_metrics)
    all_f1_list["test"].append(test_f1)
    print(
        "Epoch:{} completed, Total Train Loss:{}, Valid Loss:{}, Spend {}m ".format(
            epoch, tr_loss, valid_loss, (time.time() - train_start) / 60.0
        )
    )
    # Save a checkpoint
    # if valid_acc > valid_acc_prev:
    if perform_metrics > perform_metrics_prev:
        to_save = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "valid_acc": valid_acc,
            "lower_case": do_lower_case,
            "perform_metrics": perform_metrics,
        }
        torch.save(to_save, os.path.join(output_dir, model_file_4save))
        # valid_acc_prev = valid_acc
        perform_metrics_prev = perform_metrics
        test_f1_when_valid_best = test_f1
        # train_f1_when_valid_best=tr_f1
        valid_f1_best_epoch = epoch

print(
    "\n**Optimization Finished!,Total spend:",
    (time.time() - train_start) / 60.0,
)
print(
    "**Valid weighted F1: %.3f at %d epoch."
    % (100 * perform_metrics_prev, valid_f1_best_epoch)
)
print(
    "**Test weighted F1 when valid best: %.3f"
    % (100 * test_f1_when_valid_best)
)
