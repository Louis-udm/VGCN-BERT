# -*- coding: utf-8 -*-

# @author Zhibin.LU
# @website: https://github.com/Louis-udm

"""Vanilla combination of VGCN and BERT models"""

import torch
import torch.nn as nn

# for huggingface transformers 0.6.2
from pytorch_pretrained_bert.modeling import BertForSequenceClassification

from .vgcn_bert import VocabGraphConvolution


class VanillaVGCNBert(BertForSequenceClassification):
    def __init__(
        self,
        config,
        gcn_adj_dim,
        gcn_adj_num,
        gcn_embedding_dim,
        num_labels,
        output_attentions=False,
        keep_multihead_output=False,
    ):
        super().__init__(
            config, num_labels, output_attentions=output_attentions
        )

        self.will_collect_cls_states = False
        self.all_cls_states = []
        self.output_attentions = output_attentions
        self.vocab_gcn = VocabGraphConvolution(
            gcn_adj_dim, gcn_adj_num, 128, gcn_embedding_dim
        )  # 192/256
        self.classifier = nn.Linear((gcn_embedding_dim + 1) * 768, num_labels)

    def forward(
        self,
        vocab_adj_list,
        gcn_swop_eye,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        output_all_encoded_layers=False,
        head_mask=None,
    ):

        words_embeddings = self.bert.embeddings.word_embeddings(input_ids)
        vocab_input = gcn_swop_eye.matmul(words_embeddings).transpose(1, 2)
        gcn_vocab_out = self.vocab_gcn(vocab_adj_list, vocab_input).transpose(
            1, 2
        )

        if self.output_attentions:
            all_attentions, _, pooled_output = self.bert(
                input_ids,
                token_type_ids,
                attention_mask,
                output_all_encoded_layers=True,
            )
        else:
            _, pooled_output = self.bert(
                input_ids,
                token_type_ids,
                attention_mask,
                output_all_encoded_layers=False,
            )

        if self.will_collect_cls_states:
            self.all_cls_states.append(pooled_output.cpu())
        cat_out = torch.cat((gcn_vocab_out.squeeze(1), pooled_output), dim=1)

        cat_out = self.dropout(cat_out)
        score_out = self.classifier(cat_out)

        return score_out
