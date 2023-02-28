# -*- coding: utf-8 -*-

# @author Zhibin.LU
# @website: https://github.com/Louis-udm

"""VGCN-BERT related models"""

import math

import torch
import torch.nn as nn
import torch.nn.init as init

# for huggingface transformers 0.6.2;
from pytorch_pretrained_bert.modeling import (
    BertEmbeddings,
    BertEncoder,
    BertModel,
    BertPooler,
)


class VocabGraphConvolution(nn.Module):
    """Vocabulary GCN module.

    Params:
        `voc_dim`: The size of vocabulary graph
        `num_adj`: The number of the adjacency matrix of Vocabulary graph
        `hid_dim`: The hidden dimension after XAW
        `out_dim`: The output dimension after Relu(XAW)W
        `dropout_rate`: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.

    Inputs:
        `vocab_adj_list`: The list of the adjacency matrix
        `X_dv`: the feature of mini batch document, can be TF-IDF (batch, vocab), or word embedding (batch, word_embedding_dim, vocab)

    Outputs:
        The graph embedding representation, dimension (batch, `out_dim`) or (batch, word_embedding_dim, `out_dim`)

    """

    def __init__(self, voc_dim, num_adj, hid_dim, out_dim, dropout_rate=0.2):
        super().__init__()
        self.voc_dim = voc_dim
        self.num_adj = num_adj
        self.hid_dim = hid_dim
        self.out_dim = out_dim

        for i in range(self.num_adj):
            setattr(
                self, "W%d_vh" % i, nn.Parameter(torch.randn(voc_dim, hid_dim))
            )

        self.fc_hc = nn.Linear(hid_dim, out_dim)
        self.act_func = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        self.reset_parameters()

    def reset_parameters(self):
        for n, p in self.named_parameters():
            if (
                n.startswith("W")
                or n.startswith("a")
                or n in ("W", "a", "dense")
            ):
                init.kaiming_uniform_(p, a=math.sqrt(5))

    def forward(self, vocab_adj_list, X_dv, add_linear_mapping_term=False):
        for i in range(self.num_adj):
            H_vh = vocab_adj_list[i].mm(getattr(self, "W%d_vh" % i))
            # H_vh=self.dropout(F.elu(H_vh))
            H_vh = self.dropout(H_vh)
            H_dh = X_dv.matmul(H_vh)

            if add_linear_mapping_term:
                H_linear = X_dv.matmul(getattr(self, "W%d_vh" % i))
                H_linear = self.dropout(H_linear)
                H_dh += H_linear

            if i == 0:
                fused_H = H_dh
            else:
                fused_H += H_dh

        out = self.fc_hc(fused_H)
        return out


class Pretrain_VGCN(nn.Module):
    """Pretrain_VGCN can pre-train the weights of VGCN moduel in the VGCN-BERT. It is also a pure VGCN classification model for word embedding input.

    Params:
        `word_emb`: The instance of word embedding module, we use BERT word embedding module in general.
        `word_emb_dim`: The dimension size of word embedding.
        `gcn_adj_dim`: The size of vocabulary graph
        `gcn_adj_num`: The number of the adjacency matrix of Vocabulary graph
        `hid_dim`: The hidden dimension after XAW
        `gcn_embedding_dim`: The output dimension after Relu(XAW)W
        `num_labels`: the number of classes for the classifier. Default = 2.
        `dropout_rate`: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.

    Inputs:
        `vocab_adj_list`: The list of the adjacency matrix
        `gcn_swop_eye`: The transform matrix for transform the token sequence (sentence) to the Vocabulary order (BoW order)
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary. Items in the batch should begin with the special "CLS" token. (see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).

    Outputs:
        The graph embedding representation, dimension (batch, `out_dim`) or (batch, word_embedding_dim, `out_dim`)

    """

    def __init__(
        self,
        word_emb,
        word_emb_dim,
        gcn_adj_dim,
        gcn_adj_num,
        gcn_embedding_dim,
        num_labels,
        dropout_rate=0.2,
    ):
        super(Pretrain_VGCN, self).__init__()
        self.act_func = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.word_emb = word_emb
        self.vocab_gcn = VocabGraphConvolution(
            gcn_adj_dim, gcn_adj_num, 128, gcn_embedding_dim
        )  # 192/256
        self.classifier = nn.Linear(
            gcn_embedding_dim * word_emb_dim, num_labels
        )

    def forward(
        self,
        vocab_adj_list,
        gcn_swop_eye,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
    ):
        words_embeddings = self.word_emb(input_ids)
        vocab_input = gcn_swop_eye.matmul(words_embeddings).transpose(1, 2)
        gcn_vocab_out = self.vocab_gcn(vocab_adj_list, vocab_input).transpose(
            1, 2
        )
        gcn_vocab_out = self.dropout(self.act_func(gcn_vocab_out))
        out = self.classifier(gcn_vocab_out.flatten(start_dim=1))
        return out


class VGCNBertEmbeddings(BertEmbeddings):
    """Construct the embeddings from word, VGCN graph, position and token_type embeddings.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model
        `gcn_adj_dim`: The size of vocabulary graph
        `gcn_adj_num`: The number of the adjacency matrix of Vocabulary graph
        `gcn_embedding_dim`: The output dimension after VGCN

    Inputs:
        `vocab_adj_list`: The list of the adjacency matrix
        `gcn_swop_eye`: The transform matrix for transform the token sequence (sentence) to the Vocabulary order (BoW order)
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary. Items in the batch should begin with the special "CLS" token. (see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.

    Outputs:
        the word embeddings fused by VGCN embedding, position embedding and token_type embeddings.

    """

    def __init__(self, config, gcn_adj_dim, gcn_adj_num, gcn_embedding_dim):
        super().__init__(config)
        assert gcn_embedding_dim >= 0
        self.gcn_embedding_dim = gcn_embedding_dim
        self.vocab_gcn = VocabGraphConvolution(
            gcn_adj_dim, gcn_adj_num, 128, gcn_embedding_dim
        )  # 192/256

    def forward(
        self,
        vocab_adj_list,
        gcn_swop_eye,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
    ):
        words_embeddings = self.word_embeddings(input_ids)
        vocab_input = gcn_swop_eye.matmul(words_embeddings).transpose(1, 2)

        if self.gcn_embedding_dim > 0:
            gcn_vocab_out = self.vocab_gcn(vocab_adj_list, vocab_input)

            gcn_words_embeddings = words_embeddings.clone()
            for i in range(self.gcn_embedding_dim):
                tmp_pos = (
                    attention_mask.sum(-1) - 2 - self.gcn_embedding_dim + 1 + i
                ) + torch.arange(0, input_ids.shape[0]).to(
                    input_ids.device
                ) * input_ids.shape[
                    1
                ]
                gcn_words_embeddings.flatten(start_dim=0, end_dim=1)[
                    tmp_pos, :
                ] = gcn_vocab_out[:, :, i]

        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        if self.gcn_embedding_dim > 0:
            embeddings = (
                gcn_words_embeddings
                + position_embeddings
                + token_type_embeddings
            )
        else:
            embeddings = (
                words_embeddings + position_embeddings + token_type_embeddings
            )

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class VGCNBertModel(BertModel):
    """VGCN-BERT model for text classification. It inherits from Huggingface's BertModel.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model
        `gcn_adj_dim`: The size of vocabulary graph
        `gcn_adj_num`: The number of the adjacency matrix of Vocabulary graph
        `gcn_embedding_dim`: The output dimension after VGCN
        `num_labels`: the number of classes for the classifier. Default = 2.
        `output_attentions`: If True, also output attentions weights computed by the model at each layer. Default: False
        `keep_multihead_output`: If True, saves output of the multi-head attention module with its gradient.
            This can be used to compute head importance metrics. Default: False

    Inputs:
        `vocab_adj_list`: The list of the adjacency matrix
        `gcn_swop_eye`: The transform matrix for transform the token sequence (sentence) to the Vocabulary order (BoW order)
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary. Items in the batch should begin with the special "CLS" token. (see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].
        `head_mask`: an optional torch.Tensor of shape [num_heads] or [num_layers, num_heads] with indices between 0 and 1.
            It's a mask to be used to nullify some heads of the transformer. 1.0 => head is fully masked, 0.0 => head is not masked.

    Outputs:
        Outputs the classification logits of shape [batch_size, num_labels].

    """

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
            config,
            # add_pooling_layer=True,
            output_attentions,
            keep_multihead_output,
        )
        self.embeddings = VGCNBertEmbeddings(
            config, gcn_adj_dim, gcn_adj_num, gcn_embedding_dim
        )
        self.encoder = BertEncoder(
            config,
        )
        self.pooler = BertPooler(config)  # if add_pooling_layer else None
        self.num_labels = num_labels
        # self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.will_collect_cls_states = False
        self.all_cls_states = []
        self.output_attentions = output_attentions

        self.apply(self.init_bert_weights)
        # super().post_init()

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
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        embedding_output = self.embeddings(
            vocab_adj_list,
            gcn_swop_eye,
            input_ids,
            token_type_ids,
            attention_mask,
        )

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = (
                    head_mask.unsqueeze(0)
                    .unsqueeze(0)
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                )
                head_mask = head_mask.expand_as(
                    self.config.num_hidden_layers, -1, -1, -1, -1
                )
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        if self.output_attentions:
            output_all_encoded_layers = True
        encoded_layers = self.encoder(
            embedding_output,
            extended_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
            head_mask=head_mask,
        )
        if self.output_attentions:
            all_attentions, encoded_layers = encoded_layers

        pooled_output = self.pooler(encoded_layers[-1])
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if self.output_attentions:
            return all_attentions, logits

        return logits
