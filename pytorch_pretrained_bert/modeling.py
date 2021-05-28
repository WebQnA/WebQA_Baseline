# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Changes have been made over the original file
# https://github.com/huggingface/pytorch-transformers/blob/v0.4.0/pytorch_pretrained_bert/modeling.py

"""PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import json
import math, time
import logging
import tarfile
import tempfile
import shutil
import numpy as np
import pickle

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F

from .file_utils import cached_path
from .loss import LabelSmoothingLoss
# import visdom

logger = logging.getLogger(__name__)
# vis = visdom.Visdom(port=8888, env='vlp')

PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",
}
CONFIG_NAME = 'bert_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """

    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 relax_projection=0,
                 initializer_range=0.02,
                 task_idx=None,
                 fp32_embedding=False,
                 label_smoothing=None):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.relax_projection = relax_projection
            self.initializer_range = initializer_range
            self.task_idx = task_idx
            self.fp32_embedding = fp32_embedding
            self.label_smoothing = label_smoothing
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except ImportError:
    print("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex.")

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        """Construct a layernorm module in the TF style (epsilon inside the square root)."""
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        #print("Hello")
        #if torch.isnan(self.weight).any().item(): print( "nan exists in layerNorm forward() !!! ")
        return self.weight * x + self.bias


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size)
        if hasattr(config, 'fp32_embedding'):
            self.fp32_embedding = config.fp32_embedding
        else:
            self.fp32_embedding = False

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        #print("Inside embedding: hiddensize = ", config.hidden_size)

    def forward(self, vis_feats, vis_pe, input_ids, token_type_ids=None, context_is_img=False, position_ids=None, max_len_img_cxt=200, prev_is_None=True):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        #print("\ninput_ids.size() = ", input_ids.size())
        #print("\nposition_ids.size() = ", position_ids.size())
        #print("\nvis_feats.size() = ", vis_feats.size())
        #print("\nvis_pe.size() = ", vis_pe.size())
        if context_is_img and prev_is_None: 
            #print(vis_pe[0])
            #print(vis_feats[0])
            words_embeddings = torch.cat((words_embeddings[:, :1], vis_feats,
                words_embeddings[:, max_len_img_cxt+1:]), dim=1)
            assert max_len_img_cxt == 200, 'only support region attn!'
            position_embeddings = torch.cat((position_embeddings[:, :1], vis_pe,
                position_embeddings[:, max_len_img_cxt+1:]), dim=1) # hacky...
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        #print("\nwords_embeddings.size() = ", words_embeddings.size())
        #print("\nposition_embeddings_size() = ", position_embeddings.size())
        #print("\ntoken_type_embeddings.size() = ", token_type_embeddings.size())
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        if self.fp32_embedding:
            embeddings = embeddings.half()
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[
            :-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, history_states=None):
        if history_states is None:
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)
        else:
            x_states = torch.cat((history_states, hidden_states), dim=1)
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(x_states)
            mixed_value_layer = self.value(x_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / \
            math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
            :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, history_states=None):
        self_output = self.self(
            input_tensor, attention_mask, history_states=history_states)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, history_states=None):
        attention_output = self.attention(
            hidden_states, attention_mask, history_states=history_states)
        #if attention_output.size(1) > 200:
            #print("\nattention_output: ", attention_output[0][-1][:10])
            #print("\nattention_output, img token: ", attention_output[0][10][:10])
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, prev_embedding=None, prev_encoded_layers=None, output_all_encoded_layers=True):
        assert (prev_embedding is None) == (prev_encoded_layers is None), \
                "history embedding and encoded layer must be simultanously given."
        all_encoder_layers = []
        
        if (prev_embedding is not None) and (prev_encoded_layers is not None):
            
            history_states = prev_embedding
            for i, layer_module in enumerate(self.layer):
                hidden_states = layer_module(
                    hidden_states, attention_mask, history_states=history_states)
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
                if prev_encoded_layers is not None:
                    history_states = prev_encoded_layers[i]
        else:
            #print("attention_mask.size(): ", attention_mask.size())
            #print("\nattention_mask = ", attention_mask[0][0][-1])
            for layer_module in self.layer:
                hidden_states = layer_module(hidden_states, attention_mask)
                
                #print("\nencoder layer: ", hidden_states[0][-1][:10])
                #print("\nencoder layer, img token: ", hidden_states[0][10][:10])
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.transform_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act
        hid_size = config.hidden_size
        if hasattr(config, 'relax_projection') and (config.relax_projection > 1):
            hid_size *= config.relax_projection
        self.dense = nn.Linear(config.hidden_size, hid_size)
        self.LayerNorm = BertLayerNorm(hid_size, eps=1e-5)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(
            bert_model_embedding_weights.size(0)))
        if hasattr(config, 'relax_projection') and (config.relax_projection > 1):
            self.relax_projection = config.relax_projection
        else:
            self.relax_projection = 0
        self.fp32_embedding = config.fp32_embedding

        def convert_to_type(tensor):
            if self.fp32_embedding:
                return tensor.half()
            else:
                return tensor
        self.type_converter = convert_to_type
        self.converted = False

    def forward(self, hidden_states, task_idx=None):
        if not self.converted:
            self.converted = True
            if self.fp32_embedding:
                self.transform.half()
        hidden_states = self.transform(self.type_converter(hidden_states))
        if self.relax_projection > 1:
            num_batch = hidden_states.size(0)
            num_pos = hidden_states.size(1)
            # (batch, num_pos, relax_projection*hid) -> (batch, num_pos, relax_projection, hid) 
            # -> (batch, num_pos, hid) according to task_idx
            hidden_states = hidden_states.view(
                num_batch, num_pos, self.relax_projection, -1)[torch.arange(0, num_batch).long(), :, task_idx, :]
        if self.fp32_embedding:
            hidden_states = F.linear(self.type_converter(hidden_states), self.type_converter(
                self.decoder.weight), self.type_converter(self.bias))
        else:
            hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(
            config, bert_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights, num_labels=2):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(
            config, bert_model_embedding_weights)
        # self.seq_relationship = nn.Linear(config.hidden_size, num_labels)

    def forward(self, sequence_output, pooled_output, task_idx=None):
        prediction_scores = self.predictions(sequence_output, task_idx)
        seq_relationship_score = None
        # if pooled_output is None:
        #     seq_relationship_score = None
        # else:
        #     seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class PreTrainedBertModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super(PreTrainedBertModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name, state_dict=None, cache_dir=None, *inputs, **kwargs):
        """
        Instantiate a PreTrainedBertModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `bert-base-uncased`
                    . `bert-large-uncased`
                    . `bert-base-cased`
                    . `bert-base-multilingual`
                    . `bert-base-chinese`
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        if pretrained_model_name in PRETRAINED_MODEL_ARCHIVE_MAP:
            archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name]
        else:
            archive_file = pretrained_model_name
        # redirect to the cache, if necessary
        try:
            resolved_archive_file = cached_path(
                archive_file, cache_dir=cache_dir)
        except FileNotFoundError:
            logger.error(
                "Model name '{}' was not found in model name list ({}). "
                "We assumed '{}' was a path or url but couldn't find any file "
                "associated to this path or url.".format(
                    pretrained_model_name,
                    ', '.join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
                    archive_file))
            return None
        if resolved_archive_file == archive_file:
            logger.info("loading archive file {}".format(archive_file))
        else:
            logger.info("loading archive file {} from cache at {}".format(
                archive_file, resolved_archive_file))
        tempdir = None
        if os.path.isdir(resolved_archive_file):
            serialization_dir = resolved_archive_file
        else:
            # Extract archive to temp dir
            tempdir = tempfile.mkdtemp()
            logger.info("extracting archive file {} to temp dir {}".format(
                resolved_archive_file, tempdir))
            with tarfile.open(resolved_archive_file, 'r:gz') as archive:
                archive.extractall(tempdir)
            serialization_dir = tempdir
        # Load config
        if ('config_path' in kwargs) and kwargs['config_path']:
            config_file = kwargs['config_path']
        else:
            config_file = os.path.join(serialization_dir, CONFIG_NAME)
        config = BertConfig.from_json_file(config_file)

        # define new type_vocab_size (there might be different numbers of segment ids)
        if 'type_vocab_size' in kwargs:
            config.type_vocab_size = kwargs['type_vocab_size']
        # define new relax_projection
        if ('relax_projection' in kwargs) and kwargs['relax_projection']:
            config.relax_projection = kwargs['relax_projection']
        # define new relax_projection
        if ('task_idx' in kwargs) and kwargs['task_idx']:
            config.task_idx = kwargs['task_idx']
        # define new max position embedding for length expansion
        if ('max_position_embeddings' in kwargs) and kwargs['max_position_embeddings']:
            config.max_position_embeddings = kwargs['max_position_embeddings']
        # use fp32 for embeddings
        if ('fp32_embedding' in kwargs) and kwargs['fp32_embedding']:
            config.fp32_embedding = kwargs['fp32_embedding']
        # label smoothing
        if ('label_smoothing' in kwargs) and kwargs['label_smoothing']:
            config.label_smoothing = kwargs['label_smoothing']
        if 'drop_prob' in kwargs:
            print('setting the new dropout rate!', kwargs['drop_prob'])
            config.attention_probs_dropout_prob = kwargs['drop_prob']
            config.hidden_dropout_prob = kwargs['drop_prob']

        logger.info("Model config {}".format(config))

        # clean the arguments in kwargs
        for arg_clean in ('config_path', 'type_vocab_size', 'relax_projection', 'task_idx', 'max_position_embeddings', 'fp32_embedding', 'label_smoothing', 'drop_prob'):
            if arg_clean in kwargs:
                del kwargs[arg_clean]

        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None:
            weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
            state_dict = torch.load(weights_path)

        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        # initialize new segment embeddings
        _k = 'bert.embeddings.token_type_embeddings.weight'
        if (_k in state_dict) and (config.type_vocab_size != state_dict[_k].shape[0]):
            logger.info("config.type_vocab_size != state_dict[bert.embeddings.token_type_embeddings.weight] ({0} != {1})".format(
                config.type_vocab_size, state_dict[_k].shape[0]))
            if config.type_vocab_size > state_dict[_k].shape[0]:
                # state_dict[_k].data = state_dict[_k].data.resize_(
                state_dict[_k].data = state_dict[_k].resize_(
                    config.type_vocab_size, state_dict[_k].shape[1]).data
                if config.type_vocab_size >= 6:
                    # L2R
                    state_dict[_k].data[2, :].copy_(state_dict[_k].data[0, :])
                    # R2L
                    state_dict[_k].data[3, :].copy_(state_dict[_k].data[0, :])
                    # S2S
                    state_dict[_k].data[4, :].copy_(state_dict[_k].data[0, :])
                    state_dict[_k].data[5, :].copy_(state_dict[_k].data[1, :])
            elif config.type_vocab_size < state_dict[_k].shape[0]:
                state_dict[_k].data = state_dict[_k].data[:config.type_vocab_size, :]

        # initialize new position embeddings
        _k = 'bert.embeddings.position_embeddings.weight'
        if _k in state_dict and config.max_position_embeddings != state_dict[_k].shape[0]:
            logger.info("config.max_position_embeddings != state_dict[bert.embeddings.position_embeddings.weight] ({0} - {1})".format(
                config.max_position_embeddings, state_dict[_k].shape[0]))
            if config.max_position_embeddings > state_dict[_k].shape[0]:
                old_size = state_dict[_k].shape[0]
                state_dict[_k].data = state_dict[_k].data.resize_(
                    config.max_position_embeddings, state_dict[_k].shape[1])
                start = old_size
                while start < config.max_position_embeddings:
                    chunk_size = min(
                        old_size, config.max_position_embeddings - start)
                    state_dict[_k].data[start:start+chunk_size,
                                        :].copy_(state_dict[_k].data[:chunk_size, :])
                    start += chunk_size
            elif config.max_position_embeddings < state_dict[_k].shape[0]:
                state_dict[_k].data = state_dict[_k].data[:config.max_position_embeddings, :]

        # initialize relax projection
        _k = 'cls.predictions.transform.dense.weight'
        n_config_relax = 1 if (config.relax_projection <
                               1) else config.relax_projection
        if (_k in state_dict) and (n_config_relax*config.hidden_size != state_dict[_k].shape[0]):
            logger.info("n_config_relax*config.hidden_size != state_dict[cls.predictions.transform.dense.weight] ({0}*{1} != {2})".format(
                n_config_relax, config.hidden_size, state_dict[_k].shape[0]))
            assert state_dict[_k].shape[0] % config.hidden_size == 0
            n_state_relax = state_dict[_k].shape[0]//config.hidden_size
            assert (n_state_relax == 1) != (n_config_relax ==
                                            1), "!!!!n_state_relax == 1 xor n_config_relax == 1!!!!"
            if n_state_relax == 1:
                _k = 'cls.predictions.transform.dense.weight'
                state_dict[_k].data = state_dict[_k].data.unsqueeze(0).repeat(
                    n_config_relax, 1, 1).reshape((n_config_relax*config.hidden_size, config.hidden_size))
                for _k in ('cls.predictions.transform.dense.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.LayerNorm.bias'):
                    state_dict[_k].data = state_dict[_k].data.unsqueeze(
                        0).repeat(n_config_relax, 1).view(-1)
            elif n_config_relax == 1:
                if hasattr(config, 'task_idx') and (config.task_idx is not None) and (0 <= config.task_idx <= 3):
                    _task_idx = config.task_idx
                else:
                    _task_idx = 0
                _k = 'cls.predictions.transform.dense.weight'
                state_dict[_k].data = state_dict[_k].data.view(
                    n_state_relax, config.hidden_size, config.hidden_size).select(0, _task_idx)
                for _k in ('cls.predictions.transform.dense.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.LayerNorm.bias'):
                    state_dict[_k].data = state_dict[_k].data.view(
                        n_state_relax, config.hidden_size).select(0, _task_idx)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(
                prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        load(model, prefix='' if hasattr(model, 'bert') else 'bert.')
        model.missing_keys = missing_keys
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            logger.info('\n'.join(error_msgs))
        if tempdir:
            # Clean up temp dir
            shutil.rmtree(tempdir)
        return model


class BertModel(PreTrainedBertModel):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Params:
        config: a BertConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLF`) to train on the Next-Sentence task (see BERT's paper).

    """

    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)


    def get_extended_attention_mask(self, input_ids, token_type_ids, attention_mask):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask


    def forward(self, vis_feats, vis_pe, input_ids, token_type_ids=None, attention_mask=None, context_is_img=True, output_all_encoded_layers=True, max_len_img_cxt=200):
            
        extended_attention_mask = self.get_extended_attention_mask(
            input_ids, token_type_ids, attention_mask)

        # hack to load vis feats
        embedding_output = self.embeddings(vis_feats, vis_pe, input_ids, token_type_ids, context_is_img=context_is_img, max_len_img_cxt=max_len_img_cxt)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output


class BertModelIncr(BertModel):
    def __init__(self, config):
        super(BertModelIncr, self).__init__(config)

    def forward(self, vis_feats, vis_pe, input_ids, token_type_ids, position_ids, attention_mask, context_is_img, 
                prev_embedding=None, prev_encoded_layers=None, output_all_encoded_layers=True, max_len_img_cxt=200):
        
        extended_attention_mask = self.get_extended_attention_mask(
            input_ids, token_type_ids, attention_mask)
        #if prev_embedding is None:
            #print("\nattention_mask: ", attention_mask[0][10])
            #print("\nextended_attention_mask.size() = ", extended_attention_mask.size())
        prev_is_None = prev_embedding is None
        #print(prev_is_None)
        embedding_output = self.embeddings(
            vis_feats, vis_pe, input_ids, token_type_ids, position_ids=position_ids, context_is_img=context_is_img, max_len_img_cxt=max_len_img_cxt, prev_is_None=prev_is_None)
        
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      prev_embedding=prev_embedding,
                                      prev_encoded_layers=prev_encoded_layers,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        #if prev_is_None:
            #print(embedding_output[0][-1][:10])
            #print("encoded_layers[0].size() = ", encoded_layers[0].size())
            #print(encoded_layers[0][0][-1][:10])
            #print(encoded_layers[-1][0][-1][:10])
            #print(pooled_output[0][0][:10])
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return embedding_output, encoded_layers, pooled_output


class BertForPreTraining(PreTrainedBertModel):
    """BERT model with pre-training heads.
    This module comprises the BERT model followed by the two pre-training heads:
        - the masked language modeling head, and
        - the next sentence classification head.
    Params:
        config: a BertConfig class instance with the configuration to build a new model.
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]
        `next_sentence_label`: next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.
    Outputs:
        if `masked_lm_labels` and `next_sentence_label` are not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `masked_lm_labels` or `next_sentence_label` is `None`:
            Outputs a tuple comprising
            - the masked language modeling logits of shape [batch_size, sequence_length, vocab_size], and
            - the next sentence classification logits of shape [batch_size, 2].
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    model = BertForPreTraining(config)
    masked_lm_logits_scores, seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(BertForPreTraining, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None, next_sentence_label=None):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                                   output_all_encoded_layers=False)
        prediction_scores, seq_relationship_score = self.cls(
            sequence_output, pooled_output)

        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(
                seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            return total_loss
        else:
            return prediction_scores, seq_relationship_score


class BertPreTrainingPairTransform(nn.Module):
    def __init__(self, config):
        super(BertPreTrainingPairTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act
        # self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)

    def forward(self, pair_x, pair_y):
        hidden_states = torch.cat([pair_x, pair_y], dim=-1)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        # hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertPreTrainingPairRel(nn.Module):
    def __init__(self, config, num_rel=0):
        super(BertPreTrainingPairRel, self).__init__()
        self.R_xy = BertPreTrainingPairTransform(config)
        self.rel_emb = nn.Embedding(num_rel, config.hidden_size)

    def forward(self, pair_x, pair_y, pair_r, pair_pos_neg_mask):
        # (batch, num_pair, hidden)
        xy = self.R_xy(pair_x, pair_y)
        r = self.rel_emb(pair_r)
        _batch, _num_pair, _hidden = xy.size()
        pair_score = (xy * r).sum(-1)
        # torch.bmm(xy.view(-1, 1, _hidden),r.view(-1, _hidden, 1)).view(_batch, _num_pair)
        # .mul_(-1.0): objective to loss
        return F.logsigmoid(pair_score * pair_pos_neg_mask.type_as(pair_score)).mul_(-1.0)


""" for VLP, based on UniLM """
class BertForPreTrainingLossMask(PreTrainedBertModel):
    """refer to BertForPreTraining"""

    def __init__(self, config, num_labels=2, enable_butd=False, len_vis_input=49, tasks='img2txt'):
        super(BertForPreTrainingLossMask, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight, num_labels=num_labels) # num_labels not applicable for VLP
        self.apply(self.init_bert_weights)
        self.crit_mask_lm = nn.CrossEntropyLoss(reduction='none')
        self.num_labels = num_labels
        self.len_vis_input = len_vis_input
        self.enable_butd = enable_butd
        if hasattr(config, 'label_smoothing') and config.label_smoothing:
            self.crit_mask_lm_smoothed = LabelSmoothingLoss(
                config.label_smoothing, config.vocab_size, ignore_index=0, reduction='none')
        else:
            self.crit_mask_lm_smoothed = None

        # will not be initialized when loading BERT weights
        if enable_butd:
            self.vis_embed = nn.Sequential(nn.Linear(2048, 2048),
                                       nn.ReLU(),
                                       nn.Linear(2048, config.hidden_size),
                                       nn.ReLU(),
                                       nn.Dropout(config.hidden_dropout_prob)) # use to be 0.3
            try:
                self.vis_embed[0].weight.data.copy_(torch.from_numpy(pickle.load(
                    open('detectron_weights/fc7_w.pkl', 'rb'))))
                self.vis_embed[0].bias.data.copy_(torch.from_numpy(pickle.load(
                    open('detectron_weights/fc7_b.pkl', 'rb'))))
            except:
                raise Exception('Cannot find Detectron fc7 weights! Download from https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/detectron_weights.tar.gz and uncompress under the code root directory.')

            self.vis_pe_embed = nn.Sequential(nn.Linear(6+1601, config.hidden_size),
                                       nn.ReLU(),
                                       nn.Dropout(config.hidden_dropout_prob))
        else:
            self.vis_embed = nn.Sequential(nn.Linear(2048, config.hidden_size*2),
                                       nn.ReLU(),
                                       nn.Linear(config.hidden_size*2, config.hidden_size),
                                       nn.ReLU(),
                                       nn.Dropout(config.hidden_dropout_prob)) # use to be 0.3
        self.tasks = tasks
        if tasks == 'vqa2':
            self.ans_classifier = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size*2),
                                       nn.ReLU(),
                                       nn.Linear(config.hidden_size*2, 3129)) # 3129 hard coded...
            self.vqa2_crit = nn.BCEWithLogitsLoss()


    def forward(self, vis_feats, vis_pe, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None, ans_labels=None, next_sentence_label=None, masked_pos=None, masked_weights=None, task_idx=None, vis_masked_pos=[], mask_image_regions=False, drop_worst_ratio=0.2, vqa_inference=False):

        vis_feats = self.vis_embed(vis_feats) # image region features Bx100xhidden_size
        vis_pe = self.vis_pe_embed(vis_pe) # image region positional encodings Bx100xhidden_size
        
        # VQA inference
        if vqa_inference: # vqa_inference=False during training
            assert(ans_labels == None) # in inference mode, need no gth label
            sequence_output, pooled_output = self.bert(vis_feats, vis_pe, input_ids, token_type_ids,
                attention_mask, output_all_encoded_layers=False, len_vis_input=self.len_vis_input)

            vqa2_embed = sequence_output[:, 0]*sequence_output[:, self.len_vis_input+1]
            vqa2_pred = self.ans_classifier(vqa2_embed)
            ans_idx = torch.max(vqa2_pred[:, 1:], -1)[1] + 1
            return ans_idx

        # zero out vis_masked_pos
        if mask_image_regions:
            vis_feat_mask = vis_masked_pos.new(*vis_feats.size()[:2], 1).fill_(0).byte() # Bx100x1
            for bb in range(vis_masked_pos.size(0)):
                for pp in range(vis_masked_pos.size(1)): # when vis_masked_pos was generated, 1 were added to pos_ids to account for [CLS]
                    vis_feat_mask[bb, vis_masked_pos[bb, pp]-1] = 1
            sequence_output, pooled_output = self.bert(vis_feats.masked_fill(vis_feat_mask, 0.),
                vis_pe.masked_fill(vis_feat_mask, 0.), input_ids, token_type_ids,
                attention_mask, output_all_encoded_layers=False, len_vis_input=self.len_vis_input)
        else:
            sequence_output, pooled_output = self.bert(vis_feats, vis_pe, input_ids, token_type_ids,
                attention_mask, output_all_encoded_layers=False, len_vis_input=self.len_vis_input)

        if masked_lm_labels is None or next_sentence_label is None:
            raise NotImplementedError
            # prediction_scores, seq_relationship_score = self.cls(
            #     sequence_output, pooled_output, task_idx=task_idx)
            # return prediction_scores, seq_relationship_score

        def gather_seq_out_by_pos(seq, pos):
            return torch.gather(seq, 1, pos.unsqueeze(2).expand(-1, -1, seq.size(-1)))

        def gather_seq_out_by_pos_average(seq, pos, mask): # Unused function
            # pos/mask: (batch, num_pair, max_token_num)
            batch_size, max_token_num = pos.size(0), pos.size(-1)
            # (batch, num_pair, max_token_num, seq.size(-1))
            pos_vec = torch.gather(seq, 1, pos.view(batch_size, -1).unsqueeze(
                2).expand(-1, -1, seq.size(-1))).view(batch_size, -1, max_token_num, seq.size(-1))
            # (batch, num_pair, seq.size(-1))
            mask = mask.type_as(pos_vec)
            pos_vec_masked_sum = (
                pos_vec * mask.unsqueeze(3).expand_as(pos_vec)).sum(2)
            return pos_vec_masked_sum / mask.sum(2, keepdim=True).expand_as(pos_vec_masked_sum)

        def loss_mask_and_normalize(loss, mask, drop_worst_ratio):
            mask = mask.type_as(loss) # B x max_pred?
            loss = loss * mask

            # Ruotian Luo's drop worst (drop batches with worst losses)
            # <less_than_B> x max_pred
            keep_loss, keep_ind = torch.topk(loss.sum(-1), int(loss.size(0)*(1-drop_worst_ratio)), largest=False)

            # denominator = torch.sum(mask) + 1e-5
            # return (loss / denominator).sum()
            # each batch has different number of actual predictions
            # divide by num_actual_predictions for each batch
            # losses on the placeholder tokens should be zero, still being zero after divided by 1e-5
            denominator = torch.sum(mask.sum(-1)[keep_ind]) + 1e-5
            return (keep_loss / denominator).sum() # sum losses over <less_than_B> batches

        # masked lm
        if masked_pos.numel() == 0: # it seems that this won't happen for VLP?
            # hack to avoid empty masked_pos during training for now
            masked_lm_loss = pooled_output.new(1).fill_(0) # tensor([0])
        else:
            sequence_output_masked = gather_seq_out_by_pos(
                sequence_output, masked_pos) # B x max_pred x hidden
            prediction_scores_masked, _ = self.cls(
                sequence_output_masked, pooled_output, task_idx=task_idx) # B x max_pred x vocab_size
            if self.crit_mask_lm_smoothed:
                masked_lm_loss = self.crit_mask_lm_smoothed(
                    F.log_softmax(prediction_scores_masked.float(), dim=-1), masked_lm_labels)
            else:
                
                masked_lm_loss = self.crit_mask_lm(
                    prediction_scores_masked.transpose(1, 2).float(), masked_lm_labels)
            masked_lm_loss = loss_mask_and_normalize(
                masked_lm_loss.float(), masked_weights, drop_worst_ratio)

        # vis_feats, vis_pe have been projected to hidden_size dim
        if mask_image_regions:
            # Selfie-like pretext
            # gth img_feats for masked regions
            masked_vis_feats = torch.gather(vis_feats, 1,
                (vis_masked_pos-1).unsqueeze(-1).expand((-1, -1, vis_feats.size(-1))))
            # vis_masked_pos: B x <len_vis_input*vis_mask_prob>
            # unsqueeze: B x <len_vis_input*vis_mask_prob> x 1
            # expand: B x <len_vis_input*vis_mask_prob> x hidden
            # output of this line: B x <len_vis_input*vis_mask_prob> x hidden

            # gth vis_pe for masked regions
            if self.enable_butd:
                masked_pos_enc = torch.gather(vis_pe, 1,
                (vis_masked_pos-1).unsqueeze(-1).expand((-1, -1, vis_pe.size(-1))))
                # B x <len_vis_input*vis_mask_prob> x hidden
            else:
                masked_pos_enc = self.bert.embeddings.position_embeddings(vis_masked_pos)

            # pooled_output: B x hidden
            # unsqueeze: B x 1 x hidden
            # expand_as: B x <len_vis_input*vis_mask_prob> x hidden
            # only pooled_output here gets trained???
            masked_pos_enc += pooled_output.unsqueeze(1).expand_as(masked_pos_enc) # ? Why add pooled_output to gth masked vis pe?
            assert(masked_vis_feats.size() == masked_pos_enc.size())
            # pe of a particular region should be most compatible with vis_feat of that region
            sim_mat = torch.matmul(masked_pos_enc, masked_vis_feats.permute(0, 2, 1).contiguous())
            # B x <len_vis_input*vis_mask_prob> x <len_vis_input*vis_mask_prob>
            sim_mat = F.log_softmax(sim_mat, dim=-1)
            vis_pretext_loss = []
            for i in range(sim_mat.size(0)): # gth is the Identity matrix
                vis_pretext_loss.append(sim_mat[i].diag().mean().view(1)*-1.) # cross entropy for ones
            vis_pretext_loss = torch.cat(vis_pretext_loss).mean() # mean over B batches??? but masked_lm_loss is sum over batches?
        else:
            vis_pretext_loss = masked_lm_loss.new(1).fill_(0)

        if self.tasks == 'vqa2':
            assert(ans_labels is not None)
            # vqa2_embed = pooled_output
            vqa2_embed = sequence_output[:, 0]*sequence_output[:, self.len_vis_input+1]
            vqa2_pred = self.ans_classifier(vqa2_embed) # B x 3129 # without softmax
            # ans_labels: B x 3129
            vqa2_loss = self.vqa2_crit(vqa2_pred, ans_labels) * ans_labels.size(1) # should not avg over answer dimension
            # vqa2_loss has been averaged over batches
            return masked_lm_loss.new(1).fill_(0), vis_pretext_loss, vqa2_loss # works better when combined with max_pred=1
        else:
            return masked_lm_loss, vis_pretext_loss, masked_lm_loss.new(1).fill_(0)


""" for VLP, based on UniLM """
class BertForSeq2SeqDecoder(PreTrainedBertModel):
    """refer to BertForPreTraining"""

    def __init__(self, config, mask_word_id=0, num_labels=2,
                 search_beam_size=1, length_penalty=1.0, eos_id=0,
                 forbid_duplicate_ngrams=False, forbid_ignore_set=None,
                 ngram_size=3, min_len=0, enable_butd=False, len_vis_input=49):
        super(BertForSeq2SeqDecoder, self).__init__(config)
        self.bert = BertModelIncr(config)
        self.cls = BertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight, num_labels=num_labels)
        self.apply(self.init_bert_weights)
        self.crit_mask_lm = nn.CrossEntropyLoss(reduction='none')
        self.mask_word_id = mask_word_id
        self.num_labels = num_labels
        self.len_vis_input = len_vis_input
        self.search_beam_size = search_beam_size
        self.length_penalty = length_penalty
        self.eos_id = eos_id
        self.forbid_duplicate_ngrams = forbid_duplicate_ngrams
        self.forbid_ignore_set = forbid_ignore_set
        self.ngram_size = ngram_size
        self.min_len = min_len

        # will not be initialized when loading BERT weights
        if enable_butd:
            self.vis_embed = nn.Sequential(nn.Linear(2048, 2048),
                                       nn.ReLU(),
                                       nn.Linear(2048, config.hidden_size),
                                       nn.ReLU(),
                                       nn.Dropout(config.hidden_dropout_prob))
            self.vis_pe_embed = nn.Sequential(nn.Linear(6+1601, config.hidden_size),
                                       nn.ReLU(),
                                       nn.Dropout(config.hidden_dropout_prob))
        else:
            self.vis_embed = nn.Sequential(nn.Linear(2048, config.hidden_size*2),
                                       nn.ReLU(),
                                       nn.Linear(config.hidden_size*2, config.hidden_size),
                                       nn.ReLU(),
                                       nn.Dropout(config.hidden_dropout_prob))


    def forward(self, vis_feats, vis_pe, input_ids, token_type_ids, position_ids, attention_mask, task_idx=None, sample_mode='greedy'):

        vis_feats = self.vis_embed(vis_feats) # image region features
        vis_pe = self.vis_pe_embed(vis_pe) # image region positional encodings

        if self.search_beam_size > 1:
            return self.beam_search(vis_feats, vis_pe, input_ids, token_type_ids, position_ids, attention_mask, task_idx)
        input_shape = list(input_ids.size())
        batch_size = input_shape[0]
        input_length = input_shape[1]
        output_shape = list(token_type_ids.size())
        output_length = output_shape[1]

        output_ids = []
        output_probs = []
        prev_embedding = None
        prev_encoded_layers = None
        curr_ids = input_ids
        mask_ids = input_ids[:, :1] * 0 + self.mask_word_id
        next_pos = input_length

        while next_pos < output_length:
            curr_length = list(curr_ids.size())[1]
            start_pos = next_pos - curr_length
            x_input_ids = torch.cat((curr_ids, mask_ids), dim=1)
            curr_token_type_ids = token_type_ids[:, start_pos:next_pos + 1]
            curr_attention_mask = attention_mask[:,
                start_pos:next_pos + 1, :next_pos + 1]
            curr_position_ids = position_ids[:, start_pos:next_pos + 1]
            new_embedding, new_encoded_layers, _ = \
                self.bert(vis_feats, vis_pe, x_input_ids, curr_token_type_ids, curr_position_ids,
                curr_attention_mask, prev_embedding=prev_embedding,
                prev_encoded_layers=prev_encoded_layers,
                output_all_encoded_layers=True, len_vis_input=self.len_vis_input)

            last_hidden = new_encoded_layers[-1][:, -1:, :]
            prediction_scores, _ = self.cls(
                last_hidden, None, task_idx=task_idx)
            if sample_mode == 'greedy':
                max_probs, max_ids = torch.max(prediction_scores, dim=-1)
            elif sample_mode == 'sample':
                prediction_scores.squeeze_(1)
                prediction_probs = F.softmax(prediction_scores, dim=-1).detach()
                max_ids = torch.multinomial(prediction_probs, num_samples=1,
                    replacement=True)
                max_probs = torch.gather(F.log_softmax(prediction_scores, dim=-1),
                    1, max_ids) # this should be logprobs
            else:
                raise NotImplementedError
            output_ids.append(max_ids)
            output_probs.append(max_probs)
            if prev_embedding is None:
                prev_embedding = new_embedding[:, :-1, :]
            else:
                prev_embedding = torch.cat(
                    (prev_embedding, new_embedding[:, :-1, :]), dim=1)
            if prev_encoded_layers is None:
                prev_encoded_layers = [x[:, :-1, :]
                                       for x in new_encoded_layers]
            else:
                prev_encoded_layers = [torch.cat((x[0], x[1][:, :-1, :]), dim=1)
                                       for x in zip(prev_encoded_layers, new_encoded_layers)]
            curr_ids = max_ids
            next_pos += 1
        return torch.cat(output_ids, dim=1), torch.cat(output_probs, dim=1)


    def beam_search(self, vis_feats, vis_pe, input_ids, token_type_ids, position_ids, attention_mask, task_idx=None):

        input_shape = list(input_ids.size()) # batch_size x (max_len_a+2)
        batch_size = input_shape[0]
        input_length = input_shape[1]
        output_shape = list(token_type_ids.size()) # batch_size x max_len_in_batch
        output_length = output_shape[1]

        output_ids = []
        prev_embedding = None
        prev_encoded_layers = None
        curr_ids = input_ids
        mask_ids = input_ids[:, :1] * 0 + self.mask_word_id
        next_pos = input_length

        K = self.search_beam_size

        total_scores = []
        beam_masks = []
        step_ids = []
        step_back_ptrs = []
        partial_seqs = []
        forbid_word_mask = None
        buf_matrix = None

        while next_pos < output_length:
            curr_length = list(curr_ids.size())[1]
            start_pos = next_pos - curr_length
            x_input_ids = torch.cat((curr_ids, mask_ids), dim=1)
            curr_token_type_ids = token_type_ids[:, start_pos:next_pos + 1]
            curr_attention_mask = attention_mask[:,
                                                 start_pos:next_pos + 1, :next_pos + 1]
            curr_position_ids = position_ids[:, start_pos:next_pos + 1]
            new_embedding, new_encoded_layers, _ = \
                self.bert(vis_feats, vis_pe, x_input_ids, curr_token_type_ids, curr_position_ids,
                curr_attention_mask, prev_embedding=prev_embedding,
                prev_encoded_layers=prev_encoded_layers,
                output_all_encoded_layers=True, len_vis_input=self.len_vis_input)
            # compared with BertModel, BertModelIncr returns an additional embedding_output from forward()
            # new_encoded_layers == sequence_output

            last_hidden = new_encoded_layers[-1][:, -1:, :] # batch_size(*K) x 1 x emb_dim
            prediction_scores, _ = self.cls(last_hidden, None, task_idx=task_idx) # batch_size(*K) x 1 x vocab_size
            log_scores = torch.nn.functional.log_softmax(prediction_scores, dim=-1) # batch_size(*K) x 1 x vocab_size
            if forbid_word_mask is not None:
                log_scores += (forbid_word_mask * -10000.0) 
                # forbid_word_mask has the same size as log_scores, positions with 1 indicate forbid_words
            if self.min_len and (next_pos-input_length+1 <= self.min_len):
                log_scores[:, :, self.eos_id].fill_(-10000.0) # forbid generating <eos> when min_len is not reached
            kk_scores, kk_ids = torch.topk(log_scores, k=K) # batch_size(*K) x 1 x K
            if len(total_scores) == 0:
                k_ids = torch.reshape(kk_ids, [batch_size, K])
                back_ptrs = torch.zeros(batch_size, K, dtype=torch.long)
                k_scores = torch.reshape(kk_scores, [batch_size, K])
            else:
                last_eos = torch.reshape(
                    beam_masks[-1], [batch_size * K, 1, 1])
                last_seq_scores = torch.reshape(
                    total_scores[-1], [batch_size * K, 1, 1])
                kk_scores += last_eos * (-10000.0) + last_seq_scores # don't consider beams that already have <eos> in the last step
                kk_scores = torch.reshape(kk_scores, [batch_size, K * K])
                k_scores, k_ids = torch.topk(kk_scores, k=K) # sample K from K*K # batch_size x K
                back_ptrs = torch.div(k_ids, K) # batch_size x K (choose top K from K*K)
                kk_ids = torch.reshape(kk_ids, [batch_size, K * K])
                k_ids = torch.gather(kk_ids, 1, k_ids) # batch_size x K    top K idx of kk_scores --> top K vocab idx
            step_back_ptrs.append(back_ptrs) # record prev_step beam id. get the actual word_id from corresponding pos in step_ids
            step_ids.append(k_ids)
            beam_masks.append(torch.eq(k_ids, self.eos_id).float()) # block beams that reach <eos> at an early point
            # stop beam if <eos> is generated
            # but that won't happen bc <eos> is masked in log_scores if output_len < min_len ???
            total_scores.append(k_scores)

            def first_expand(x):
                input_shape = list(x.size())
                expanded_shape = input_shape[:1] + [1] + input_shape[1:]
                x = torch.reshape(x, expanded_shape) # batch_size x 1 x ori_len x emb_dim
                repeat_count = [1, K] + [1] * (len(input_shape) - 1) # [1, K, 1, 1] 
                x = x.repeat(*repeat_count) # --> batch_size x K x ori_len x emb_dim
                x = torch.reshape(x, [input_shape[0] * K] + input_shape[1:]) # (batch_size * K) x ori_len x emb_dim
                return x

            def select_beam_items(x, ids):
                # ids:
                # batch_size x K (choose top K from K*K)
                id_shape = list(ids.size())
                id_rank = len(id_shape)
                assert len(id_shape) == 2
                x_shape = list(x.size())
                x = torch.reshape(x, [batch_size, K] + x_shape[1:]) # batch_size x K x (ori_len+1) x emb_dim
                x_rank = len(x_shape) + 1
                assert x_rank >= 2
                if id_rank < x_rank:
                    ids = torch.reshape(
                        ids, id_shape + [1] * (x_rank - id_rank)) # batch_size x K x 1 x 1
                    ids = ids.expand(id_shape + x_shape[1:]) # batch_size x K x (ori_len+1) x emb_dim
                y = torch.gather(x, 1, ids) # batch_size x K x (ori_len+1) x emb_dim
                y = torch.reshape(y, x_shape) # batch_size*K x (ori_len+1) x emb_dim
                return y

            is_first = (prev_embedding is None)

            if prev_embedding is None:
                prev_embedding = first_expand(new_embedding[:, :-1, :]) 
                # ori_len = length before appending mask
                # batch_size x ori_len x emb_dim 
            else:
                prev_embedding = torch.cat(
                    (prev_embedding, new_embedding[:, :-1, :]), dim=1) # new_embedding[:, :-1, :].size() = batch_size*K x 1 x emb_dim
                    # --> batch_size*K x (ori_len+1) x emb_dim
                prev_embedding = select_beam_items(prev_embedding, back_ptrs)
            if prev_encoded_layers is None:
                # new_encoded_layers: num_layers x batch_size x ori_len x emb_dim
                prev_encoded_layers = [first_expand(
                    x[:, :-1, :]) for x in new_encoded_layers]
            else:
                prev_encoded_layers = [torch.cat((x[0], x[1][:, :-1, :]), dim=1)
                                       for x in zip(prev_encoded_layers, new_encoded_layers)]
                prev_encoded_layers = [select_beam_items(
                    x, back_ptrs) for x in prev_encoded_layers]

            curr_ids = torch.reshape(k_ids, [batch_size * K, 1])

            if is_first:
                token_type_ids = first_expand(token_type_ids)
                position_ids = first_expand(position_ids)
                attention_mask = first_expand(attention_mask)
                mask_ids = first_expand(mask_ids)

            if self.forbid_duplicate_ngrams:
                wids = step_ids[-1].tolist()
                ptrs = step_back_ptrs[-1].tolist()
                if is_first:
                    partial_seqs = []
                    for b in range(batch_size):
                        for k in range(K):
                            partial_seqs.append([wids[b][k]])
                else:
                    new_partial_seqs = []
                    for b in range(batch_size):
                        for k in range(K):
                            new_partial_seqs.append(
                                partial_seqs[ptrs[b][k] + b * K] + [wids[b][k]])
                    partial_seqs = new_partial_seqs

                def get_dup_ngram_candidates(seq, n):
                    cands = set()
                    if len(seq) < n:
                        return []
                    tail = seq[-(n-1):]
                    if self.forbid_ignore_set and any(tk in self.forbid_ignore_set for tk in tail):
                        return []
                    for i in range(len(seq) - (n - 1)):
                        mismatch = False
                        for j in range(n - 1):
                            if tail[j] != seq[i + j]:
                                mismatch = True
                                break
                        if (not mismatch) and not(self.forbid_ignore_set and (seq[i + n - 1] in self.forbid_ignore_set)):
                            cands.add(seq[i + n - 1])
                    return list(sorted(cands))

                if len(partial_seqs[0]) >= self.ngram_size:
                    dup_cands = []
                    for seq in partial_seqs:
                        dup_cands.append(
                            get_dup_ngram_candidates(seq, self.ngram_size))
                    if max(len(x) for x in dup_cands) > 0:
                        if buf_matrix is None:
                            vocab_size = list(log_scores.size())[-1]
                            buf_matrix = np.zeros(
                                (batch_size * K, vocab_size), dtype=float)
                        else:
                            buf_matrix.fill(0)
                        for bk, cands in enumerate(dup_cands):
                            for i, wid in enumerate(cands):
                                buf_matrix[bk, wid] = 1.0
                        forbid_word_mask = torch.tensor(
                            buf_matrix, dtype=log_scores.dtype)
                        forbid_word_mask = torch.reshape(
                            forbid_word_mask, [batch_size * K, 1, vocab_size]).cuda()
                    else:
                        forbid_word_mask = None
            next_pos += 1

        # [(batch, beam)]
        total_scores = [x.tolist() for x in total_scores] # each x: batch_size x K
        step_ids = [x.tolist() for x in step_ids]
        step_back_ptrs = [x.tolist() for x in step_back_ptrs]
        # back tracking
        traces = {'pred_seq': [], 'scores': [], 'wids': [], 'ptrs': []}
        for b in range(batch_size):
            # [(beam,)]
            scores = [x[b] for x in total_scores] # output_len x K 
            wids_list = [x[b] for x in step_ids]  # output_len x K 
            ptrs = [x[b] for x in step_back_ptrs] # output_len x K 
            traces['scores'].append(scores) # 这个跟 total_scores, step_ids, step_back_ptrs有什么区别啊。。。又重新一个个append一遍。。。
            traces['wids'].append(wids_list)
            traces['ptrs'].append(ptrs)
            # first we need to find the eos frame where all symbols are eos
            # any frames after the eos frame are invalid
            last_frame_id = len(scores) - 1 # last id within output_len
            for i, wids in enumerate(wids_list): 
                # if all topK beams reach <eos> before reaching output_len, then set last_frame_id to an earlier frame.
                if all(wid == self.eos_id for wid in wids):
                    last_frame_id = i
                    break
            max_score = -math.inf
            frame_id = -1
            pos_in_frame = -1

            for fid in range(last_frame_id + 1):
                for i, wid in enumerate(wids_list[fid]):
                    if wid == self.eos_id or fid == last_frame_id:
                        s = scores[fid][i] + self.length_penalty * (fid + 1)
                        if s > max_score:
                            max_score = s
                            frame_id = fid
                            pos_in_frame = i
            if frame_id == -1: # prediction is empty
                traces['pred_seq'].append([0])
            else:
                seq = [wids_list[frame_id][pos_in_frame]]
                for fid in range(frame_id, 0, -1):
                    pos_in_frame = ptrs[fid][pos_in_frame]
                    seq.append(wids_list[fid - 1][pos_in_frame])
                seq.reverse()
                traces['pred_seq'].append(seq)

        def _pad_sequence(sequences, max_len, padding_value=0):
            # sequences: batch_size x output_len x K
            trailing_dims = sequences[0].size()[1:] # K
            out_dims = (len(sequences), max_len) + trailing_dims # batch_size x max_len x K

            out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
            for i, tensor in enumerate(sequences):
                length = tensor.size(0)
                # use index notation to prevent duplicate references to the tensor
                out_tensor[i, :length, ...] = tensor # ... is equivalent to :   ???
            return out_tensor

        # convert to tensors for DataParallel
        for k in ('pred_seq', 'scores', 'wids', 'ptrs'):
            ts_list = traces[k]
            if not isinstance(ts_list[0], torch.Tensor):
                dt = torch.float if k == 'scores' else torch.long
                ts_list = [torch.tensor(it, dtype=dt) for it in ts_list]
            traces[k] = _pad_sequence(
                ts_list, output_length, 0).to(input_ids.device)

        return traces

""" for webqa, based on VLP """
class BertForWebqa(PreTrainedBertModel):
    """refer to BertForPreTraining"""

    def __init__(self, config, num_labels=2, max_len_img_cxt=200):
        super(BertForWebqa, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight, num_labels=num_labels) # num_labels not applicable for VLP
        self.apply(self.init_bert_weights)
        self.crit_mask_lm = nn.CrossEntropyLoss(reduction='none')
        self.crit_filter = nn.CrossEntropyLoss(reduction='none')
        self.num_labels = num_labels
        self.max_len_img_cxt = max_len_img_cxt
        if hasattr(config, 'label_smoothing') and config.label_smoothing:
            self.crit_mask_lm_smoothed = LabelSmoothingLoss(
                config.label_smoothing, config.vocab_size, ignore_index=0, reduction='none')
        else:
            self.crit_mask_lm_smoothed = None

        # will not be initialized when loading BERT weights
        self.vis_embed = nn.Sequential(nn.Linear(2048, 2048),
                                       nn.ReLU(),
                                       nn.Linear(2048, config.hidden_size),
                                       nn.ReLU(),
                                       nn.Dropout(config.hidden_dropout_prob)) # use to be 0.3
        try:
            self.vis_embed[0].weight.data.copy_(torch.from_numpy(pickle.load(
                    open('/home/yingshac/CYS/WebQnA/cpts/detectron_weights/fc7_w.pkl', 'rb'))))
            self.vis_embed[0].bias.data.copy_(torch.from_numpy(pickle.load(
                    open('/home/yingshac/CYS/WebQnA/cpts/detectron_weights/fc7_b.pkl', 'rb'))))
        except:
            raise Exception('Cannot find Detectron fc7 weights! Download from https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/detectron_weights.tar.gz and uncompress under the code root directory.')

        self.vis_pe_embed = nn.Sequential(nn.Linear(6+1601, config.hidden_size),
                                       nn.ReLU(),
                                       nn.Dropout(config.hidden_dropout_prob))
        
        # self.ans_classifier = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size*2),
                                       #nn.ReLU(),
                                       #nn.Linear(config.hidden_size*2, 3129)) # 3129 hard coded...
        self.context_classifier = nn.Linear(config.hidden_size, 2) # each choice gets a single logit
        #self.context_crit = nn.BCEWithLogitsLoss()


    def forward(self, vis_feats=None, vis_pe=None, input_ids=None, token_type_ids=None, attention_mask=None, masked_lm_labels=None, do_filter_task=None, filter_label=None, logit_mask=None, context_is_img=None, next_sentence_label=None, masked_pos=None, masked_weights=None, task_idx=None, drop_worst_ratio=0.2, filter_infr_th=None, tokenizer=None):
        #print("\n")
        #print(context_is_img)
        if context_is_img[0]: 
            vis_feats = self.vis_embed(vis_feats) # image region features Bx100xhidden_size
            vis_pe = self.vis_pe_embed(vis_pe) # image region positional encodings Bx100xhidden_size
        #print("\nattention_mask: ", attention_mask[0][0][230])
        #print("\n", attention_mask[0][0][10])
        '''
        # VQA inference
        if vqa_inference: # vqa_inference=False during training
            assert(ans_labels == None) # in inference mode, need no gth label
            sequence_output, pooled_output = self.bert(vis_feats, vis_pe, input_ids, token_type_ids,
                attention_mask, output_all_encoded_layers=False, len_vis_input=self.len_vis_input)

            vqa2_embed = sequence_output[:, 0]*sequence_output[:, self.len_vis_input+1]
            vqa2_pred = self.ans_classifier(vqa2_embed)
            ans_idx = torch.max(vqa2_pred[:, 1:], -1)[1] + 1
            return ans_idx
        '''

        
        if do_filter_task[0]:
            # input_ids.size() = (B, num_choices, max_len)
            num_choices = input_ids.size(1)
            B = input_ids.size(0)
            assert filter_label is not None
            if context_is_img[0]:
                vis_feats = vis_feats.view(B*num_choices, -1, vis_feats.size(-1))
                vis_pe = vis_pe.view(B*num_choices, -1, vis_pe.size(-1))
            input_ids = input_ids.view(B*num_choices, -1)
            token_type_ids = token_type_ids.view(B*num_choices, -1)
            attention_mask = attention_mask.view(B*num_choices, -1, attention_mask.size(-1))

            def cross_entropy_with_logits_loss(prediction, target, logit_mask):
                # prediction: batch_size x num_choices x 2
                # target: batch_size x num_choices x 2. Targets with multiple flags look like [[0,1], [1,0], [0,1], [0,1], [0,1]] (there is no need to normalize them)
                num_choices = prediction.size(1)
                batch_size = prediction.size(0)
                lp = F.log_softmax(prediction, dim=-1) # batch_size x num_choices x 2
                #num_flags = torch.sum(target, dim=-1)
                #num_flags = torch.max(num_flags, torch.ones_like(num_flags))
                #labels = target / num_flags.unsqueeze(-1).repeat(1, prediction.size(-1))
                #m = lp * labels
                #m = torch.where(torch.isnan(m), torch.zeros_like(m), m)
                #loss = torch.sum(- m, dim=-1) * num_flags
                normalizer = torch.sum(logit_mask, dim=-1)
                m = lp * target * logit_mask.view(-1, num_choices, 1).repeat(1,1,2) # target.transpose --> batch_size x num_choices x 2
                
                #print(normalizer)
                #print(m)
                #print(target)
                loss = (-m).view(batch_size, -1).sum(dim=-1)/(normalizer+1e-8)
                #print(loss)
                return torch.mean(loss)
            def filter_metric(prediction, target, logit_mask, th_list):
                # prediction: batch_size x num_choices x 2
                # target: batch_size x num_choices x 2
                # logit_mask: batch_size x num_choices

                pred = F.softmax(prediction, dim=-1).transpose(2,1)
                #print(pred)
                label = target.transpose(2,1)[:, 0, :] #batch_size x num_choices
                pred = pred[:, 0, :] #batch_size x num_choices
                th_dict = {}
                
                for th in th_list:
                    #time.sleep(1)
                    #print("\nth = ", th)
                    cur_pred = (pred>th).float() * logit_mask
                    overlap = torch.sum(cur_pred * label, dim=-1) # batch_size
                    #print(overlap[0])
                    pr = overlap / (torch.sum(cur_pred, dim=-1) + 1e-5) # batch_size
                    #print(torch.sum(cur_pred, dim=-1)[0])
                    re = overlap / (torch.sum(label, dim=-1) + 1e-5) # batch_size
                    #print(torch.sum(label, dim=-1)[0])
                    f1 = 2*pr*re / (pr+re+1e-5)
                    th_dict[th] = [torch.mean(pr).item(), torch.mean(re).item(), torch.mean(f1).item()]
                return th_dict, pred.detach().cpu()
                '''
                num_flags = torch.sum(target, dim=-1)
                num_flags = torch.max(num_flags, torch.ones_like(num_flags))
                labels = target / num_flags.unsqueeze(-1).repeat(1, prediction.size(-1))
                m = lp * labels
                m = torch.where(torch.isnan(m), torch.zeros_like(m), m)
                loss = torch.sum(- m, dim=-1) * num_flags

                probs = torch.nn.functional.softmax(prediction+logit_mask, dim=-1)
                m = probs * labels
                m = torch.where(torch.isnan(m), torch.zeros_like(m), m)
                metric1 = torch.sum(m, dim=-1)

                values, indices = torch.topk(probs, 2, dim=-1)
                res = torch.zeros_like(prediction)
                try:
                    res = res.scatter(1, indices, torch.ones_like(values))
                except:
                    print(res.size())
                    print(indices.size())
                    print(values.size())
                m = res * labels
                m = torch.where(torch.isnan(m), torch.zeros_like(m), m)
                metric2 = torch.sum(m, dim=-1)
                return torch.mean(loss), torch.mean(metric1), torch.mean(metric2)
                '''

            
            sequence_output, pooled_output = self.bert(vis_feats, vis_pe, input_ids, token_type_ids,\
                                            attention_mask, context_is_img[0], output_all_encoded_layers=False, max_len_img_cxt=self.max_len_img_cxt)
            # calculate classification loss for filter function
            # vqa2_embed = pooled_output
            cls_embed = sequence_output[:, 0] #*sequence_output[:, self.max_len_a+1] 
            # Don't do multiplication for not cuz cxt_meta wasn't padded to fixed length during preprocessing
            cls_pred = self.context_classifier(cls_embed) # B*num_choices x 2
            cls_pred = cls_pred.view(-1, num_choices, 2)
            # cls_labels: B
            if filter_infr_th is not None:
                #assert ori_choices is not None, "In inference mode, must provide ori_choices"
                th_dict, pred = filter_metric(cls_pred, filter_label, logit_mask, filter_infr_th)
                return th_dict, pred
            cls_loss = cross_entropy_with_logits_loss(cls_pred, filter_label, logit_mask)
            masked_lm_loss = cls_loss.new(1).fill_(0)
            return masked_lm_loss, cls_loss

        else:
            assert masked_lm_labels is not None
            sequence_output, pooled_output = self.bert(vis_feats, vis_pe, input_ids, token_type_ids,\
                                            attention_mask, context_is_img[0], output_all_encoded_layers=False, max_len_img_cxt=self.max_len_img_cxt)
            #print("\n", attention_mask[0][10])
            #print("\n", attention_mask[0][107])
            def gather_seq_out_by_pos(seq, pos):
                return torch.gather(seq, 1, pos.unsqueeze(2).expand(-1, -1, seq.size(-1)))

            def mlmloss_mask_and_normalize(loss, mask, drop_worst_ratio):
                mask = mask.type_as(loss) # B x max_pred?
                #filter_mask = torch.ones(mask.size())
                #filter_mask[do_filter_task.nonzero().squeeze(1), :] = 0
                #filter_mask = filter_mask.type_as(loss)
                #print("\nloss before multiplying mask = ", loss)
                loss = loss * mask

                # Ruotian Luo's drop worst (drop batches with worst losses)
                # <less_than_B> x max_pred
                keep_loss, keep_ind = torch.topk(loss.sum(-1), int(loss.size(0)*(1-drop_worst_ratio)), largest=False)
                #print("\nkeep_loss = ", keep_loss)
                # denominator = torch.sum(mask) + 1e-5
                # return (loss / denominator).sum()
                # each batch has different number of actual predictions
                # divide by total num_actual_predictions across all survived batches
                # losses on the placeholder tokens should be zero, still being zero after divided by 1e-5

                try: 
                    denominator = torch.sum(torch.sum(mask, dim=-1)[keep_ind]) + 1e-5
                except:
                    print("keep_ind = ", keep_ind)
                    print("loss.size() = ", loss.size())
                    print("loss = ", loss)
                    raise

                #print("denominator = ", denominator)
                return (keep_loss / denominator).sum() # sum losses over <less_than_B> batches
            '''
            def clfloss_mask_and_normalize(loss, do_filter_task):
                filter_mask = torch.zeros(loss.size())
                filter_mask[do_filter_task.nonzero().squeeze(1)] = 1.
                filter_mask = filter_mask.type_as(loss)
                return (loss*filter_mask).mean()
            '''
            # masked lm
            #print("\n", attention_mask[0][107])
            #print(attention_mask[0][10])
            if torch.isnan(sequence_output).any().item(): print("\nsequence_output is nan !!")
            sequence_output_masked = gather_seq_out_by_pos(sequence_output, masked_pos) # B x max_pred x hidden
            prediction_scores_masked, _ = self.cls(
                sequence_output_masked, pooled_output, task_idx=task_idx) # B x max_pred x vocab_size
            
            if self.crit_mask_lm_smoothed:
                masked_lm_loss = self.crit_mask_lm_smoothed(
                    F.log_softmax(prediction_scores_masked.float(), dim=-1), masked_lm_labels)
            else:
                #print("\nprediction_scores_maksed.transpose(1,2) size = ", prediction_scores_masked.transpose(1, 2).float().size())
                masked_lm_loss = self.crit_mask_lm(
                    prediction_scores_masked.transpose(1, 2).float(), masked_lm_labels)
            if torch.isnan(prediction_scores_masked).any().item(): print("\nprediction_scores_masked is nan !!!!!! ")
            masked_lm_loss = mlmloss_mask_and_normalize(masked_lm_loss.float(), masked_weights, drop_worst_ratio)
            cls_loss= masked_lm_loss.new(1).fill_(0)
            #print("\n ----------------- before returning from forward(), masked_lm_loss = --------------------")
            if tokenizer is not None:
                ids = torch.argmax(prediction_scores_masked[0], dim=-1).detach().cpu().numpy() # max_pred x 1
                sequence = tokenizer.convert_ids_to_tokens([i for i in list(ids) if i>0])
                print(sequence)
                #print(masked_lm_loss.item())
                print(tokenizer.convert_ids_to_tokens([i for i in list(input_ids.detach().cpu().numpy()[0][:]) if i>0]))
                print(tokenizer.convert_ids_to_tokens([i for i in list(masked_lm_labels.detach().cpu().numpy()[0]) if i>0]))
                #print("\n")
                time.sleep(2)
            return masked_lm_loss, cls_loss
        ''' 
        # deprecated
        # vis_feats, vis_pe have been projected to hidden_size dim
        if mask_image_regions:
            # Selfie-like pretext
            # gth img_feats for masked regions
            masked_vis_feats = torch.gather(vis_feats, 1,
                (vis_masked_pos-1).unsqueeze(-1).expand((-1, -1, vis_feats.size(-1))))
            # vis_masked_pos: B x <len_vis_input*vis_mask_prob>
            # unsqueeze: B x <len_vis_input*vis_mask_prob> x 1
            # expand: B x <len_vis_input*vis_mask_prob> x hidden
            # output of this line: B x <len_vis_input*vis_mask_prob> x hidden

            # gth vis_pe for masked regions
            if self.enable_butd:
                masked_pos_enc = torch.gather(vis_pe, 1,
                (vis_masked_pos-1).unsqueeze(-1).expand((-1, -1, vis_pe.size(-1))))
                # B x <len_vis_input*vis_mask_prob> x hidden
            else:
                masked_pos_enc = self.bert.embeddings.position_embeddings(vis_masked_pos)

            # pooled_output: B x hidden
            # unsqueeze: B x 1 x hidden
            # expand_as: B x <len_vis_input*vis_mask_prob> x hidden
            # only pooled_output here gets trained???
            masked_pos_enc += pooled_output.unsqueeze(1).expand_as(masked_pos_enc) # ? Why add pooled_output to gth masked vis pe?
            assert(masked_vis_feats.size() == masked_pos_enc.size())
            # pe of a particular region should be most compatible with vis_feat of that region
            sim_mat = torch.matmul(masked_pos_enc, masked_vis_feats.permute(0, 2, 1).contiguous())
            # B x <len_vis_input*vis_mask_prob> x <len_vis_input*vis_mask_prob>
            sim_mat = F.log_softmax(sim_mat, dim=-1)
            vis_pretext_loss = []
            for i in range(sim_mat.size(0)): # gth is the Identity matrix
                vis_pretext_loss.append(sim_mat[i].diag().mean().view(1)*-1.) # cross entropy for ones
            vis_pretext_loss = torch.cat(vis_pretext_loss).mean() # mean over B batches??? but masked_lm_loss is sum over batches?
        else:
            vis_pretext_loss = masked_lm_loss.new(1).fill_(0)
        '''

        return masked_lm_loss, cls_loss # works better when combined with max_pred=1

""" for webqa, based on VLP """
class BertForWebqaDecoder(PreTrainedBertModel):
    """refer to BertForPreTraining"""

    def __init__(self, config, mask_word_id=0, num_labels=2,
                 search_beam_size=1, length_penalty=1.0, eos_id=0,
                 forbid_duplicate_ngrams=False, forbid_ignore_set=None,
                 ngram_size=3, min_len=0, max_len_img_cxt=200):
        super(BertForWebqaDecoder, self).__init__(config)
        self.bert = BertModelIncr(config)
        self.cls = BertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight, num_labels=num_labels)
        self.apply(self.init_bert_weights)
        self.crit_mask_lm = nn.CrossEntropyLoss(reduction='none')
        self.mask_word_id = mask_word_id
        self.num_labels = num_labels
        self.max_len_img_cxt = max_len_img_cxt
        self.search_beam_size = search_beam_size
        self.length_penalty = length_penalty
        self.eos_id = eos_id
        self.forbid_duplicate_ngrams = forbid_duplicate_ngrams
        self.forbid_ignore_set = forbid_ignore_set
        self.ngram_size = ngram_size
        self.min_len = min_len

        # will not be initialized when loading BERT weights
        self.vis_embed = nn.Sequential(nn.Linear(2048, 2048),
                                       nn.ReLU(),
                                       nn.Linear(2048, config.hidden_size),
                                       nn.ReLU(),
                                       nn.Dropout(config.hidden_dropout_prob))
        self.vis_pe_embed = nn.Sequential(nn.Linear(6+1601, config.hidden_size),
                                       nn.ReLU(),
                                       nn.Dropout(config.hidden_dropout_prob))
    


    def forward(self, vis_feats, vis_pe, input_ids, token_type_ids, position_ids, attention_mask, context_is_img, task_idx=None, sample_mode='greedy', tokenizer=None):
        if context_is_img[0]:
            vis_feats = self.vis_embed(vis_feats) # image region features
            vis_pe = self.vis_pe_embed(vis_pe) # image region positional encodings

        if self.search_beam_size > 1:
            return self.beam_search(vis_feats, vis_pe, input_ids, token_type_ids, position_ids, attention_mask, context_is_img, task_idx)
        input_shape = list(input_ids.size())
        batch_size = input_shape[0]
        input_length = input_shape[1]
        output_shape = list(token_type_ids.size())
        output_length = min(output_shape[1], input_length+30)

        output_ids = []
        output_probs = []
        prev_embedding = None
        prev_encoded_layers = None
        curr_ids = input_ids
        mask_ids = input_ids[:, :1] * 0 + self.mask_word_id # same size as input_ids[:, :1], filled with mask_word_id # batch_size x 1
        next_pos = input_length
        #print(next_pos, output_length)
        while next_pos < output_length:
            
            curr_length = list(curr_ids.size())[1]
            start_pos = next_pos - curr_length
            x_input_ids = torch.cat((curr_ids, mask_ids), dim=1)
            curr_token_type_ids = token_type_ids[:, start_pos:next_pos + 1]
            curr_attention_mask = attention_mask[:,
                start_pos:next_pos + 1, :next_pos + 1]
            curr_position_ids = position_ids[:, start_pos:next_pos + 1]
            new_embedding, new_encoded_layers, _ = \
                self.bert(vis_feats, vis_pe, x_input_ids, curr_token_type_ids, curr_position_ids, curr_attention_mask, 
                context_is_img[0], prev_embedding=prev_embedding, prev_encoded_layers=prev_encoded_layers, 
                output_all_encoded_layers=True, max_len_img_cxt=self.max_len_img_cxt)

            last_hidden = new_encoded_layers[-1][:, -1:, :]
            prediction_scores, _ = self.cls(
                last_hidden, None, task_idx=task_idx) # batch_size x vocab_size
            if tokenizer is not None:

                ids = torch.argmax(prediction_scores[0], dim=-1).detach().cpu().numpy() # batch_size x 1 x 1
                sequence = tokenizer.convert_ids_to_tokens([i for i in list(ids) if i>0 and i!=102])
                print(sequence)
                print(tokenizer.convert_ids_to_tokens([i for i in list(input_ids.detach().cpu().numpy()[0][202:]) if i>0 and i!=102]))
                #print("\n")
                time.sleep(2)

            if sample_mode == 'greedy':
                max_probs, max_ids = torch.max(prediction_scores, dim=-1)
            elif sample_mode == 'sample':
                prediction_scores.squeeze_(1)
                prediction_probs = F.softmax(prediction_scores, dim=-1).detach()
                max_ids = torch.multinomial(prediction_probs, num_samples=1,
                    replacement=True)
                max_probs = torch.gather(F.log_softmax(prediction_scores, dim=-1),
                    1, max_ids) # this should be logprobs
            else:
                raise NotImplementedError
            output_ids.append(max_ids)
            output_probs.append(max_probs)
            if prev_embedding is None:
                prev_embedding = new_embedding[:, :-1, :]
            else:
                prev_embedding = torch.cat(
                    (prev_embedding, new_embedding[:, :-1, :]), dim=1)
            if prev_encoded_layers is None:
                prev_encoded_layers = [x[:, :-1, :]
                                       for x in new_encoded_layers]
            else:
                prev_encoded_layers = [torch.cat((x[0], x[1][:, :-1, :]), dim=1)
                                       for x in zip(prev_encoded_layers, new_encoded_layers)]
            curr_ids = max_ids
            next_pos += 1
        #print(output_probs[-1].size())
        #print(torch.cat(output_ids, dim=1).detach().cpu().numpy()[0])
        #print(output_probs[-1].size())
        return [{s.item():seq} for s, seq in zip(output_probs[-1].detach().cpu().numpy(), torch.cat(output_ids, dim=1).detach().cpu().numpy())]
        #return torch.cat(output_ids, dim=1), torch.cat(output_probs, dim=1)


    def beam_search(self, vis_feats, vis_pe, input_ids, token_type_ids, position_ids, attention_mask, context_is_img, task_idx=None):

        input_shape = list(input_ids.size()) # batch_size x (max_len_a+2)
        batch_size = input_shape[0]
        input_length = input_shape[1]
        output_shape = list(token_type_ids.size()) # batch_size x max_len_in_batch
        output_length = min(output_shape[1], input_length+30)

        output_ids = []
        prev_embedding = None
        prev_encoded_layers = None
        curr_ids = input_ids
        mask_ids = input_ids[:, :1] * 0 + self.mask_word_id
        next_pos = input_length

        K = self.search_beam_size

        total_scores = []
        beam_masks = []
        step_ids = []
        step_back_ptrs = []
        partial_seqs = []
        forbid_word_mask = None
        buf_matrix = None
        #print("\nIn beam search: ", attention_mask[0][-1])
        while next_pos < output_length:
            curr_length = list(curr_ids.size())[1]
            start_pos = next_pos - curr_length
            x_input_ids = torch.cat((curr_ids, mask_ids), dim=1)
            curr_token_type_ids = token_type_ids[:, start_pos:next_pos + 1]
            curr_attention_mask = attention_mask[:,
                                                 start_pos:next_pos + 1, :next_pos + 1]
            curr_position_ids = position_ids[:, start_pos:next_pos + 1]
            new_embedding, new_encoded_layers, _ = \
                self.bert(vis_feats, vis_pe, x_input_ids, curr_token_type_ids, curr_position_ids, curr_attention_mask, 
                context_is_img[0], prev_embedding=prev_embedding, prev_encoded_layers=prev_encoded_layers, 
                output_all_encoded_layers=True, max_len_img_cxt=self.max_len_img_cxt)
            # compared with BertModel, BertModelIncr returns an additional embedding_output from forward()
            # new_encoded_layers == sequence_output

            last_hidden = new_encoded_layers[-1][:, -1:, :] # batch_size(*K) x 1 x emb_dim
            prediction_scores, _ = self.cls(last_hidden, None, task_idx=task_idx) # batch_size(*K) x 1 x vocab_size
            #if prev_embedding is None:
                #print("\n", prediction_scores[0][:20])
            log_scores = torch.nn.functional.log_softmax(prediction_scores, dim=-1) # batch_size(*K) x 1 x vocab_size
            if forbid_word_mask is not None:
                log_scores += (forbid_word_mask * -10000.0) 
                # forbid_word_mask has the same size as log_scores, positions with 1 indicate forbid_words
            if self.min_len and (next_pos-input_length+1 <= self.min_len):
                log_scores[:, :, self.eos_id].fill_(-10000.0) # forbid generating <eos> when min_len is not reached
            kk_scores, kk_ids = torch.topk(log_scores, k=K) # batch_size(*K) x 1 x K
            if len(total_scores) == 0:
                k_ids = torch.reshape(kk_ids, [batch_size, K])
                back_ptrs = torch.zeros(batch_size, K, dtype=torch.long)
                k_scores = torch.reshape(kk_scores, [batch_size, K])
            else:
                last_eos = torch.reshape(
                    beam_masks[-1], [batch_size * K, 1, 1])
                last_seq_scores = torch.reshape(
                    total_scores[-1], [batch_size * K, 1, 1])
                kk_scores += last_eos * (-10000.0) + last_seq_scores # don't consider beams that already have <eos> in the last step
                kk_scores = torch.reshape(kk_scores, [batch_size, K * K])
                k_scores, k_ids = torch.topk(kk_scores, k=K) # sample K from K*K # batch_size x K
                back_ptrs = torch.div(k_ids, K) # batch_size x K (choose top K from K*K)
                kk_ids = torch.reshape(kk_ids, [batch_size, K * K])
                k_ids = torch.gather(kk_ids, 1, k_ids) # batch_size x K    top K idx of kk_scores --> top K vocab idx
            step_back_ptrs.append(back_ptrs) # record prev_step beam id. get the actual word_id from corresponding pos in step_ids
            step_ids.append(k_ids)
            beam_masks.append(torch.eq(k_ids, self.eos_id).float()) # block beams that reach <eos> at an early point
            # stop beam if <eos> is generated
            # but that won't happen bc <eos> is masked in log_scores if output_len < min_len ???
            total_scores.append(k_scores)

            def first_expand(x):
                input_shape = list(x.size())
                expanded_shape = input_shape[:1] + [1] + input_shape[1:]
                x = torch.reshape(x, expanded_shape) # batch_size x 1 x ori_len x emb_dim
                repeat_count = [1, K] + [1] * (len(input_shape) - 1) # [1, K, 1, 1] 
                x = x.repeat(*repeat_count) # --> batch_size x K x ori_len x emb_dim
                x = torch.reshape(x, [input_shape[0] * K] + input_shape[1:]) # (batch_size * K) x ori_len x emb_dim
                return x

            def select_beam_items(x, ids):
                # ids:
                # batch_size x K (choose top K from K*K)
                id_shape = list(ids.size())
                id_rank = len(id_shape)
                assert len(id_shape) == 2
                x_shape = list(x.size())
                x = torch.reshape(x, [batch_size, K] + x_shape[1:]) # batch_size x K x (ori_len+1) x emb_dim
                x_rank = len(x_shape) + 1
                assert x_rank >= 2
                if id_rank < x_rank:
                    ids = torch.reshape(
                        ids, id_shape + [1] * (x_rank - id_rank)) # batch_size x K x 1 x 1
                    ids = ids.expand(id_shape + x_shape[1:]) # batch_size x K x (ori_len+1) x emb_dim
                y = torch.gather(x, 1, ids) # batch_size x K x (ori_len+1) x emb_dim
                y = torch.reshape(y, x_shape) # batch_size*K x (ori_len+1) x emb_dim
                return y

            is_first = (prev_embedding is None)

            if prev_embedding is None:
                prev_embedding = first_expand(new_embedding[:, :-1, :]) 
                # ori_len = length before appending mask
                # batch_size x ori_len x emb_dim 
            else:
                prev_embedding = torch.cat(
                    (prev_embedding, new_embedding[:, :-1, :]), dim=1) # new_embedding[:, :-1, :].size() = batch_size*K x 1 x emb_dim
                    # --> batch_size*K x (ori_len+1) x emb_dim
                prev_embedding = select_beam_items(prev_embedding, back_ptrs)
            if prev_encoded_layers is None:
                # new_encoded_layers: num_layers x batch_size x ori_len x emb_dim
                prev_encoded_layers = [first_expand(
                    x[:, :-1, :]) for x in new_encoded_layers]
            else:
                prev_encoded_layers = [torch.cat((x[0], x[1][:, :-1, :]), dim=1)
                                       for x in zip(prev_encoded_layers, new_encoded_layers)]
                prev_encoded_layers = [select_beam_items(
                    x, back_ptrs) for x in prev_encoded_layers]

            curr_ids = torch.reshape(k_ids, [batch_size * K, 1])

            if is_first:
                token_type_ids = first_expand(token_type_ids)
                position_ids = first_expand(position_ids)
                attention_mask = first_expand(attention_mask)
                mask_ids = first_expand(mask_ids)

            if self.forbid_duplicate_ngrams:
                wids = step_ids[-1].tolist()
                ptrs = step_back_ptrs[-1].tolist()
                if is_first:
                    partial_seqs = []
                    for b in range(batch_size):
                        for k in range(K):
                            partial_seqs.append([wids[b][k]])
                else:
                    new_partial_seqs = []
                    for b in range(batch_size):
                        for k in range(K):
                            new_partial_seqs.append(
                                partial_seqs[ptrs[b][k] + b * K] + [wids[b][k]])
                    partial_seqs = new_partial_seqs

                def get_dup_ngram_candidates(seq, n):
                    cands = set()
                    if len(seq) < n:
                        return []
                    tail = seq[-(n-1):]
                    if self.forbid_ignore_set and any(tk in self.forbid_ignore_set for tk in tail):
                        return []
                    for i in range(len(seq) - (n - 1)):
                        mismatch = False
                        for j in range(n - 1):
                            if tail[j] != seq[i + j]:
                                mismatch = True
                                break
                        if (not mismatch) and not(self.forbid_ignore_set and (seq[i + n - 1] in self.forbid_ignore_set)):
                            cands.add(seq[i + n - 1])
                    return list(sorted(cands))

                if len(partial_seqs[0]) >= self.ngram_size:
                    dup_cands = []
                    for seq in partial_seqs:
                        dup_cands.append(
                            get_dup_ngram_candidates(seq, self.ngram_size))
                    if max(len(x) for x in dup_cands) > 0:
                        if buf_matrix is None:
                            vocab_size = list(log_scores.size())[-1]
                            buf_matrix = np.zeros(
                                (batch_size * K, vocab_size), dtype=float)
                        else:
                            buf_matrix.fill(0)
                        for bk, cands in enumerate(dup_cands):
                            for i, wid in enumerate(cands):
                                buf_matrix[bk, wid] = 1.0
                        forbid_word_mask = torch.tensor(
                            buf_matrix, dtype=log_scores.dtype)
                        forbid_word_mask = torch.reshape(
                            forbid_word_mask, [batch_size * K, 1, vocab_size]).cuda()
                    else:
                        forbid_word_mask = None
            next_pos += 1

        # output_len x batch_size x K
        total_scores = [x.tolist() for x in total_scores] # each x: batch_size x K
        step_ids = [x.tolist() for x in step_ids]
        step_back_ptrs = [x.tolist() for x in step_back_ptrs]
        # back tracking
        #traces = {'pred_seq': [], 'scores': [], 'wids': [], 'ptrs': []}
        traces = [] # list of dict {scores: pred_seq}. Will be sorted by score.
        for b in range(batch_size):
            traces.append({})
            scores = [x[b] for x in total_scores] # output_len x K 
            wids_list = [x[b] for x in step_ids]  # output_len x K 
            ptrs = [x[b] for x in step_back_ptrs] # output_len x K 
            #traces['scores'].append(scores) # 这些跟 total_scores, step_ids, step_back_ptrs有什么区别啊。。。又重新一个个append一遍。。。
            #traces['wids'].append(wids_list)
            #traces['ptrs'].append(ptrs)
            
            # first we need to find the eos frame where all symbols are eos
            # any frames after the eos frame are invalid
            last_frame_id = len(scores) - 1 # last id within output_len
            for i, wids in enumerate(wids_list): 
                # if all topK beams reach <eos> before reaching output_len, then set last_frame_id to an earlier frame.
                if all(wid == self.eos_id for wid in wids):
                    last_frame_id = i
                    break
            #max_score = -math.inf
            #frame_id = -1
            #pos_in_frame = -1

            score2pos = {}
            used_pos = []
            for fid in range(last_frame_id + 1):
                for i, wid in enumerate(wids_list[fid]):
                    if (wid == self.eos_id or fid == last_frame_id) and not i in used_pos:
                        s = scores[fid][i] + self.length_penalty * (fid + 1)
                        if s > -math.inf:
                            score2pos[s] = (fid, i) # (frame_id, pos_in_frame)
                            used_pos.append(i)
                #if len(score2pos)>0: break
                        #if s > max_score:
                            #max_score = s
                            #frame_id = fid
                            #pos_in_frame = i
            
            for s in sorted(score2pos.keys(), reverse=True):
                frame_id = score2pos[s][0]
                pos_in_frame = score2pos[s][1]
                seq = [wids_list[frame_id][pos_in_frame]]
                for fid in range(frame_id, 0, -1):
                    pos_in_frame = ptrs[fid][pos_in_frame]
                    seq.append(wids_list[fid - 1][pos_in_frame])
                seq.reverse()
                traces[-1][s] = seq
            if len(score2pos) < self.search_beam_size: # create placeholder so that #predictions = K
                n_placeholder = self.search_beam_size-len(score2pos)
                for s in range(-10000, -10000-n_placeholder, -1):
                    traces[-1][s] = [0]

                #seq = [wids_list[frame_id][pos_in_frame]]
                #for fid in range(frame_id, 0, -1):
                    #pos_in_frame = ptrs[fid][pos_in_frame]
                    #seq.append(wids_list[fid - 1][pos_in_frame])
                #seq.reverse()
                #traces['pred_seq'].append(seq)

        def _pad_sequence(sequences, max_len, padding_value=0):
            # sequences: batch_size x output_len x K
            trailing_dims = sequences[0].size()[1:] # K
            out_dims = (len(sequences), max_len) + trailing_dims # batch_size x max_len x K

            out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
            for i, tensor in enumerate(sequences):
                length = tensor.size(0)
                # use index notation to prevent duplicate references to the tensor
                out_tensor[i, :length, ...] = tensor # ... is equivalent to :   ???
            return out_tensor

        '''
        # convert to tensors for DataParallel
        for k in ('pred_seq', 'scores', 'wids', 'ptrs'):
            ts_list = traces[k]
            if not isinstance(ts_list[0], torch.Tensor):
                dt = torch.float if k == 'scores' else torch.long
                ts_list = [torch.tensor(it, dtype=dt) for it in ts_list]
            traces[k] = _pad_sequence(
                ts_list, output_length, 0).to(input_ids.device)
        '''
        return traces


class BertForExtractiveSummarization(PreTrainedBertModel):
    """refer to BertForPreTraining"""

    def __init__(self, config):
        super(BertForExtractiveSummarization, self).__init__(config)
        self.bert = BertModel(config)
        self.secondary_pred_proj = nn.Embedding(2, config.hidden_size)
        self.cls2 = BertPreTrainingHeads(
            config, self.secondary_pred_proj.weight, num_labels=2)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_pos_2=None, masked_weights_2=None, task_idx=None):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                                   output_all_encoded_layers=False)

        def gather_seq_out_by_pos(seq, pos):
            return torch.gather(seq, 1, pos.unsqueeze(2).expand(-1, -1, seq.size(-1)))

        sequence_output_masked_2 = gather_seq_out_by_pos(
            sequence_output, masked_pos_2)
        prediction_scores_masked_2, _ = self.cls2(
            sequence_output_masked_2, None, task_idx=task_idx)

        predicted_probs = torch.nn.functional.softmax(
            prediction_scores_masked_2, dim=-1)

        return predicted_probs, masked_pos_2, masked_weights_2


class BertForMaskedLM(PreTrainedBertModel):
    """BERT model with the masked language modeling head.
    This module comprises the BERT model followed by the masked language modeling head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]

    Outputs:
        if `masked_lm_labels` is `None`:
            Outputs the masked language modeling loss.
        if `masked_lm_labels` is `None`:
            Outputs the masked language modeling logits of shape [batch_size, sequence_length, vocab_size].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForMaskedLM(config)
    masked_lm_logits_scores = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(BertForMaskedLM, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(
            config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask,
                                       output_all_encoded_layers=False)
        prediction_scores = self.cls(sequence_output)

        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            return masked_lm_loss
        else:
            return prediction_scores


class BertForNextSentencePrediction(PreTrainedBertModel):
    """BERT model with next sentence prediction head.
    This module comprises the BERT model followed by the next sentence classification head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `next_sentence_label`: next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.

    Outputs:
        if `next_sentence_label` is not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `next_sentence_label` is `None`:
            Outputs the next sentence classification logits of shape [batch_size, 2].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForNextSentencePrediction(config)
    seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(BertForNextSentencePrediction, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyNSPHead(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, next_sentence_label=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                     output_all_encoded_layers=False)
        seq_relationship_score = self.cls(pooled_output)

        if next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            next_sentence_loss = loss_fct(
                seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            return next_sentence_loss
        else:
            return seq_relationship_score


class BertForSequenceClassification(PreTrainedBertModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
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

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config, num_labels=2):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            if labels.dtype == torch.long:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            elif labels.dtype == torch.half or labels.dtype == torch.float:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                print('unkown labels.dtype')
                loss = None
            return loss
        else:
            return logits


class BertForMultipleChoice(PreTrainedBertModel):
    """BERT model for multiple choice tasks.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_choices`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, num_choices, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, num_choices, sequence_length]
            with the token types indices selected in [0, 1]. Type 0 corresponds to a `sentence A`
            and type 1 corresponds to a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, num_choices, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_choices].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[[31, 51, 99], [15, 5, 0]], [[12, 16, 42], [14, 28, 57]]])
    input_mask = torch.LongTensor([[[1, 1, 1], [1, 1, 0]],[[1,1,0], [1, 0, 0]]])
    token_type_ids = torch.LongTensor([[[0, 0, 1], [0, 1, 0]],[[0, 1, 1], [0, 0, 1]]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_choices = 2

    model = BertForMultipleChoice(config, num_choices)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config, num_choices=2):
        super(BertForMultipleChoice, self).__init__(config)
        self.num_choices = num_choices
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        _, pooled_output = self.bert(
            flat_input_ids, flat_token_type_ids, flat_attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, self.num_choices)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            return loss
        else:
            return reshaped_logits


class BertForTokenClassification(PreTrainedBertModel):
    """BERT model for token-level classification.
    This module is composed of the BERT model with a linear layer on top of
    the full hidden state of the last layer.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
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

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, sequence_length, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForTokenClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config, num_labels=2):
        super(BertForTokenClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        sequence_output, _ = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class BertForQuestionAnswering(PreTrainedBertModel):
    """BERT model for Question Answering (span extraction).
    This module is composed of the BERT model with a linear layer on top of
    the sequence output that computes start_logits and end_logits

    Params:
        `config`: either
            - a BertConfig class instance with the configuration to build a new model, or
            - a str with the name of a pre-trained model to load selected in the list of:
                . `bert-base-uncased`
                . `bert-large-uncased`
                . `bert-base-cased`
                . `bert-base-multilingual`
                . `bert-base-chinese`
                The pre-trained model will be downloaded and cached if needed.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `start_positions`: position of the first token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.
        `end_positions`: position of the last token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.

    Outputs:
        if `start_positions` and `end_positions` are not `None`:
            Outputs the total_loss which is the sum of the CrossEntropy loss for the start and end token positions.
        if `start_positions` or `end_positions` is `None`:
            Outputs a tuple of start_logits, end_logits which are the logits respectively for the start and end
            position tokens of shape [batch_size, sequence_length].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForQuestionAnswering(config)
    start_logits, end_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(BertForQuestionAnswering, self).__init__(config)
        self.bert = BertModel(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None, end_positions=None):
        sequence_output, _ = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss
        else:
            return start_logits, end_logits
