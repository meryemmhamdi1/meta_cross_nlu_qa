from transformers import BertTokenizer, BertModel, BertConfig, BertForQuestionAnswering, \
    OpenAIGPTTokenizer, OpenAIGPTModel, GPT2Tokenizer, GPT2Model, \
    CTRLTokenizer, CTRLModel, TransfoXLTokenizer, TransfoXLModel, \
    XLNetTokenizer, XLNetModel, XLNetConfig, XLNetForQuestionAnswering, \
    XLMTokenizer, XLMModel, XLMConfig, XLMForQuestionAnswering, \
    RobertaTokenizer, RobertaModel, XLMRobertaTokenizer, \
    DistilBertTokenizer, DistilBertModel, DistilBertConfig, DistilBertForQuestionAnswering,  \
    AlbertTokenizer, AlbertModel, AlbertConfig, AlbertForQuestionAnswering


import logging

from transformers.configuration_roberta import RobertaConfig
from transformers.file_utils import add_start_docstrings
#from roberta import (
#    RobertaForMaskedLM,
#    RobertaForMultipleChoice,
#    RobertaForSequenceClassification,
#    RobertaForTokenClassification,
#    RobertaForQuestionAnswering,
#    RobertaModel,
#)

## Optimization
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    get_linear_schedule_with_warmup,
)

## SQUAD Metrics
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
)

logger = logging.getLogger(__name__)

# coding=utf-8
# Copyright 2019 Facebook AI Research and the HuggingFace Inc. team.
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
"""PyTorch XLM-RoBERTa model. """

XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "xlm-roberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-base-config.json",
    "xlm-roberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-large-config.json",
    "xlm-roberta-large-finetuned-conll02-dutch": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-large-finetuned-conll02-dutch-config.json",
    "xlm-roberta-large-finetuned-conll02-spanish": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-large-finetuned-conll02-spanish-config.json",
    "xlm-roberta-large-finetuned-conll03-english": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-large-finetuned-conll03-english-config.json",
    "xlm-roberta-large-finetuned-conll03-german": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-large-finetuned-conll03-german-config.json",
}

XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "xlm-roberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-base-pytorch_model.bin",
    "xlm-roberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-large-pytorch_model.bin",
    "xlm-roberta-large-finetuned-conll02-dutch": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-large-finetuned-conll02-dutch-pytorch_model.bin",
    "xlm-roberta-large-finetuned-conll02-spanish": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-large-finetuned-conll02-spanish-pytorch_model.bin",
    "xlm-roberta-large-finetuned-conll03-english": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-large-finetuned-conll03-english-pytorch_model.bin",
    "xlm-roberta-large-finetuned-conll03-german": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-large-finetuned-conll03-german-pytorch_model.bin",
}

XLM_ROBERTA_START_DOCSTRING = r"""    The XLM-RoBERTa model was proposed in
    `Unsupervised Cross-lingual Representation Learning at Scale`_
    by Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer and Veselin Stoyanov. It is based on Facebook's RoBERTa model released in 2019.

    It is a large multi-lingual language model, trained on 2.5TB of filtered CommonCrawl data.

    This implementation is the same as RoBERTa.

    This model is a PyTorch `torch.nn.Module`_ sub-class. Use it as a regular PyTorch Module and
    refer to the PyTorch documentation for all matter related to general usage and behavior.

    .. _`Unsupervised Cross-lingual Representation Learning at Scale`:
        https://arxiv.org/abs/1911.02116

    .. _`torch.nn.Module`:
        https://pytorch.org/docs/stable/nn.html#module

    Parameters:
        config (:class:`~transformers.XLMRobertaConfig`): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

XLM_ROBERTA_INPUTS_DOCSTRING = r"""
    Inputs:
        **input_ids**: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of input sequence tokens in the vocabulary.
            To match pre-training, XLM-RoBERTa input sequence should be formatted with <s> and </s> tokens as follows:

            (a) For sequence pairs:

                ``tokens:         <s> Is this Jacksonville ? </s> </s> No it is not . </s>``

            (b) For single sequences:

                ``tokens:         <s> the dog is hairy . </s>``

            Fully encoded sequences or sequence pairs can be obtained using the XLMRobertaTokenizer.encode function with
            the ``add_special_tokens`` parameter set to ``True``.

            XLM-RoBERTa is a model with absolute position embeddings so it's usually advised to pad the inputs on
            the right rather than the left.

            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
        **attention_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        **token_type_ids**: (`optional` need to be trained) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Optional segment token indices to indicate first and second portions of the inputs.
            This embedding matrice is not trained (not pretrained during XLM-RoBERTa pretraining), you will have to train it
            during finetuning.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token
            (see `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`_ for more details).
        **position_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1[``.
        **head_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``:
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
        **inputs_embeds**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, embedding_dim)``:
            Optionally, instead of passing ``input_ids`` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
"""

from xlm_roberta import XLMRobertaForQuestionAnswering, XLMRobertaConfig


MODELS_dict = {"BertLarge": ('bert-large-uncased', BertTokenizer, BertModel, BertConfig, BertForQuestionAnswering),
               "BertBaseCased": ('bert-base-cased', BertTokenizer, BertModel, BertConfig, BertForQuestionAnswering),
               "BertBaseMultilingualCased": ('bert-base-multilingual-cased', BertTokenizer, BertModel, BertConfig, BertForQuestionAnswering),
               "Xlnet_base": ('xlnet-base-cased', XLNetTokenizer, XLNetModel, XLNetConfig, XLNetForQuestionAnswering),
               "Xlnet_large": ('xlnet-large-cased', XLNetTokenizer, XLNetModel, XLNetConfig, XLNetForQuestionAnswering),
               "XLM": ('xlm-mlm-enfr-1024', XLMTokenizer, XLMModel, XLMConfig, XLMForQuestionAnswering),
               "DistilBert_base": ('distilbert-base-uncased', DistilBertTokenizer, DistilBertModel, DistilBertConfig, DistilBertForQuestionAnswering),
               "DistilBert_large": ('distilbert-large-cased', DistilBertTokenizer, DistilBertModel, DistilBertConfig, DistilBertForQuestionAnswering),
               "ALBERT-base-v1": ('albert-base-v1', AlbertTokenizer, AlbertModel, AlbertConfig, AlbertForQuestionAnswering),
               "ALBERT-large-v1": ('albert-large-v1', AlbertTokenizer, AlbertModel, AlbertConfig, AlbertForQuestionAnswering),
               "ALBERT-xlarge-v1": ('albert-xlarge-v1', AlbertTokenizer, AlbertModel, AlbertConfig, AlbertForQuestionAnswering),
               "ALBERT-xxlarge-v1": ('albert-xxlarge-v1', AlbertTokenizer, AlbertModel, AlbertConfig, AlbertForQuestionAnswering),
               "ALBERT-base-v2": ( 'albert-base-v2', AlbertTokenizer, AlbertModel, AlbertConfig, AlbertForQuestionAnswering),
               "ALBERT-large-v2": ('albert-large-v2', AlbertTokenizer, AlbertModel, AlbertConfig, AlbertForQuestionAnswering),
               "ALBERT-xlarge-v2": ('albert-xlarge-v2', AlbertTokenizer, AlbertModel, AlbertConfig, AlbertForQuestionAnswering),
               "ALBERT-xxlarge-v2": ('albert-xxlarge-v2', AlbertTokenizer, AlbertModel, AlbertConfig, AlbertForQuestionAnswering),
               "Roberta_base": ('roberta-base', RobertaTokenizer, RobertaModel),
               "Roberta_large": ('roberta-large', RobertaTokenizer, RobertaModel),
               "XLMRoberta_base": ('xlm-roberta-base', XLMRobertaTokenizer, XLMRobertaForQuestionAnswering, XLMRobertaConfig, XLMRobertaForQuestionAnswering),
               #"XLMRoberta_large": ('xlm-roberta-large', XLMRobertaTokenizer, XLMRobertaModel, XLMRobertaConfig, XLMRobertaModel)
               }

