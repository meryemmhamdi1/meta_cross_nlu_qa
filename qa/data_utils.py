# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors,
# The HuggingFace Inc. team, and The XTREME Benchmark Authors.
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

"""Taken from https://github.com/google-research/xtreme"""
import json
import logging
import os
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm

from transformers.file_utils import is_tf_available, is_torch_available
from transformers.tokenization_bert import whitespace_tokenize
from transformers import DataProcessor
import torch.nn as nn

if is_torch_available():
    import torch
    from torch.utils.data import TensorDataset

from torch import LongTensor
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    pass
import scipy.spatial as sp
import random

if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text, lang='en', lang2id=None):
    """Returns tokenized answer spans that better match the annotated answer."""
    if lang2id is None:
        tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))
    else:
        tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text, lang=lang))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start : (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def _new_check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    # if len(doc_spans) == 1:
    # return True
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span["start"] + doc_span["length"] - 1
        if position < doc_span["start"]:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span["start"]
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span["length"]
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def squad_convert_example_to_features(example, max_seq_length, doc_stride, max_query_length, is_training, lang2id):
    features = []
    if is_training and not example.is_impossible:
        # Get start and end position
        start_position = example.start_position
        end_position = example.end_position

        # If the answer cannot be found in the text, then skip this example.
        actual_text = " ".join(example.doc_tokens[start_position : (end_position + 1)])
        cleaned_answer_text = " ".join(whitespace_tokenize(example.answer_text))
        if actual_text.find(cleaned_answer_text) == -1:
            #logger.warning("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
            return []

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        if lang2id is None:
            sub_tokens = tokenizer.tokenize(token)
        else:
            sub_tokens = tokenizer.tokenize(token, lang=example.language)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    if is_training and not example.is_impossible:
        tok_start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1

        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer, example.answer_text,
            lang=example.language, lang2id=lang2id
        )

    spans = []

    truncated_query = tokenizer.encode(example.question_text, add_special_tokens=False, max_length=max_query_length)
    sequence_added_tokens = (
        tokenizer.max_len - tokenizer.max_len_single_sentence + 1
        if "roberta" in str(type(tokenizer))
        else tokenizer.max_len - tokenizer.max_len_single_sentence
    )
    sequence_pair_added_tokens = tokenizer.max_len - tokenizer.max_len_sentences_pair

    span_doc_tokens = all_doc_tokens
    while len(spans) * doc_stride < len(all_doc_tokens):
        encoded_dict = tokenizer.encode_plus(
            truncated_query if tokenizer.padding_side == "right" else span_doc_tokens,
            span_doc_tokens if tokenizer.padding_side == "right" else truncated_query,
            max_length=max_seq_length,
            return_overflowing_tokens=True,
            pad_to_max_length=True,
            truncation=True,
            stride=max_seq_length - doc_stride - len(truncated_query) - sequence_pair_added_tokens,
            truncation_strategy="only_second" if tokenizer.padding_side == "right" else "only_first",
            return_token_type_ids=True
        )

        paragraph_len = min(
            len(all_doc_tokens) - len(spans) * doc_stride,
            max_seq_length - len(truncated_query) - sequence_pair_added_tokens,
            )

        if tokenizer.pad_token_id in encoded_dict["input_ids"]:
            non_padded_ids = encoded_dict["input_ids"][: encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
        else:
            non_padded_ids = encoded_dict["input_ids"]

        tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

        token_to_orig_map = {}
        for i in range(paragraph_len):
            index = len(truncated_query) + sequence_added_tokens + i if tokenizer.padding_side == "right" else i
            token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + i]

        encoded_dict["paragraph_len"] = paragraph_len
        encoded_dict["tokens"] = tokens
        encoded_dict["token_to_orig_map"] = token_to_orig_map
        encoded_dict["truncated_query_with_special_tokens_length"] = len(truncated_query) + sequence_added_tokens
        encoded_dict["token_is_max_context"] = {}
        encoded_dict["start"] = len(spans) * doc_stride
        encoded_dict["length"] = paragraph_len
        encoded_dict["rankings_spt"] = example.rankings_spt
        encoded_dict["rankings_qry"] = example.rankings_qry

        spans.append(encoded_dict)

        if "overflowing_tokens" not in encoded_dict:
            break
        span_doc_tokens = encoded_dict["overflowing_tokens"]

    for doc_span_index in range(len(spans)):
        for j in range(spans[doc_span_index]["paragraph_len"]):
            is_max_context = _new_check_is_max_context(spans, doc_span_index, doc_span_index * doc_stride + j)
            index = (
                j
                if tokenizer.padding_side == "left"
                else spans[doc_span_index]["truncated_query_with_special_tokens_length"] + j
            )
            spans[doc_span_index]["token_is_max_context"][index] = is_max_context

    for span in spans:
        # Identify the position of the CLS token
        cls_index = span["input_ids"].index(tokenizer.cls_token_id)

        # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
        # Original TF implem also keep the classification token (set to 0) (not sure why...)
        p_mask = np.array(span["token_type_ids"])

        p_mask = np.minimum(p_mask, 1)

        if tokenizer.padding_side == "right":
            # Limit positive values to one
            p_mask = 1 - p_mask

        p_mask[np.where(np.array(span["input_ids"]) == tokenizer.sep_token_id)[0]] = 1

        # Set the CLS index to '0'
        p_mask[cls_index] = 0

        span_is_impossible = example.is_impossible
        start_position = 0
        end_position = 0
        if is_training and not span_is_impossible:
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            doc_start = span["start"]
            doc_end = span["start"] + span["length"] - 1
            out_of_span = False

            if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                out_of_span = True

            if out_of_span:
                start_position = cls_index
                end_position = cls_index
                span_is_impossible = True
            else:
                if tokenizer.padding_side == "left":
                    doc_offset = 0
                else:
                    doc_offset = len(truncated_query) + sequence_added_tokens

                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset

        if lang2id is not None:
            lid = lang2id.get(example.language, lang2id["en"])
        else:
            lid = 0
        langs = [lid] * max_seq_length

        features.append(
            SquadFeatures(
                span["input_ids"],
                span["attention_mask"],
                span["token_type_ids"],
                span["rankings_spt"],
                span["rankings_qry"],
                cls_index,
                p_mask.tolist(),
                example_index=0,  # Can not set unique_id and example_index here. They will be set after multiple processing.
                unique_id=0,
                paragraph_len=span["paragraph_len"],
                token_is_max_context=span["token_is_max_context"],
                tokens=span["tokens"],
                token_to_orig_map=span["token_to_orig_map"],
                start_position=start_position,
                end_position=end_position,
                langs=langs
            )
        )

    return features


def squad_convert_example_to_features_init(tokenizer_for_convert):
    global tokenizer
    tokenizer = tokenizer_for_convert


def squad_convert_examples_to_features(
        examples, tokenizer, max_seq_length, doc_stride, max_query_length, is_training, return_dataset=False, threads=1,
        lang2id=None):
    """
    Converts a list of examples into a list of features that can be directly given as input to a model.
    It is model-dependant and takes advantage of many of the tokenizer's features to create the model's inputs.

    Args:
        examples: list of :class:`~transformers.data.processors.squad.SquadExample`
        tokenizer: an instance of a child of :class:`~transformers.PreTrainedTokenizer`
        max_seq_length: The maximum sequence length of the inputs.
        doc_stride: The stride used when the context is too large and is split across several features.
        max_query_length: The maximum length of the query.
        is_training: whether to create features for model evaluation or model training.
        return_dataset: Default False. Either 'pt' or 'tf'.
            if 'pt': returns a torch.data.TensorDataset,
            if 'tf': returns a tf.data.Dataset
        threads: multiple processing threadsa-smi


    Returns:
        list of :class:`~transformers.data.processors.squad.SquadFeatures`

    Example::

        processor = SquadV2Processor()
        examples = processor.get_dev_examples(data_dir)

        features = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
        )
    """

    # Defining helper methods
    features = []
    count_not_found = 0
    count_found = 0
    squad_convert_example_to_features_init(tokenizer)
    print("====> Examples .....")
    for example in tqdm(examples):
        example_results = squad_convert_example_to_features(example, max_seq_length, doc_stride, max_query_length,
                                                            is_training, lang2id)
        features.append(example_results)

        if len(example_results) == 0:
            count_not_found += 1
        else:
            count_found += 1

    new_features = []
    unique_id = 1000000000
    example_index = 0
    print("====> Features .....")
    for example_features in tqdm(features): #tqdm(features, total=len(features), desc="add example index and unique id"):
        if not example_features:
            continue
        for example_feature in example_features:
            example_feature.example_index = example_index
            example_feature.unique_id = unique_id
            new_features.append(example_feature)
            unique_id += 1
        example_index += 1
    features = new_features
    del new_features
    if return_dataset == "pt":
        if not is_torch_available():
            raise RuntimeError("PyTorch must be installed to return a PyTorch dataset.")

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
        all_langs = torch.tensor([f.langs for f in features], dtype=torch.long)

        if not is_training:
            all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids, all_attention_masks, all_token_type_ids, all_example_index, all_cls_index, all_p_mask, all_langs
            )
        else:
            all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
            all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids,
                all_attention_masks,
                all_token_type_ids,
                all_start_positions,
                all_end_positions,
                all_cls_index,
                all_p_mask,
                all_langs
            )

        return features, dataset
    elif return_dataset == "tf":
        if not is_tf_available():
            raise RuntimeError("TensorFlow must be installed to return a TensorFlow dataset.")

        def gen():
            for ex in features:
                yield (
                    {
                        "input_ids": ex.input_ids,
                        "attention_mask": ex.attention_mask,
                        "token_type_ids": ex.token_type_ids,
                        "rankings": ex.rankings,
                    },
                    {
                        "start_position": ex.start_position,
                        "end_position": ex.end_position,
                        "cls_index": ex.cls_index,
                        "p_mask": ex.p_mask,
                        "rankings": ex.rankings,
                    },
                )

        return tf.data.Dataset.from_generator(
            gen,
            (
                {"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32, "rankings": tf.int32},
                {"start_position": tf.int64, "end_position": tf.int64, "cls_index": tf.int64, "p_mask": tf.int32,
                 "rankings": tf.int32},
            ),
            (
                {
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "token_type_ids": tf.TensorShape([None]),
                    "rankings": tf.TensorShape([None])
                },
                {
                    "start_position": tf.TensorShape([]),
                    "end_position": tf.TensorShape([]),
                    "cls_index": tf.TensorShape([]),
                    "p_mask": tf.TensorShape([None]),
                    "rankings": tf.TensorShape([])
                },
            ),
        )

    return features


def meta_squad_convert_examples_to_features(spt_examples, qry_examples, tokenizer, max_seq_length, doc_stride,
                                            max_query_length, is_training, opt_config, data_config,
                                            return_dataset=False, threads=1, lang2id=None):
    """
    Converts a list of examples into a list of features that can be directly given as input to a model.
    It is model-dependant and takes advantage of many of the tokenizer's features to create the model's inputs.

    Args:
        examples: list of :class:`~transformers.data.processors.squad.SquadExample`
        tokenizer: an instance of a child of :class:`~transformers.PreTrainedTokenizer`
        max_seq_length: The maximum sequence length of the inputs.
        doc_stride: The stride used when the context is too large and is split across several features.
        max_query_length: The maximum length of the query.
        is_training: whether to create features for model evaluation or model training.
        return_dataset: Default False. Either 'pt' or 'tf'.
            if 'pt': returns a torch.data.TensorDataset,
            if 'tf': returns a tf.data.Dataset
        threads: multiple processing threadsa-smi


    Returns:
        list of :class:`~transformers.data.processors.squad.SquadFeatures`

    Example::

        processor = SquadV2Processor()
        examples = processor.get_dev_examples(data_dir)

        features = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
        )
    """

    # Defining helper methods
    spt_features = []
    qry_features = []

    squad_convert_example_to_features_init(tokenizer)
    for example in spt_examples:
        example_results = squad_convert_example_to_features(example, max_seq_length, doc_stride, max_query_length,
                                                            is_training, lang2id)
        spt_features.append(example_results)

    for example in qry_examples:
        example_results = squad_convert_example_to_features(example, max_seq_length, doc_stride, max_query_length,
                                                            is_training, lang2id)
        qry_features.append(example_results)

    new_features = []
    unique_id = 1000000000
    example_index = 0
    for example_features in spt_features: #tqdm(features, total=len(features), desc="add example index and unique id"):
        if not example_features:
            continue
        for example_feature in example_features:
            example_feature.example_index = example_index
            example_feature.unique_id = unique_id
            new_features.append(example_feature)
            unique_id += 1
        example_index += 1
    spt_features = new_features
    del new_features

    new_features = []
    unique_id = 1000000000
    example_index = 0
    for example_features in qry_features: #tqdm(features, total=len(features), desc="add example index and unique id"):
        if not example_features:
            continue
        for example_feature in example_features:
            example_feature.example_index = example_index
            example_feature.unique_id = unique_id
            new_features.append(example_feature)
            unique_id += 1
        example_index += 1
    qry_features = new_features
    del new_features

    if return_dataset == "pt":
        if not is_torch_available():
            raise RuntimeError("PyTorch must be installed to return a PyTorch dataset.")

        # Convert to Tensors and build dataset
        # Construct Support/Query datasets
        num_spt = int(data_config["k_spt"]/data_config["q_qry"])

        l_input_ids_s = []
        l_attention_masks_s = []
        l_token_type_ids_s = []
        l_cls_index_s = []
        l_p_mask_s = []
        l_langs_s = []
        l_example_index_s = []
        l_start_positions_s = []
        l_end_positions_s = []

        l_input_ids_q = []
        l_attention_masks_q = []
        l_token_type_ids_q = []
        l_cls_index_q = []
        l_p_mask_q = []
        l_langs_q = []
        l_example_index_q = []
        l_start_positions_q = []
        l_end_positions_q = []

        random.shuffle(qry_examples)
        s = 0
        print("META Batching")
        for _ in tqdm(range(opt_config["batch_sz"])):
            l_input_ids_spt = []
            l_attention_masks_spt = []
            l_token_type_ids_spt = []
            l_all_cls_index_spt = []
            l_p_mask_spt = []
            l_langs_spt = []
            l_start_positions_spt = []
            l_end_positions_spt = []

            l_input_ids_qry = []
            l_attention_masks_qry = []
            l_token_type_ids_qry = []
            l_all_cls_index_qry = []
            l_p_mask_qry = []
            l_langs_qry = []
            l_start_positions_qry = []
            l_end_positions_qry = []

            # Pick q_qry from qry examples randomly
            if s+data_config["q_qry"] < len(qry_examples):
                qry_indices = range(s, s+data_config["q_qry"])
                s = s + data_config["q_qry"]
            elif s < len(qry_examples):
                t = s+data_config["q_qry"] - len(qry_examples)
                qry_indices = list(range(s, len(qry_examples))) + list(range(0, t))
                s = t
            else:
                s = 0
                qry_indices = range(s, len(qry_examples))

            #random.sample(len(qry_examples), k=data_config["q_qry"])

            for index in qry_indices:
                # Pick qry features
                l_input_ids_qry.append(qry_features[index].input_ids)
                l_attention_masks_qry.append(qry_features[index].attention_mask)
                l_token_type_ids_qry.append(qry_features[index].token_type_ids)
                l_all_cls_index_qry.append(qry_features[index].cls_index)
                l_p_mask_qry.append(qry_features[index].p_mask)
                l_langs_qry.append(qry_features[index].langs)
                l_start_positions_qry.append(qry_features[index].start_position)
                l_end_positions_qry.append(qry_features[index].end_position)


                # Pick s_spt from spt examples based on rankings
                spt_indices = qry_features[index].rankings_spt[:num_spt]
                for spt_index in spt_indices:
                    l_input_ids_spt.append(spt_features[spt_index].input_ids)
                    l_attention_masks_spt.append(spt_features[spt_index].attention_mask)
                    l_token_type_ids_spt.append(spt_features[spt_index].token_type_ids)
                    l_all_cls_index_spt.append(spt_features[spt_index].cls_index)
                    l_p_mask_spt.append(spt_features[spt_index].p_mask)
                    l_langs_spt.append(spt_features[spt_index].langs)
                    l_start_positions_spt.append(spt_features[index].start_position)
                    l_end_positions_spt.append(spt_features[index].end_position)

            all_input_ids = torch.tensor(l_input_ids_spt, dtype=torch.long)
            l_input_ids_s.append(all_input_ids)
            l_attention_masks_s.append(torch.tensor(l_attention_masks_spt, dtype=torch.long))
            l_token_type_ids_s.append(torch.tensor(l_token_type_ids_spt, dtype=torch.long))
            l_cls_index_s.append(torch.tensor(l_all_cls_index_spt, dtype=torch.long))
            l_p_mask_s.append(torch.tensor(l_p_mask_spt, dtype=torch.float))
            l_langs_s.append(torch.tensor(l_langs_spt, dtype=torch.long))
            l_example_index_s.append(torch.arange(all_input_ids.size(0), dtype=torch.long))
            l_start_positions_s.append(torch.tensor(l_start_positions_spt, dtype=torch.long))
            l_end_positions_s.append(torch.tensor(l_end_positions_spt, dtype=torch.long))

            all_input_ids = torch.tensor(l_input_ids_qry, dtype=torch.long)
            l_input_ids_q.append(all_input_ids)
            l_attention_masks_q.append(torch.tensor(l_attention_masks_qry, dtype=torch.long))
            l_token_type_ids_q.append(torch.tensor(l_token_type_ids_qry, dtype=torch.long))
            l_cls_index_q.append(torch.tensor(l_all_cls_index_qry, dtype=torch.long))
            l_p_mask_q.append(torch.tensor(l_p_mask_qry, dtype=torch.float))
            l_langs_q.append(torch.tensor(l_langs_qry, dtype=torch.long))
            l_example_index_q.append(torch.arange(all_input_ids.size(0), dtype=torch.long))
            l_start_positions_q.append(torch.tensor(l_start_positions_qry, dtype=torch.long))
            l_end_positions_q.append(torch.tensor(l_end_positions_qry, dtype=torch.long))

        if not is_training:
            dataset = TensorDataset(
                torch.stack(l_input_ids_s), torch.stack(l_attention_masks_s), torch.stack(l_token_type_ids_s),
                torch.stack(l_example_index_s), torch.stack(l_cls_index_s), torch.stack(l_p_mask_s), torch.stack(l_langs_s),
                torch.stack(l_input_ids_q), torch.stack(l_attention_masks_q), torch.stack(l_token_type_ids_q),
                torch.stack(l_example_index_q), torch.stack(l_cls_index_q), torch.stack(l_p_mask_q), torch.stack(l_langs_q))
        else:
            dataset = TensorDataset(
                torch.stack(l_input_ids_s), torch.stack(l_attention_masks_s), torch.stack(l_token_type_ids_s),
                torch.stack(l_start_positions_s), torch.stack(l_end_positions_s), torch.stack(l_cls_index_s),
                torch.stack(l_p_mask_s), torch.stack(l_langs_s), torch.stack(l_input_ids_q), torch.stack(l_attention_masks_q),
                torch.stack(l_token_type_ids_q), torch.stack(l_start_positions_q), torch.stack(l_end_positions_q) ,
                torch.stack(l_cls_index_q), torch.stack(l_p_mask_q), torch.stack(l_langs_q))

        return spt_features, qry_features, dataset

    return spt_features, qry_features


def meta_adapt_squad_convert_examples_to_features(qry_examples, tokenizer, max_seq_length, doc_stride,
                                                  max_query_length, is_training, opt_config, data_config,
                                                  return_dataset=False, threads=1, lang2id=None
                                                  ):
    """
    Converts a list of examples into a list of features that can be directly given as input to a model.
    It is model-dependant and takes advantage of many of the tokenizer's features to create the model's inputs.

    Args:
        examples: list of :class:`~transformers.data.processors.squad.SquadExample`
        tokenizer: an instance of a child of :class:`~transformers.PreTrainedTokenizer`
        max_seq_length: The maximum sequence length of the inputs.
        doc_stride: The stride used when the context is too large and is split across several features.
        max_query_length: The maximum length of the query.
        is_training: whether to create features for model evaluation or model training.
        return_dataset: Default False. Either 'pt' or 'tf'.
            if 'pt': returns a torch.data.TensorDataset,
            if 'tf': returns a tf.data.Dataset
        threads: multiple processing threadsa-smi


    Returns:
        list of :class:`~transformers.data.processors.squad.SquadFeatures`

    Example::

        processor = SquadV2Processor()
        examples = processor.get_dev_examples(data_dir)

        features = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
        )
    """

    # Defining helper methods
    qry_features = []

    squad_convert_example_to_features_init(tokenizer)

    for example in qry_examples:
        example_results = squad_convert_example_to_features(example, max_seq_length, doc_stride, max_query_length,
                                                            is_training, lang2id)
        qry_features.append(example_results)

    new_features = []
    unique_id = 1000000000
    example_index = 0
    for example_features in qry_features: #tqdm(features, total=len(features), desc="add example index and unique id"):
        if not example_features:
            continue
        for example_feature in example_features:
            example_feature.example_index = example_index
            example_feature.unique_id = unique_id
            new_features.append(example_feature)
            unique_id += 1
        example_index += 1
    qry_features = new_features
    del new_features

    if return_dataset == "pt":
        if not is_torch_available():
            raise RuntimeError("PyTorch must be installed to return a PyTorch dataset.")

        # Convert to Tensors and build dataset
        # Construct Support/Query datasets
        num_spt = int(data_config["k_spt"]/data_config["q_qry"])

        l_input_ids_s = []
        l_attention_masks_s = []
        l_token_type_ids_s = []
        l_cls_index_s = []
        l_p_mask_s = []
        l_langs_s = []
        l_example_index_s = []
        l_start_positions_s = []
        l_end_positions_s = []

        l_input_ids_q = []
        l_attention_masks_q = []
        l_token_type_ids_q = []
        l_cls_index_q = []
        l_p_mask_q = []
        l_langs_q = []
        l_example_index_q = []
        l_start_positions_q = []
        l_end_positions_q = []

        random.shuffle(qry_examples)
        s = 0
        print("META Batching")
        for _ in tqdm(range(opt_config["batch_sz"])):
            l_input_ids_spt = []
            l_attention_masks_spt = []
            l_token_type_ids_spt = []
            l_all_cls_index_spt = []
            l_p_mask_spt = []
            l_langs_spt = []
            l_start_positions_spt = []
            l_end_positions_spt = []

            l_input_ids_qry = []
            l_attention_masks_qry = []
            l_token_type_ids_qry = []
            l_all_cls_index_qry = []
            l_p_mask_qry = []
            l_langs_qry = []
            l_start_positions_qry = []
            l_end_positions_qry = []

            # Pick q_qry from qry examples randomly
            if s+data_config["q_qry"] < len(qry_examples):
                qry_indices = range(s, s+data_config["q_qry"])
                s = s + data_config["q_qry"]
            elif s < len(qry_examples):
                t = s+data_config["q_qry"] - len(qry_examples)
                qry_indices = list(range(s, len(qry_examples))) + list(range(0, t))
                s = t
            else:
                s = 0
                qry_indices = range(s, len(qry_examples))

            #random.sample(len(qry_examples), k=data_config["q_qry"])

            for index in qry_indices:
                # Pick qry features
                l_input_ids_qry.append(qry_features[index].input_ids)
                l_attention_masks_qry.append(qry_features[index].attention_mask)
                l_token_type_ids_qry.append(qry_features[index].token_type_ids)
                l_all_cls_index_qry.append(qry_features[index].cls_index)
                l_p_mask_qry.append(qry_features[index].p_mask)
                l_langs_qry.append(qry_features[index].langs)
                l_start_positions_qry.append(qry_features[index].start_position)
                l_end_positions_qry.append(qry_features[index].end_position)


                # Pick s_spt from spt examples based on rankings
                spt_indices = qry_features[index].rankings_qry[1:2]
                for spt_index in spt_indices:
                    l_input_ids_spt.append(qry_features[spt_index].input_ids)
                    l_attention_masks_spt.append(qry_features[spt_index].attention_mask)
                    l_token_type_ids_spt.append(qry_features[spt_index].token_type_ids)
                    l_all_cls_index_spt.append(qry_features[spt_index].cls_index)
                    l_p_mask_spt.append(qry_features[spt_index].p_mask)
                    l_langs_spt.append(qry_features[spt_index].langs)
                    l_start_positions_spt.append(qry_features[index].start_position)
                    l_end_positions_spt.append(qry_features[index].end_position)

            all_input_ids = torch.tensor(l_input_ids_spt, dtype=torch.long)
            l_input_ids_s.append(all_input_ids)
            l_attention_masks_s.append(torch.tensor(l_attention_masks_spt, dtype=torch.long))
            l_token_type_ids_s.append(torch.tensor(l_token_type_ids_spt, dtype=torch.long))
            l_cls_index_s.append(torch.tensor(l_all_cls_index_spt, dtype=torch.long))
            l_p_mask_s.append(torch.tensor(l_p_mask_spt, dtype=torch.float))
            l_langs_s.append(torch.tensor(l_langs_spt, dtype=torch.long))
            l_example_index_s.append(torch.arange(all_input_ids.size(0), dtype=torch.long))
            l_start_positions_s.append(torch.tensor(l_start_positions_spt, dtype=torch.long))
            l_end_positions_s.append(torch.tensor(l_end_positions_spt, dtype=torch.long))

            all_input_ids = torch.tensor(l_input_ids_qry, dtype=torch.long)
            l_input_ids_q.append(all_input_ids)
            l_attention_masks_q.append(torch.tensor(l_attention_masks_qry, dtype=torch.long))
            l_token_type_ids_q.append(torch.tensor(l_token_type_ids_qry, dtype=torch.long))
            l_cls_index_q.append(torch.tensor(l_all_cls_index_qry, dtype=torch.long))
            l_p_mask_q.append(torch.tensor(l_p_mask_qry, dtype=torch.float))
            l_langs_q.append(torch.tensor(l_langs_qry, dtype=torch.long))
            l_example_index_q.append(torch.arange(all_input_ids.size(0), dtype=torch.long))
            l_start_positions_q.append(torch.tensor(l_start_positions_qry, dtype=torch.long))
            l_end_positions_q.append(torch.tensor(l_end_positions_qry, dtype=torch.long))

        if not is_training:
            dataset = TensorDataset(
                torch.stack(l_input_ids_s), torch.stack(l_attention_masks_s), torch.stack(l_token_type_ids_s),
                torch.stack(l_example_index_s), torch.stack(l_cls_index_s), torch.stack(l_p_mask_s), torch.stack(l_langs_s),
                torch.stack(l_input_ids_q), torch.stack(l_attention_masks_q), torch.stack(l_token_type_ids_q),
                torch.stack(l_example_index_q), torch.stack(l_cls_index_q), torch.stack(l_p_mask_q), torch.stack(l_langs_q))
        else:
            dataset = TensorDataset(
                torch.stack(l_input_ids_s), torch.stack(l_attention_masks_s), torch.stack(l_token_type_ids_s),
                torch.stack(l_start_positions_s), torch.stack(l_end_positions_s), torch.stack(l_cls_index_s),
                torch.stack(l_p_mask_s), torch.stack(l_langs_s), torch.stack(l_input_ids_q), torch.stack(l_attention_masks_q),
                torch.stack(l_token_type_ids_q), torch.stack(l_start_positions_q), torch.stack(l_end_positions_q) ,
                torch.stack(l_cls_index_q), torch.stack(l_p_mask_q), torch.stack(l_langs_q))

        return qry_features, dataset

    return qry_features


def find_similarities_query_spt(spt_examples, qry_examples):
    model = SentenceTransformer("xlm-r-distilroberta-base-paraphrase-v1")

    spt_sentences = []
    for spt in tqdm(spt_examples):
        spt_tokens = spt.question_text + " " + spt.context_text + " " + spt.answer_text

        spt_sentences.append(spt_tokens)

    spt_embeddings = model.encode(spt_sentences)

    qry_sentences = []
    for qry in tqdm(qry_examples):
        qry_tokens = qry.question_text + " " + qry.context_text + " " + qry.answer_text
        qry_sentences.append(qry_tokens)

    qry_embeddings = model.encode(qry_sentences)

    cos_sim_spt = 1 - sp.distance.cdist(spt_embeddings, qry_embeddings, 'cosine')
    cos_sim_qry = 1 - sp.distance.cdist(qry_embeddings, qry_embeddings, 'cosine')

    for i, qry in enumerate(qry_examples):
        rankings_spt = np.argsort(cos_sim_spt[i])[::-1]
        rankings_qry = np.argsort(cos_sim_qry[i])[::-1]

        qry.set_rankings_spt(rankings_spt[1:11])
        qry.set_rankings_qry(rankings_qry[1:11])


class SquadProcessor(DataProcessor):
    """
    Processor for the SQuAD data set.
    Overriden by SquadV1Processor and SquadV2Processor, used by the version 1.1 and version 2.0 of SQuAD, respectively.
    """

    train_file = None
    dev_file = None

    def _get_example_from_tensor_dict(self, tensor_dict, evaluate=False):
        if not evaluate:
            answer = tensor_dict["answers"]["text"][0].numpy().decode("utf-8")
            answer_start = tensor_dict["answers"]["answer_start"][0].numpy()
            answers = []
        else:
            answers = [
                {"answer_start": start.numpy(), "text": text.numpy().decode("utf-8")}
                for start, text in zip(tensor_dict["answers"]["answer_start"], tensor_dict["answers"]["text"])
            ]

            answer = None
            answer_start = None

        return SquadExample(
            qas_id=tensor_dict["id"].numpy().decode("utf-8"),
            question_text=tensor_dict["question"].numpy().decode("utf-8"),
            context_text=tensor_dict["context"].numpy().decode("utf-8"),
            answer_text=answer,
            start_position_character=answer_start,
            title=tensor_dict["title"].numpy().decode("utf-8"),
            answers=answers,
        )

    def get_examples_from_dataset(self, dataset, evaluate=False):
        """
        Creates a list of :class:`~transformers.data.processors.squad.SquadExample` using a TFDS dataset.

        Args:
            dataset: The tfds dataset loaded from `tensorflow_datasets.load("squad")`
            evaluate: boolean specifying if in evaluation mode or in training mode

        Returns:
            List of SquadExample

        Examples::

            import tensorflow_datasets as tfds
            dataset = tfds.load("squad")

            training_examples = get_examples_from_dataset(dataset, evaluate=False)
            evaluation_examples = get_examples_from_dataset(dataset, evaluate=True)
        """

        if evaluate:
            dataset = dataset["validation"]
        else:
            dataset = dataset["train"]

        examples = []
        for tensor_dict in dataset:
            examples.append(self._get_example_from_tensor_dict(tensor_dict, evaluate=evaluate))

        return examples

    def get_train_examples(self, data_dir, task, languages=['en']):
        """
        Returns the training examples from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the training file has a different name than the original one
                which is `train-v1.1.json` and `train-v2.0.json` for squad versions 1.1 and 2.0 respectively.

        """
        if data_dir is None:
            data_dir = ""

        train_examples = []

        for lang in languages:
            if task == "tydiqa":
                filename = os.path.join(data_dir, "tydiqa-goldp-v1.1-train") + "/tydiqa." + lang + ".train.json"
            else:
                filename = os.path.join(data_dir, "SQUAD") + "/squad/train-v1.1.json"

            with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as reader:
                input_data = json.load(reader)["data"]

            train_examples.extend(self._create_examples(input_data, "train", lang))

        return train_examples

    def get_dev_examples(self, data_dir, task, languages=['en']):
        """
        Returns the few-shot example from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the evaluation file has a different name than the original one
                which is `train-v1.1.json` and `train-v2.0.json` for squad versions 1.1 and 2.0 respectively.
        """
        if data_dir is None:
            data_dir = ""

        dev_examples = []
        for lang in languages:
            if task == "tydiqa":
                filename = os.path.join(data_dir, "tydiqa-goldp-v1.1-train") + "/tydiqa." + lang + ".dev.json"
            else:
                filename = os.path.join(data_dir, "MLQA_V1") + "/dev/dev-context-" + lang + "-question-"+lang+".json"

            with open(
                    os.path.join(data_dir, filename), "r", encoding="utf-8"
            ) as reader:
                input_data = json.load(reader)["data"]
            dev_examples.extend(self._create_examples(input_data, "train", languages))

        return dev_examples

    def get_squad_dev_examples(self, data_dir, task, languages=['en']):
        """
        Returns the few-shot example from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the evaluation file has a different name than the original one
                which is `train-v1.1.json` and `train-v2.0.json` for squad versions 1.1 and 2.0 respectively.
        """
        if data_dir is None:
            data_dir = ""

        dev_examples = []
        for lang in languages:
            if task == "tydiqa":
                filename = os.path.join(data_dir, "tydiqa-goldp-v1.1-train") + "/tydiqa." + lang + ".dev.json"
            else:
                filename = os.path.join(data_dir, "SQUAD") + "/squad/dev-v1.1.json"

            with open(
                    os.path.join(data_dir, filename), "r", encoding="utf-8"
            ) as reader:
                input_data = json.load(reader)["data"]
            dev_examples.extend(self._create_examples(input_data, "train", languages))

        return dev_examples

    def get_test_examples(self, data_dir, task, language ='en'):
        """
        Returns the evaluation example from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the evaluation file has a different name than the original one
                which is `train-v1.1.json` and `train-v2.0.json` for squad versions 1.1 and 2.0 respectively.
        """
        if data_dir is None:
            data_dir = ""

        if task == "tydiqa":
            filename = os.path.join(data_dir, "tydiqa-goldp-v1.1-dev") + "/tydiqa." + language + ".test.json"
        else:
            filename = os.path.join(data_dir, "MLQA_V1") + "/test/test-context-" + language + "-question-" + language \
                       + ".json"
        with open(
                os.path.join(data_dir, filename), "r", encoding="utf-8"
        ) as reader:
            input_data = json.load(reader)["data"]
        return self._create_examples(input_data, "dev", language)

    def _create_examples(self, input_data, set_type, language):
        is_training = set_type == "train"
        examples = []
        for entry in input_data:
            title = entry["title"] if "title" in entry else ""
            for paragraph in entry["paragraphs"]:
                context_text = paragraph["context"]
                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    start_position_character = None
                    answer_text = None
                    answers = []

                    if "is_impossible" in qa:
                        is_impossible = qa["is_impossible"]
                    else:
                        is_impossible = False

                    if not is_impossible:
                        #if is_training:
                        answer = qa["answers"][0]
                        answer_text = answer["text"]
                        start_position_character = answer["answer_start"]
                        #else:
                        #    answers = qa["answers"]

                    example = SquadExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        context_text=context_text,
                        answer_text=answer_text,
                        start_position_character=start_position_character,
                        title=title,
                        is_impossible=is_impossible,
                        answers=answers,
                        language=language
                    )

                    examples.append(example)
        return examples


class SquadV1Processor(SquadProcessor):
    train_file = "train-v1.1.json"
    dev_file = "dev-v1.1.json"


class SquadV2Processor(SquadProcessor):
    train_file = "train-v2.0.json"
    dev_file = "dev-v2.0.json"


class SquadExample(object):
    """
    A single training/test example for the Squad dataset, as loaded from disk.

    Args:
        qas_id: The example's unique identifier
        question_text: The question string
        context_text: The context string
        answer_text: The answer string
        start_position_character: The character position of the start of the answer
        title: The title of the example
        answers: None by default, this is used during evaluation. Holds answers as well as their start positions.
        is_impossible: False by default, set to True if the example has no possible answer.
    """

    def __init__(
            self,
            qas_id,
            question_text,
            context_text,
            answer_text,
            start_position_character,
            title,
            answers=[],
            is_impossible=False,
            language='en',
            rankings_spt=[],
            rankings_qry=[]
    ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.context_text = context_text
        self.answer_text = answer_text
        self.title = title
        self.is_impossible = is_impossible
        self.answers = answers

        self.start_position, self.end_position = 0, 0

        self.language = language
        self.rankings_spt = rankings_spt
        self.rankings_qry = rankings_qry

        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True

        # Split on whitespace so that different tokens may be attributed to their original position.
        for c in self.context_text:
            if _is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        self.doc_tokens = doc_tokens
        self.char_to_word_offset = char_to_word_offset

        # Start end end positions only has a value during evaluation.
        if start_position_character is not None and not is_impossible:
            self.start_position = char_to_word_offset[start_position_character]
            self.end_position = char_to_word_offset[
                min(start_position_character + len(answer_text) - 1, len(char_to_word_offset) - 1)
            ]

    def set_rankings_spt(self, rankings_spt):
        self.rankings_spt = rankings_spt

    def set_rankings_qry(self, rankings_qry):
        self.rankings_qry = rankings_qry


class SquadFeatures(object):
    """
    Single squad example features to be fed to a model.
    Those features are model-specific and can be crafted from :class:`~transformers.data.processors.squad.SquadExample`
    using the :method:`~transformers.data.processors.squad.squad_convert_examples_to_features` method.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        cls_index: the index of the CLS token.
        p_mask: Mask identifying tokens that can be answers vs. tokens that cannot.
            Mask with 1 for tokens than cannot be in the answer and 0 for token that can be in an answer
        example_index: the index of the example
        unique_id: The unique Feature identifier
        paragraph_len: The length of the context
        token_is_max_context: List of booleans identifying which tokens have their maximum context in this feature object.
            If a token does not have their maximum context in this feature object, it means that another feature object
            has more information related to that token and should be prioritized over this feature for that token.
        tokens: list of tokens corresponding to the input ids
        token_to_orig_map: mapping between the tokens and the original text, needed in order to identify the answer.
        start_position: start of the answer token index
        end_position: end of the answer token index
    """

    def __init__(
            self,
            input_ids,
            attention_mask,
            token_type_ids,
            rankings_spt,
            rankings_qry,
            cls_index,
            p_mask,
            example_index,
            unique_id,
            paragraph_len,
            token_is_max_context,
            tokens,
            token_to_orig_map,
            start_position,
            end_position,
            langs
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.rankings_spt = rankings_spt
        self.rankings_qry = rankings_qry
        self.cls_index = cls_index
        self.p_mask = p_mask

        self.example_index = example_index
        self.unique_id = unique_id
        self.paragraph_len = paragraph_len
        self.token_is_max_context = token_is_max_context
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map

        self.start_position = start_position
        self.end_position = end_position
        self.langs = langs


class SquadResult(object):
    """
    Constructs a SquadResult which can be used to evaluate a model's output on the SQuAD dataset.

    Args:
        unique_id: The unique identifier corresponding to that example.
        start_logits: The logits corresponding to the start of the answer
        end_logits: The logits corresponding to the end of the answer
    """

    def __init__(self, unique_id, start_logits, end_logits, start_top_index=None, end_top_index=None, cls_logits=None):
        self.start_logits = start_logits
        self.end_logits = end_logits
        self.unique_id = unique_id

        if start_top_index:
            self.start_top_index = start_top_index
            self.end_top_index = end_top_index
            self.cls_logits = cls_logits
