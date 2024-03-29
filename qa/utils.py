from __future__ import print_function
import logging
import os
from tqdm import tqdm

from transformers_config import *


import torch
from torch.utils.data import DataLoader, SequentialSampler

## SQUAD Metrics
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from data_utils import (
    SquadResult,
    SquadV1Processor,
    SquadV2Processor,
    squad_convert_examples_to_features
)


logger = logging.getLogger(__name__)

from collections import Counter
import string
import re
import json
import sys
import argparse

no_decay = ["bias", "LayerNorm.weight"]


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate_sq(dataset, predictions):
    """ Taken from https://github.com/google-research/xtreme/blob/master/third_party/evaluate_squad.py#L77 """
    f1 = exact_match = total = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + qa['id'] + \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = predictions[qa['id']]
                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                f1 += metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def evaluate(tokenizer, model, examples, language, prefix, model_type, out_dir, n_best_size, max_answer_length,
             version_2_with_negative, verbose_logging, do_lower_case, null_score_diff_threshold, lang2id, data_path,
             device, args):

    features, dataset = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length,
        is_training=False,
        return_dataset="pt",
        threads=8,
        lang2id=lang2id
    )

    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=1)

    all_results = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": None if model_type in ["xlm", "distilbert", "xlm-roberta"] else batch[2],
            }

            example_indices = batch[3]

            # XLNet and XLM use more arguments for their predictions
            if model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[4], "p_mask": batch[5]})
            if model_type == "xlm":
                inputs["langs"] = batch[6]

            outputs = model(**inputs)

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            output = [to_list(output[i]) for output in outputs]

            # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
            # models only use two.
            if len(output) >= 5:
                start_logits = output[0]
                start_top_index = output[1]
                end_logits = output[2]
                end_top_index = output[3]
                cls_logits = output[4]

                result = SquadResult(
                    unique_id,
                    start_logits,
                    end_logits,
                    start_top_index=start_top_index,
                    end_top_index=end_top_index,
                    cls_logits=cls_logits,
                )

            else:
                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)

            all_results.append(result)

    # Compute predictions
    output_prediction_file = os.path.join(out_dir, "predictions_{}_{}.json".format(language, prefix))
    output_nbest_file = os.path.join(out_dir, "nbest_predictions_{}_{}.json".format(language, prefix))

    output_null_log_odds_file = os.path.join(out_dir, "null_odds_{}.json".format(prefix))

    if model_type in ["xlnet", "xlm"]:
        start_n_top = model.config.start_n_top if hasattr(model, "config") else model.module.config.start_n_top
        end_n_top = model.config.end_n_top if hasattr(model, "config") else model.module.config.end_n_top

        # Saves the predictions directly in output_prediction_file
        compute_predictions_log_probs(examples,
                                      features,
                                      all_results,
                                      n_best_size,
                                      max_answer_length,
                                      output_prediction_file,
                                      output_nbest_file,
                                      output_null_log_odds_file,
                                      start_n_top,
                                      end_n_top,
                                      version_2_with_negative,
                                      tokenizer,
                                      verbose_logging)

    else:
        # Saves the predictions directly in output_prediction_file
        compute_predictions_logits(examples,
                                   features,
                                   all_results,
                                   n_best_size,
                                   max_answer_length,
                                   do_lower_case,
                                   output_prediction_file,
                                   output_nbest_file,
                                   output_null_log_odds_file,
                                   verbose_logging,
                                   version_2_with_negative,
                                   null_score_diff_threshold,
                                   tokenizer)

    # Load the predictions
    with open(output_prediction_file) as prediction_file:
        predictions = json.load(prediction_file)

    # Read the gold examples
    if prefix == "test":
        data_file = data_path + "/tydiqa-goldp-v1.1-dev/tydiqa."+language+".test.json"
    else:
        data_file = data_path + "/tydiqa-goldp-v1.1-train/tydiqa."+language+".train.json"

    with open(data_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        dataset = dataset_json['data']

    results = evaluate_sq(dataset, predictions)
    return results


def get_meta_arguments():
    parser = argparse.ArgumentParser()

    ## Meta-learning Dataset Hyperparameters (Have to run using grid search to analyze learning curves)
    parser.add_argument('--n-way', help='Number of classes for each task in the meta-learning', type=int, default=11)
    parser.add_argument('--k-spt', help='Number of support examples per class', type=int, default=4)
    parser.add_argument('--q-qry', help='Number of query examples per class', type=int, default=4)
    parser.add_argument('--k-tune', help='Number of query examples per class', type=int, default=2)
    parser.add_argument('--batch-sz', help='Number of iterations', type=int, default=1000)

    ## Meta-learning optimization Hyperparameters (tunable => hyperparameter optimization search or some automatic tool)
    parser.add_argument('--n-task', help='Number of tasks', type=int, default=4)

    parser.add_argument('--n-up-train-step', help='Number of update steps in the meta-training stage', type=int,
                        default=5)

    parser.add_argument('--n-up-test-step', help="Number of update steps in the meta-update stage", type=int,
                        default=5)

    parser.add_argument('--alpha-lr', help='Learning rate during the meta-training stage (inner loop)', type=int,
                        default=3e-5)

    parser.add_argument('--beta-lr', help='Learning rate during the meta-update stage (outer loop)', type=int,
                        default=3e-5)

    parser.add_argument('--gamma-lr', help='Learning rate during the meta-update in the adaptation', type=int,
                        default=3e-5)

    parser.add_argument('--use-adapt', help='Whether to use meta-adaptation', action='store_true')

    args = parser.parse_args()

    return args


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-langs", help="train languages list", nargs="+", default=[])
    parser.add_argument("--dev-langs", help="dev languages list", nargs="+", default=[])
    parser.add_argument("--test-langs", help="test languages list", nargs="+", default=[])
    parser.add_argument("--option", help="PRE, MONO, FT", default="FT")

    parser.add_argument('--data-dir', help='Path of data',  default="")

    parser.add_argument("--version-2-with-negative", action="store_true",
                        help="If true, the SQuAD examples contain some that do not have an answer.")

    parser.add_argument('--out-dir', help='Path of output data', default="")
    parser.add_argument('--pre-trained-model-name', help='Path of output pre-trained model binary', default="")
    parser.add_argument('--data-format', help='Whether it is tsv or json', default="tsv")

    parser.add_argument("--max-seq-length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.") # Fixed from XTREME

    parser.add_argument("--doc-stride", default=128, type=int, help="When splitting up a long document into chunks, "
                                                                    "how much stride to take between chunks.") # Fixed from XTREME

    parser.add_argument("--max-query-length", default=64, type=int, help="The maximum number of tokens for the question"
                                                                         ". Questions longer than this will be truncated to this length.") # Fixed from XTREME

    parser.add_argument("--n-best-size", default=20, type=int, help="The total number of n-best predictions to generate"
                                                                    " in the nbest_predictions.json output file.") # This is for displaying the n-best predictions for each test example

    parser.add_argument("--max-answer-length", default=30, type=int, help="The maximum length of an answer that can be"
                                                                          " generated. This is needed because the start"
                                                                          " and end predictions are not conditioned on"
                                                                          " one another.") # Fixed from XTREME


    ## Transformers options
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")

    parser.add_argument("--trans-model", help="name of transformer model", default="BertBaseMultilingualCased")
    parser.add_argument("--config-name", default="", type=str, help="Pretrained config name or path if not the same "
                                                                    "as model_name")

    parser.add_argument("--model-type", help="name of transformer model: xlnet, xlm, bert, xlm-roberta, distilbert",
                        default="bert")

    parser.add_argument("--do-lower-case", action="store_true", help="Set this flag if you are using an uncased model.")

    parser.add_argument("--null-score-diff-threshold", type=float, default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.")

    parser.add_argument("--verbose-logging", action="store_true",
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--task', help='tydiqa, mlqa or else', default="tydiqa")

    ## Pre-training hyperparameters
    parser.add_argument('--pre-train-steps', help='the number of iterations if pre-training is done from scratch',
                        type=int, default=5000)

    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument('--batch-size', help="batch size in the pre-training process", type=int, default=8)
    parser.add_argument('--eval-batch-size', help="batch size in the pre-training process", type=int, default=8)

    parser.add_argument('--adam-lr', help="learning rate of adam optimizer when training base model from scratch",
                        type=float, default=3e-5)
    parser.add_argument('--adam-eps', help="epsilon of adam optimizer when training base model from scratch",
                        type=float, default=1e-08)

    parser.add_argument("--weight-decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--warmup-steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument('--use-pretrained-model', help='If true, use pre-trained NLU model', action='store_true')
    parser.add_argument("--save-steps", type=int, default=50, help="Save checkpoint every X updates steps.")
    parser.add_argument("--logging-steps", type=int, default=50, help="Log every X updates steps.")

    parser.add_argument('--seed', help="Random seed for initialization", type=int, default=42)
    parser.add_argument('--epoch', help='Number of epochs', type=int, default=10)  # Early stopping

    parser.add_argument('--local_rank', type=int, help='local rank for DistributedDataParallel')
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument('--use-dpp', help='Whether to use DPP or not', action='store_true')

    args = parser.parse_args()

    return args