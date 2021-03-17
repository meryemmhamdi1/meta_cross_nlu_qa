from __future__ import print_function
import argparse
import glob
import logging
import os
import random
import timeit
import numpy as np
from tqdm import tqdm, trange


import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from transformers_config import *

from data_utils import (
    SquadResult,
    SquadV1Processor,
    SquadV2Processor,
    squad_convert_examples_to_features
)

import learn2learn as l2l
from meta_learner_l2l import *

no_decay = ["bias", "LayerNorm.weight"]


logger = logging.getLogger(__name__)

from collections import Counter
import string
import re
import argparse
import json
import sys


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


def evaluate(tokenizer, model, features, examples, dataset, language, prefix, eval_batch_size, model_type, out_dir,
             n_best_size, max_answer_length, version_2_with_negative, verbose_logging, do_lower_case,
             null_score_diff_threshold, lang2id):

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

        predictions = compute_predictions_log_probs(examples,
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
        predictions = compute_predictions_logits(examples,
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

    # Compute the F1 and exact scores.
    #results = squad_evaluate(examples, predictions)
    if prefix == "test":
        data_file = "/nas/clear/users/meryem/Datasets/QA/tydiqa/tydiqa-goldp-v1.1-dev/tydiqa."+language+".test.json"
    else:
        data_file = "/nas/clear/users/meryem/Datasets/QA/tydiqa/tydiqa-goldp-v1.1-train/tydiqa."+language+".train.json"

    with open(data_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        dataset = dataset_json['data']
    with open(output_prediction_file) as prediction_file:
        predictions = json.load(prediction_file)
    results = evaluate_sq(dataset, predictions)
    return results


def run(config_name, trans_model, model_type, tokenizer_name, do_lower_case, cache_dir, device, version_2_with_negative,
        null_score_diff_threshold, verbose_logging, data_dir, train_langs, dev_langs, test_langs, max_seq_length,
        doc_stride, max_query_length, pre_train_config, data_config, opt_config, out_dir, writer, use_pretrained_model,
        freeze_bert, pre_trained_model_path):

    model_name, tokenizer_class, model_class, config_class, qa_class = MODELS_dict[trans_model]

    config = config_class.from_pretrained(config_name if config_name else model_name,
                                          cache_dir=cache_dir if cache_dir else None)

    # Set usage of language embedding to True if model is xlm
    if model_type == "xlm":
        config.use_lang_emb = True

    tokenizer = tokenizer_class.from_pretrained(tokenizer_name if tokenizer_name else model_name,
                                                do_lower_case=do_lower_case,
                                                cache_dir=cache_dir if cache_dir else None)

    model = qa_class.from_pretrained(model_name,
                                     from_tf=bool(".ckpt" in model_name),
                                     config=config,
                                     cache_dir=cache_dir if cache_dir else None)

    lang2id = config.lang2id if model_type == "xlm" else None

    model.to(device)

    processor = SquadV2Processor() if version_2_with_negative else SquadV1Processor()

    print("Few shot examples convertion to features")
    few_shot_examples = processor.get_train_examples(data_dir, languages=dev_langs)+ \
                        processor.get_dev_examples(data_dir, languages=dev_langs)

    few_shot_features, few_shot_dataset = squad_convert_examples_to_features(
        examples=few_shot_examples,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        max_query_length=max_query_length,
        is_training=True,
        return_dataset="pt",
        threads=8,
        lang2id=lang2id
    )

    test_features = {}
    test_dataset = {}
    test_examples = {}
    for lang in test_langs:
        test_examples.update({lang: processor.get_test_examples(data_dir, language=lang)})
        print("Test examples convertion to features %s len(test_examples[lang]):%d", lang, len(test_examples[lang]))
        test_features_lang, test_dataset_lang = squad_convert_examples_to_features(examples=test_examples[lang],
                                                                                   tokenizer=tokenizer,
                                                                                   max_seq_length=max_seq_length,
                                                                                   doc_stride=doc_stride,
                                                                                   max_query_length=max_query_length,
                                                                                   is_training=True,
                                                                                   return_dataset="pt",
                                                                                   threads=8,
                                                                                   lang2id=lang2id)
        test_features.update({lang: test_features_lang})
        test_dataset.update({lang: test_dataset_lang})

    fine_tune_sampler = RandomSampler(few_shot_dataset)
    fine_tune_dataloader = DataLoader(few_shot_dataset, sampler=fine_tune_sampler, batch_size=pre_train_config["batch_size"])

    num_train_epochs = 20
    t_total = len(fine_tune_dataloader) // pre_train_config["gradient_accumulation_steps"] * num_train_epochs

    local_rank = -1

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": pre_train_config["weight_decay"],
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=pre_train_config["adam_lr"], eps=pre_train_config["adam_eps"])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=pre_train_config["warmup_steps"],
                                                num_training_steps=t_total)

    ####

    print("MONOLINGUAL TRAINING ON LANGUAGE:", dev_langs)

    global_step, tr_loss, logging_loss = 0, 0.0, 0.0
    for _ in tqdm(range(opt_config["epoch"])):
        epoch_iterator = tqdm(fine_tune_dataloader, desc="Iteration", disable=local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": None if model_type in ["xlm", "xlm-roberta", "distilbert"] else batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
            }

            if model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[5], "p_mask": batch[6]})
                if version_2_with_negative:
                    inputs.update({"is_impossible": batch[7]})
            if model_type == "xlm":
                inputs["langs"] = batch[7]

            outputs = model(**inputs)

            loss = outputs[0]

            loss = loss.mean()

            if pre_train_config["gradient_accumulation_steps"] > 1:
                loss = loss / pre_train_config["gradient_accumulation_steps"]

            loss.backward()

            tr_loss += loss.item()

            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()

            if (step + 1) % pre_train_config["gradient_accumulation_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), pre_train_config["max_grad_norm"])

                global_step += 1

                ## Write loss metrics
                writer.add_scalar("fine_tune_lr", scheduler.get_lr()[0], global_step)
                writer.add_scalar("FINE_TUNE_loss", (tr_loss - logging_loss) / pre_train_config["logging_steps"],
                                  global_step)
                logging_loss = tr_loss

            if pre_train_config["save_steps"] > 0 and global_step % pre_train_config["save_steps"] == 0:
                output_dir = os.path.join(out_dir, "checkpoint-{}".format(global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    # Take care of distributed/parallel training
                model_to_save = model.module if hasattr(model, "module") else model
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)

                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                logger.info("Saving model checkpoint to %s", output_dir)

                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if global_step % 50 == 0:
                for lang in test_langs:
                    test_results = evaluate(tokenizer, model, test_features[lang], test_examples[lang], test_dataset[lang],
                                            lang, "test", pre_train_config["eval_batch_size"],
                                            model_type, out_dir, pre_train_config["n_best_size"],
                                            pre_train_config["max_answer_length"], version_2_with_negative,
                                            verbose_logging, do_lower_case, null_score_diff_threshold, lang2id)

                    print("FINE TUNE lang:", lang, " test_results:", test_results)
                    for key, value in test_results.items():
                        writer.add_scalar("FINE_TUNE_test_{}_{}".format(lang, key), value, global_step)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    devices = torch.cuda.device_count()
    print("devices:", devices)
    if devices > 1:
        torch.cuda.manual_seed_all(args.seed)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-langs", help="train languages list", nargs="+", default=[])
    parser.add_argument("--dev-langs", help="dev languages list", nargs="+", default=[])
    parser.add_argument("--test-langs", help="test languages list", nargs="+", default=[])
    parser.add_argument('--use-few-shot', help='If true, use test languages in the meta-adaptation stage',
                        action='store_true')  # zero-shot by default

    parser.add_argument('--use-slots', help='If true, optimize for slot filling loss too', action='store_true')

    parser.add_argument('--data-dir', help='Path of data',  default="")

    parser.add_argument("--version-2-with-negative", action="store_true",
                        help="If true, the SQuAD examples contain some that do not have an answer.")

    parser.add_argument('--out-dir', help='Path of output data', default="")
    parser.add_argument('--pre-trained-model-name', help='Path of output pre-trained model binary', default="")
    parser.add_argument('--data-format', help='Whether it is tsv or json', default="tsv")

    parser.add_argument( "--max-seq-length", default=384, type=int,
                         help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                              "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc-stride", default=128, type=int, help="When splitting up a long document into chunks, "
                                                                    "how much stride to take between chunks.")

    parser.add_argument("--max-query-length", default=64, type=int, help="The maximum number of tokens for the question"
                        ". Questions longer than this will be truncated to this length.")

    parser.add_argument("--n-best-size", default=20, type=int, help="The total number of n-best predictions to generate"
                                                                    " in the nbest_predictions.json output file.")

    parser.add_argument("--max-answer-length", default=30, type=int, help="The maximum length of an answer that can be"
                                                                          " generated. This is needed because the start"
                                                                          " and end predictions are not conditioned on"
                                                                          " one another.")

    ## Transformers options
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")

    parser.add_argument("--do-lower-case", action="store_true", help="Set this flag if you are using an uncased model.")

    parser.add_argument("--null-score-diff-threshold", type=float, default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.")

    parser.add_argument("--verbose-logging", action="store_true",
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")

    parser.add_argument("--trans-model", help="name of transformer model", default="BertBaseMultilingualCased")
    parser.add_argument("--config-name", default="", type=str, help="Pretrained config name or path if not the same "
                                                                    "as model_name")

    parser.add_argument("--model-type", help="name of transformer model: xlnet, xlm, bert, xlm-roberta, distilbert",
                        default="bert")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--task', help='tydiqa, mlqa or else', default="tydiqa")

    ## Pre-training hyperparameters
    parser.add_argument('--pre-train-steps', help='the number of iterations if pre-training is done from scratch',
                        type=int, default=2000)

    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument('--batch-size', help="batch size in the pre-training process", type=int, default=8)
    parser.add_argument('--eval-batch-size', help="batch size in the pre-training process", type=int, default=8)
    parser.add_argument('--adam-lr', help="learning rate of adam optimizer when training base model from scratch",
                        type=float, default=3e-5)
    parser.add_argument('--adam-eps', help="epsilon of adam optimizer when training base model from scratch",
                        type=float, default= 1e-08)
    parser.add_argument("--weight-decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--warmup-steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument('--use-pretrained-model', help='If true, use pre-trained NLU model', action='store_true')
    parser.add_argument("--save-steps", type=int, default=50, help="Save checkpoint every X updates steps.")
    parser.add_argument("--logging-steps", type=int, default=50, help="Log every X updates steps.")

    ## Meta-learning Dataset Hyperparameters (Have to run using grid search to analyze learning curves)
    parser.add_argument('--n-way', help='Number of classes for each task in the meta-learning', type=int, default=11)
    parser.add_argument('--k-spt', help='Number of support examples per class', type=int, default=4)
    parser.add_argument('--q-qry', help='Number of query examples per class', type=int, default=4)
    parser.add_argument('--k-tune', help='Number of query examples per class', type=int, default=2)
    parser.add_argument('--batch-sz', help='Number of iterations', type=int, default=1000)
    parser.add_argument('--seed', help="Random seed for initialization", type=int, default=42)

    ## Meta-learning optimization Hyperparameters (tunable => hyperparameter optimization search or some automatic tool)
    parser.add_argument('--epoch', help='Number of epochs', type=int, default=10)  # Early stopping

    parser.add_argument('--n-task', help='Number of tasks', type=int, default=4)

    parser.add_argument('--n-up-train-step', help='Number of update steps in the meta-training stage', type=int,
                        default=5)

    parser.add_argument('--n-up-test-step', help="Number of update steps in the meta-update stage", type=int,
                        default=10)

    parser.add_argument('--alpha-lr', help='Learning rate during the meta-training stage (inner loop)', type=int,
                        default=1e-2)

    parser.add_argument('--beta-lr', help='Learning rate during the meta-update stage (outer loop)', type=int,
                        default=1e-3)

    parser.add_argument('--gamma-lr', help='Learning rate during the meta-update in the adaptation', type=int,
                        default=1e-3)

    parser.add_argument('--local_rank', type=int, help='local rank for DistributedDataParallel')
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")

    parser.add_argument('--use-dpp', help='Whether to use DPP or not', action='store_true')
    parser.add_argument('--use-adapt', help='Whether to use meta-adaptation', action='store_true')
    parser.add_argument('--freeze-bert', help='Whether to use meta-adaptation', nargs="+", default=[])
    parser.add_argument("--pre-trained-model-path", default="/nas/clear/users/meryem/Results/meta_zsl_qa/tydiqa/MAML/"
                                                            "train_en-test_ar,bn,en,fi,id,ko,ru,sw,te/l2l/kspt_2-qqry_2/"
                                                            "en_train_set/few_shot_ar/use_adapt/checkpoint-2000/")

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = get_arguments()
    set_seed(args)

    """ Config Parameters """
    pre_train_config = {"pre_train_steps": args.pre_train_steps, "batch_size": args.batch_size, "adam_lr": args.adam_lr,
                        "adam_eps": args.adam_eps, "gradient_accumulation_steps": args.gradient_accumulation_steps,
                        "warmup_steps": args.warmup_steps, "max_grad_norm": args.max_grad_norm,
                        "save_steps": args.save_steps, "weight_decay": args.weight_decay,
                        "logging_steps": args.logging_steps, "eval_batch_size": args.eval_batch_size,
                        "n_best_size": args.n_best_size, "max_answer_length": args.max_seq_length}

    data_config = {"n_way": args.n_way, "k_spt": args.k_spt, "q_qry": args.q_qry, "batch_sz": args.batch_sz}

    opt_config = {"epoch": args.epoch, "n_task": args.n_task, "n_up_train_step": args.n_up_train_step,
                  "n_up_test_step": args.n_up_test_step, "alpha_lr": args.alpha_lr, "beta_lr": args.beta_lr,
                  "gamma_lr": args.gamma_lr}

    """ Output Directory """
    if args.use_adapt:
        flag_adapt = "use_adapt/"
    else:
        flag_adapt = "no_adapt/"

    freeze_bert_flag = ""
    if len(args.freeze_bert) > 0:
        freeze_bert_flag = "freeze_bert_" + ",".join(args.freeze_bert)

    out_dir = os.path.join(args.out_dir, "MAML/train_"+",".join(args.train_langs)+"-test_"+",".join(args.test_langs)
                           + "/l2l/kspt_" + str(data_config["k_spt"]) + "-qqry_" + str(data_config["q_qry"])
                           + "/en_train_set/" + freeze_bert_flag + "/few_shot_"+",".join(args.dev_langs)+"/"
                           + flag_adapt)

    writer = SummaryWriter(os.path.join(out_dir, 'runs'))

    """ Cuda/CPU device setup"""

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        n_gpu = 1

    run(args.config_name, args.trans_model, args.model_type, args.tokenizer_name, args.do_lower_case, args.cache_dir,
        device, args.version_2_with_negative, args.null_score_diff_threshold, args.verbose_logging, args.data_dir,
        args.train_langs, args.dev_langs, args.test_langs, args.max_seq_length, args.doc_stride, args.max_query_length,
        pre_train_config, data_config, opt_config, out_dir, writer, args.use_pretrained_model, args.freeze_bert)
