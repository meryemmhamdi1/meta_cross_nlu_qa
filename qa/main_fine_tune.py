from __future__ import print_function
import logging
import os
import random
import numpy as np
from tqdm import tqdm, trange


import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

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


logger = logging.getLogger(__name__)

from collections import Counter
import string
import re
import argparse
import json
import sys

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
             version_2_with_negative, verbose_logging, do_lower_case, null_score_diff_threshold, lang2id, data_path):

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


    with open(output_prediction_file) as prediction_file:
        predictions = json.load(prediction_file)

    if prefix == "test":
        data_file = data_path + "/tydiqa-goldp-v1.1-dev/tydiqa."+language+".test.json"
    else:
        data_file = data_path + "/tydiqa-goldp-v1.1-train/tydiqa."+language+".train.json"

    with open(data_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        dataset = dataset_json['data']

    results = evaluate_sq(dataset, predictions)
    return results

def run(args, device, fine_tune_config, out_dir, writer):

    model_name, tokenizer_class, model_class, config_class, qa_class = MODELS_dict[args.trans_model]

    if args.cache_dir == "":
        config = config_class.from_pretrained(args.config_name if args.config_name else model_name,
                                             cache_dir=args.cache_dir if args.cache_dir else None)
    else:
        config = config_class.from_pretrained(args.cache_dir)

    # Set usage of language embedding to True if model is xlm
    if args.model_type == "xlm":
        config.use_lang_emb = True

    if args.cache_dir == "":
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else model_name,
                                                    do_lower_case=args.do_lower_case,
                                                    cache_dir=args.cache_dir if args.cache_dir else None)

        model = qa_class.from_pretrained(model_name,
                                         from_tf=bool(".ckpt" in model_name),
                                         config=config,
                                         cache_dir=args.cache_dir if args.cache_dir else None)
    else:
        tokenizer = tokenizer_class.from_pretrained(args.cache_dir)

        model = qa_class.from_pretrained(args.cache_dir)

    lang2id = config.lang2id if args.model_type == "xlm" else None

    model.to(device)

    processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()

    ## TRAIN EXAMPLES
    train_examples = processor.get_train_examples(args.data_dir, task="tydiqa", languages=args.train_langs)
    print("Train examples convertion to features")
    train_features, train_dataset = squad_convert_examples_to_features(
        examples=train_examples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length,
        is_training=True,
        return_dataset="pt",
        threads=8,
        lang2id=lang2id)

    ### TEST EXAMPLES
    test_features = {}
    test_dataset = {}
    test_examples = {}
    for lang in args.test_langs:
        test_examples.update({lang: processor.get_test_examples(args.data_dir, task="tydiqa", language=lang)})
        print("Test examples convertion to features %s len(test_examples[lang]):%d", lang, len(test_examples[lang]))
        test_features_lang, test_dataset_lang = squad_convert_examples_to_features(examples=test_examples[lang],
                                                                                   tokenizer=tokenizer,
                                                                                   max_seq_length=args.max_seq_length,
                                                                                   doc_stride=args.doc_stride,
                                                                                   max_query_length=args.max_query_length,
                                                                                   is_training=True,
                                                                                   return_dataset="pt",
                                                                                   threads=8,
                                                                                   lang2id=lang2id)
        test_features.update({lang: test_features_lang})
        test_dataset.update({lang: test_dataset_lang})



    ### Training
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=fine_tune_config["batch_size"])

    num_train_epochs = args.epoch
    train_dataloader_num = len(train_dataloader)
    t_total = train_dataloader_num // fine_tune_config["gradient_accumulation_steps"] * num_train_epochs

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": fine_tune_config["weight_decay"],
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=fine_tune_config["adam_lr"], eps=fine_tune_config["adam_eps"])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=fine_tune_config["warmup_steps"],
                                                num_training_steps=t_total)

    local_rank = -1

    if args.option == "FT":
        if args.use_pretrained_model:
            print("LOADING PRE-TRAINING MODEL ON ENGLISH ...")
            optimizer.load_state_dict(torch.load(os.path.join(args.pre_trained_model_name, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(args.pre_trained_model_name, "scheduler.pt")))

            # Load a trained model and vocabulary that you have fine-tuned
            model_dict = torch.load(args.pre_trained_model_name+"pytorch_model.bin")
            model.load_state_dict(model_dict)
            model.to(device)
    elif args.option == "PRE":
        print("TRAINING FROM SCRATCH ...")

        global_step, epochs_trained, tr_loss, logging_loss = 1, 0, 0.0,  0.0

        model.zero_grad()

        train_iterator = trange(epochs_trained, int(num_train_epochs), desc="Epoch", disable=local_rank not in [-1, 0])

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=local_rank not in [-1, 0])
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

                if args.model_type in ["xlnet", "xlm"]:
                    inputs.update({"cls_index": batch[5], "p_mask": batch[6]})
                    if args.version_2_with_negative:
                        inputs.update({"is_impossible": batch[7]})
                if args.model_type == "xlm":
                    inputs["langs"] = batch[7]
                outputs = model(**inputs)
                # model outputs are always tuple in transformers (see doc)
                loss = outputs[0]

                loss = loss.mean()

                if fine_tune_config["gradient_accumulation_steps"] > 1:
                    loss = loss / fine_tune_config["gradient_accumulation_steps"]

                loss.backward()

                tr_loss += loss.item()

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()

                if (step + 1) % fine_tune_config["gradient_accumulation_steps"] == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), fine_tune_config["max_grad_norm"])

                    global_step += 1

                    ## Write loss metrics
                    writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    writer.add_scalar("TRAIN_loss", (tr_loss - logging_loss) / fine_tune_config["logging_steps"],
                                      global_step)
                    logging_loss = tr_loss

                 if fine_tune_config["save_steps"] > 0 and global_step % fine_tune_config["save_steps"] == 0:
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

    print("MULTILINGUAL TESTING ...")
    for lang in args.test_langs:
        test_results = evaluate(tokenizer, model, test_examples[lang], lang, "test", args.model_type, out_dir, fine_tune_config["n_best_size"],
                                fine_tune_config["max_answer_length"], args.version_2_with_negative,
                                args.verbose_logging, args.do_lower_case, args.null_score_diff_threshold, lang2id)

        print("lang:", lang, " test_results:", test_results)
        for key, value in test_results.items():
            writer.add_scalar("Test_{}_{}".format(lang, key), value, 0)

    ####

    if args.option in ["FT", "MONO"]:
        print("Language Specific fine tune examples convertion to features")
        fine_tune_examples = processor.get_dev_examples(args.data_dir, task="tydiqa", languages=args.dev_langs)

        fine_tune_features, fine_tune_dataset = squad_convert_examples_to_features(
            examples=fine_tune_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=True,
            return_dataset="pt",
            threads=8,
            lang2id=lang2id
        )

        print("FINE-TUNING ON LANGUAGE:", args.dev_langs)

        fine_tune_sampler = RandomSampler(fine_tune_dataset)
        fine_tune_dataloader = DataLoader(fine_tune_dataset,
                                          sampler=fine_tune_sampler,
                                          batch_size=fine_tune_config["batch_size"])

        t_total = len(fine_tune_dataloader) // fine_tune_config["gradient_accumulation_steps"] * num_train_epochs
        print("Training for t_total:", t_total)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=fine_tune_config["warmup_steps"],
                                                    num_training_steps=t_total)

        global_step, tr_loss, logging_loss = 0, 0.0, 0.0
        for _ in tqdm(range(num_train_epochs)):
            epoch_iterator = tqdm(fine_tune_dataloader, desc="Iteration", disable=local_rank not in [-1, 0])
            for step, batch in enumerate(epoch_iterator):
                model.train()
                batch = tuple(t.to(device) for t in batch)
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": None if args.model_type in ["xlm", "xlm-roberta", "distilbert"] else batch[2],
                    "start_positions": batch[3],
                    "end_positions": batch[4],
                }

                if args.model_type in ["xlnet", "xlm"]:
                    inputs.update({"cls_index": batch[5], "p_mask": batch[6]})
                    if args.version_2_with_negative:
                        inputs.update({"is_impossible": batch[7]})
                if args.model_type == "xlm":
                    inputs["langs"] = batch[7]

                outputs = model(**inputs)

                loss = outputs[0]

                loss = loss.mean()

                if fine_tune_config["gradient_accumulation_steps"] > 1:
                    loss = loss / fine_tune_config["gradient_accumulation_steps"]

                loss.backward()

                tr_loss += loss.item()

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()

                if (step + 1) % fine_tune_config["gradient_accumulation_steps"] == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), fine_tune_config["max_grad_norm"])

                    global_step += 1

                    ## Write loss metrics
                    writer.add_scalar("fine_tune_lr", scheduler.get_lr()[0], global_step)
                    writer.add_scalar("FINE_TUNE_loss", (tr_loss - logging_loss) / fine_tune_config["logging_steps"],
                                      global_step)
                    logging_loss = tr_loss

                if fine_tune_config["save_steps"] > 0 and global_step % fine_tune_config["save_steps"] == 0:
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
                    for lang in args.test_langs:
                        test_results = evaluate(tokenizer, model, test_examples[lang], lang, "test", args.model_type,
                                                out_dir, fine_tune_config["n_best_size"], fine_tune_config["max_answer_length"],
                                                args.version_2_with_negative, args.verbose_logging, args.do_lower_case,
                                                args.null_score_diff_threshold, lang2id)

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


if __name__ == "__main__":

    args = get_arguments()
    set_seed(args)

    """ Config Parameters """
    fine_tune_config = {"pre_train_steps": args.pre_train_steps, "batch_size": args.batch_size, "adam_lr": args.adam_lr,
                        "adam_eps": args.adam_eps, "gradient_accumulation_steps": args.gradient_accumulation_steps,
                        "warmup_steps": args.warmup_steps, "max_grad_norm": args.max_grad_norm,
                        "save_steps": args.save_steps, "weight_decay": args.weight_decay,
                        "logging_steps": args.logging_steps, "eval_batch_size": args.eval_batch_size,
                        "n_best_size": args.n_best_size, "max_answer_length": args.max_seq_length}

    """ Output Directory """

    if args.use_pretrained_model:
        name = "FT/"
    else:
        name = "MONO/"

    out_dir = os.path.join(args.out_dir, name +"SEED_"+str(args.seed)+"/train_"+",".join(args.train_langs)+"-test_"
                           + ",".join(args.test_langs)+ "/fine_tune_"+",".join(args.dev_langs)+"/")

    print("Saving in out_dir:", out_dir)
    writer = SummaryWriter(os.path.join(out_dir, 'runs'))

    """ Cuda/CPU device setup"""

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        n_gpu = 2

    run(args, device, fine_tune_config, out_dir, writer)