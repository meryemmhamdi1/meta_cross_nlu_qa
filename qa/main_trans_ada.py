from __future__ import print_function
import logging
import os
import random
import numpy as np
from tqdm import tqdm, trange

from transformers_config import *


import torch
from torch.utils.data import DataLoader, RandomSampler

## Optimization
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    get_linear_schedule_with_warmup,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from data_utils import (
    SquadResult,
    SquadV1Processor,
    SquadV2Processor,
    squad_convert_examples_to_features,
    meta_squad_convert_examples_to_features,
    meta_adapt_squad_convert_examples_to_features,
    find_similarities_query_spt
)

import learn2learn as l2l
import pickle


logger = logging.getLogger(__name__)

from utils import no_decay, evaluate, get_arguments, get_meta_arguments
from argparse import Namespace


def run(args, device, fine_tune_config, data_config, opt_config, writer):

    model_name, tokenizer_class, model_class, config_class, qa_class = MODELS_dict[args.trans_model]

    if args.cache_dir == "":
        config = config_class.from_pretrained(args.config_name if args.config_name else model_name,
                                              cache_dir=args.cache_dir if args.cache_dir != "" else None)
    else:
        config = config_class.from_pretrained(args.cache_dir)

    # Set usage of language embedding to True if model is xlm
    if args.model_type == "xlm":
        config.use_lang_emb = True

    if args.cache_dir == "":
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else model_name,
                                                    do_lower_case=args.do_lower_case,
                                                    cache_dir=args.cache_dir if args.cache_dir != "" else None)

        model = qa_class.from_pretrained(model_name,
                                         from_tf=bool(".ckpt" in model_name),
                                         config=config,
                                         cache_dir=args.cache_dir if args.cache_dir != "" else None)
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

    print("CONSTRUCTION of Dev examples for Meta-learning and their conversion to features and dataset")
    if args.use_adapt:
        # Loading from x-metra-ada directory
        load_dir = os.path.join("sim_datasets", 'x-metra-ada')

        ## Meta-training
        with open(os.path.join(load_dir, 'spt_examples.pkl'), 'rb') as f:
            spt_examples = pickle.load(f)

        with open(os.path.join(load_dir, args.dev_langs[0]+'_qry_examples.pkl'), 'rb') as f:
            qry_examples = pickle.load(f)

        ## Meta-adaptation
        with open(os.path.join(load_dir, args.dev_langs[0]+'_tune_spt_examples.pkl'), 'rb') as f:
            tune_spt_examples = pickle.load(f)

        with open(os.path.join(load_dir, args.dev_langs[0]+'_tune_qry_examples.pkl'), 'rb') as f:
            tune_qry_examples = pickle.load(f)

    else:
        # Loading from x-metra directory
        load_dir = os.path.join("sim_datasets", 'x-metra')

        ## Meta-training
        with open(os.path.join(load_dir, 'spt_examples.pkl'), 'rb') as f:
            spt_examples = pickle.load(f)

        with open(os.path.join(load_dir, args.dev_langs[0]+'_qry_examples.pkl'), 'rb') as f:
            qry_examples = pickle.load(f)

    spt_features, qry_features, few_shot_dataset = meta_squad_convert_examples_to_features(
        spt_examples=spt_examples,
        qry_examples=qry_examples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length,
        opt_config=opt_config,
        data_config=data_config,
        is_training=True,
        return_dataset="pt",
        threads=8,
        lang2id=lang2id
    )

    if args.use_adapt:
        tune_spt_features, tune_qry_features, ada_few_shot_dataset = meta_squad_convert_examples_to_features(
            spt_examples=tune_spt_examples,
            qry_examples=tune_qry_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            opt_config=opt_config,
            data_config=data_config,
            is_training=True,
            return_dataset="pt",
            threads=8,
            lang2id=lang2id
        )

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

    num_train_epochs = 5
    t_total = len(train_dataloader) // fine_tune_config["gradient_accumulation_steps"] * num_train_epochs
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": fine_tune_config["weight_decay"],
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=fine_tune_config["adam_lr"], eps=fine_tune_config["adam_eps"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=fine_tune_config["warmup_steps"], num_training_steps=t_total
    )

    local_rank = -1

    if args.use_pretrained_model:
        optimizer.load_state_dict(torch.load(os.path.join(args.pre_trained_model_name, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.pre_trained_model_name, "scheduler.pt")))

        # Load a trained model and vocabulary that you have fine-tuned
        model_dict = torch.load(args.pre_trained_model_name+"pytorch_model.bin")
        model.load_state_dict(model_dict)
        model.to(device)
    else:
        print("TRAINING FROM SCRATCH ...")
        global_step, epochs_trained, tr_loss, logging_loss = 1, 0, 0.0, 0.0

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
        test_results = evaluate(tokenizer, model, test_examples[lang], lang, "test", args.model_type, out_dir,
                                fine_tune_config["n_best_size"], fine_tune_config["max_answer_length"],
                                args.version_2_with_negative, args.verbose_logging, args.do_lower_case,
                                args.null_score_diff_threshold, lang2id, args.data_dir, device, args)

        print("lang:", lang, " test_results:", test_results)
        for key, value in test_results.items():
            writer.add_scalar("Test_{}_{}".format(lang, key), value, 0)

    ### Support + Query
    fst_sampler = RandomSampler(few_shot_dataset)
    fst_dataloader = DataLoader(few_shot_dataset, sampler=fst_sampler, batch_size=opt_config["n_task"])

    t_total = len(fst_dataloader) // fine_tune_config["gradient_accumulation_steps"] * num_train_epochs
    fst_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=fine_tune_config["warmup_steps"],
                                                    num_training_steps=t_total)
    if args.use_adapt:
        ada_sampler = RandomSampler(ada_few_shot_dataset)
        ada_dataloader = DataLoader(ada_few_shot_dataset, sampler=ada_sampler, batch_size=opt_config["n_task"])

        t_total = len(ada_dataloader) // fine_tune_config["gradient_accumulation_steps"] * num_train_epochs

    global_step, tr_loss, logging_loss = 0, 0.0, 0.0
    for _ in tqdm(range(opt_config["epoch"])):
        epoch_iterator = tqdm(fst_dataloader, desc="Iteration", disable=local_rank not in [-1, 0])
        if args.use_adapt:
            ada_epoch_iterator = tqdm(ada_dataloader, desc="Iteration", disable=local_rank not in [-1, 0])
        else:
            ada_epoch_iterator = epoch_iterator
        for step, (batch, ada_batch) in enumerate(zip(epoch_iterator, ada_epoch_iterator)):
            meta_train_error, meta_tune_error = 0.0, 0.0
            ## META-TRAINING BATCH
            batch = tuple(t.to(device) for t in batch)
            qry_inputs = [{
                "input_ids": batch[8][i],
                "attention_mask": batch[9][i],
                "token_type_ids": None if args.model_type in ["xlm", "xlm-roberta", "distilbert"] else batch[10][i],
                "start_positions": batch[11][i],
                "end_positions": batch[12][i],
            } for i in range(0, opt_config["n_task"])]

            spt_inputs = [{
                "input_ids": batch[0][i],
                "attention_mask": batch[1][i],
                "token_type_ids": None if args.model_type in ["xlm", "xlm-roberta", "distilbert"] else batch[2][i],
                "start_positions": batch[3][i],
                "end_positions": batch[4][i],
            } for i in range(0, opt_config["n_task"])]

            ## META-ADAPTATION BATCH
            ada_batch = tuple(t.to(device) for t in ada_batch)
            ada_qry_inputs = [{
                "input_ids": ada_batch[8][i],
                "attention_mask": ada_batch[9][i],
                "token_type_ids": None if args.model_type in ["xlm", "xlm-roberta", "distilbert"] else ada_batch[10][i],
                "start_positions": ada_batch[11][i],
                "end_positions": ada_batch[12][i],
            } for i in range(0, opt_config["n_task"])]

            ada_spt_inputs = [{
                "input_ids": ada_batch[0][i],
                "attention_mask": ada_batch[1][i],
                "token_type_ids": None if args.model_type in ["xlm", "xlm-roberta", "distilbert"] else ada_batch[2][i],
                "start_positions": ada_batch[3][i],
                "end_positions": ada_batch[4][i],
            } for i in range(0, opt_config["n_task"])]

            model.train()

            maml = l2l.algorithms.MAML(model, lr=fst_scheduler.get_lr()[0], first_order=True)
            loss_qry_all = 0.0
            for j in range(opt_config["n_task"]):
                learner = maml.clone()

                for _ in range(0, opt_config["n_up_train_step"]):
                    outputs = learner(**spt_inputs[j])
                    loss = outputs[0]

                    loss = loss.mean()

                    learner.adapt(loss, allow_nograd=True, allow_unused=True)

                # On the query data at the end of n_train optimizations
                loss_qry = learner(**qry_inputs[j])[0].mean()
                loss_qry_all += loss_qry
                meta_train_error += loss_qry.item()

            # Outer loop => Average the accumulated gradients and optimize
            loss_qry_all = loss_qry_all / opt_config["n_task"]
            for p in maml.parameters():
                if p.grad is not None:
                    p.grad.mul_(1.0 / opt_config["n_task"])
            loss_qry_all.backward()
            optimizer.step()
            fst_scheduler.step()
            model.zero_grad()

            loss_qry_tune_all = 0.0
            if args.use_adapt:
                for j in range(opt_config["n_task"]):
                    learner = maml.clone()

                    for _ in range(0, opt_config["n_up_test_step"]):
                        tune_outputs = learner(**ada_spt_inputs[j])
                        tune_loss = tune_outputs[0]

                        tune_loss = tune_loss.mean()
                        learner.adapt(tune_loss, allow_nograd=True, allow_unused=True)

                    # On the query data at the end of n_adapt optimizations
                    loss_qry = learner(**ada_qry_inputs[j])[0].mean()

                    loss_qry_tune_all += loss_qry
                    meta_tune_error += loss_qry.item()

                # Average the accumulated gradients and optimize
                loss_qry_tune_all = loss_qry_tune_all / opt_config["n_task"]
                for p in maml.parameters():
                    if p.grad is not None:
                        p.grad.mul_(1.0 / opt_config["n_task"])
                loss_qry_tune_all.backward()
                optimizer.step()
                fst_scheduler.step()
                model.zero_grad()

            writer.add_scalar("META_train_error", meta_train_error, global_step)
            writer.add_scalar("META_tune_error", meta_tune_error, global_step)

            if fine_tune_config["save_steps"] > 0 and global_step % fine_tune_config["save_steps"] == 0:
                output_dir = os.path.join(out_dir, "META_checkpoint-{}".format(global_step))
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

            if global_step % fine_tune_config["save_steps"] == 0:
                for lang in args.test_langs:
                    test_results = evaluate(tokenizer, model, test_examples[lang], lang, "test", args.model_type,
                                            out_dir, fine_tune_config["n_best_size"], fine_tune_config["max_answer_length"],
                                            args.version_2_with_negative, args.verbose_logging, args.do_lower_case,
                                            args.null_score_diff_threshold, lang2id, args.data_dir, device, args)

                    print("META lang:", lang, " test_results:", test_results)
                    for key, value in test_results.items():
                        writer.add_scalar("META_test_{}_{}".format(lang, key), value, global_step)

            global_step += 1


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    devices = torch.cuda.device_count()
    print("devices:", devices)
    if devices > 1:
        torch.cuda.manual_seed_all(args.seed)


if __name__ == "__main__":

    base_args = get_arguments()
    meta_args = get_meta_arguments()
    args = Namespace(**vars(base_args), **vars(meta_args))

    set_seed(args)

    """ Config Parameters """
    fine_tune_config = {"pre_train_steps": args.pre_train_steps, "batch_size": args.batch_size, "adam_lr": args.adam_lr,
                        "adam_eps": args.adam_eps, "gradient_accumulation_steps": args.gradient_accumulation_steps,
                        "warmup_steps": args.warmup_steps, "max_grad_norm": args.max_grad_norm,
                        "save_steps": args.save_steps, "weight_decay": args.weight_decay,
                        "logging_steps": args.logging_steps, "eval_batch_size": args.eval_batch_size,
                        "n_best_size": args.n_best_size, "max_answer_length": args.max_seq_length}

    data_config = {"k_spt": args.k_spt, "q_qry": args.q_qry}

    opt_config = {"epoch": args.epoch, "n_task": args.n_task, "n_up_train_step": args.n_up_train_step,
                  "n_up_test_step": args.n_up_test_step, "alpha_lr": args.alpha_lr, "beta_lr": args.beta_lr,
                  "gamma_lr": args.gamma_lr, "batch_sz": args.batch_sz}

    """ Output Directory """
    if args.use_adapt:
        flag_adapt = "use_adapt/"
    else:
        flag_adapt = "no_adapt/"

    out_dir = os.path.join(args.out_dir, "MAML_ADA_SEED"+str(args.seed)+"/train_"+",".join(args.train_langs)+"-test_"
                           + ",".join(args.test_langs) + "/l2l/kspt_" + str(data_config["k_spt"]) + "-qqry_"
                           + str(data_config["q_qry"]) + "/en_train_set/" + "/few_shot_" + ",".join(args.dev_langs)+"/"
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

    run(args, device, fine_tune_config, data_config, opt_config, out_dir, writer)
