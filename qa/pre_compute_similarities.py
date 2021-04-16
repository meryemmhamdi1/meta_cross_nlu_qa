from __future__ import print_function
import logging
import random
import numpy as np
import pickle
import argparse
import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


from data_utils import (
    SquadV1Processor,
    SquadV2Processor,
    find_similarities_query_spt
)

logger = logging.getLogger(__name__)


def run(version_2_with_negative, data_dir, train_langs, dev_langs):

    processor = SquadV2Processor() if version_2_with_negative else SquadV1Processor()

    spt_examples = processor.get_train_examples(data_dir, task="tydiqa", languages=train_langs)
    qry_examples = processor.get_dev_examples(data_dir, task="tydiqa", languages=dev_langs)

    print("CONSTRUCTION of meta-training examples for X-METRA")
    print("len(spt_examples):", len(spt_examples))
    print("len(qry_examples):", len(qry_examples))

    spt_examples, qry_examples = find_similarities_query_spt(spt_examples, qry_examples)
    with open('sim_datasets/x-metra/spt_examples.pkl', 'wb') as f:
        pickle.dump(spt_examples, f)

    with open('sim_datasets/x-metra/'+dev_langs[0]+'_qry_examples.pkl', 'wb') as f:
        pickle.dump(qry_examples, f)

    random.shuffle(qry_examples)
    qry_size = int(len(qry_examples)*0.60)
    meta_train_qry = qry_examples[:qry_size]
    meta_ada_tune = qry_examples[qry_size:]

    print("CONSTRUCTION of meta-training examples and their conversion to features and dataset>>>>>")
    spt_examples, meta_train_qry = find_similarities_query_spt(spt_examples, meta_train_qry)

    with open('sim_datasets/x-metra-ada/spt_examples.pkl', 'wb') as f:
        pickle.dump(spt_examples, f)

    with open('sim_datasets/x-metra-ada/'+dev_langs[0]+'_qry_examples.pkl', 'wb') as f:
        pickle.dump(meta_train_qry, f)

    print("CONSTRUCTION of meta-adaptation examples and their conversion to features and dataset")

    random.shuffle(meta_ada_tune)
    ada_qry_size = int(len(meta_ada_tune)*0.60)
    meta_ada_spt = meta_ada_tune[:ada_qry_size]
    meta_ada_qry = meta_ada_tune[ada_qry_size:]
    meta_ada_spt, meta_ada_qry = find_similarities_query_spt(meta_ada_spt, meta_ada_qry)

    with open('sim_datasets/x-metra-ada/'+dev_langs[0]+'_tune_spt_examples.pkl', 'wb') as f:
        pickle.dump(meta_ada_spt, f)

    with open('sim_datasets/x-metra-ada/'+dev_langs[0]+'_tune_qry_examples.pkl', 'wb') as f:
        pickle.dump(meta_ada_qry, f)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    devices = torch.cuda.device_count()
    if devices > 1:
        torch.cuda.manual_seed_all(args.seed)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-langs", help="train languages list", nargs="+", default=[])
    parser.add_argument("--dev-langs", help="dev languages list", nargs="+", default=[])
    parser.add_argument("--test-langs", help="test languages list", nargs="+", default=[])

    parser.add_argument('--data-dir', help='Path of data',  default="")

    parser.add_argument("--version-2-with-negative", action="store_true",
                        help="If true, the SQuAD examples contain some that do not have an answer.")

    parser.add_argument('--seed', help="Random seed for initialization", type=int, default=42)
    parser.add_argument('--local_rank', type=int, help='local rank for DistributedDataParallel', default=-1)
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = get_arguments()
    set_seed(args)

    """ Cuda/CPU device setup"""

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        n_gpu = 1

    #run(args.version_2_with_negative, args.data_dir, args.train_langs, args.dev_langs)
    for lang in ["ar", "bn", "fi", "id", "ru", "sw", "te"]:
        print("Preparing for lang:", lang)
        run(args.version_2_with_negative, args.data_dir, ["en"], [lang])
