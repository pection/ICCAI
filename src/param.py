# Parameters for LXMERT training and evaluation. Modified from original
# LXMERT code to add more parameters.

import argparse
import random

import numpy as np
import torch


def get_optimizer(optim):
    # Bind the optimizer
    if optim == "rms":
        print("Optimizer: Using RMSProp")
        optimizer = torch.optim.RMSprop
    elif optim == "adam":
        print("Optimizer: Using Adam")
        optimizer = torch.optim.Adam
    elif optim == "adamax":
        print("Optimizer: Using Adamax")
        optimizer = torch.optim.Adamax
    elif optim == "sgd":
        print("Optimizer: sgd")
        optimizer = torch.optim.SGD
    elif "bert" in optim:
        optimizer = "bert"  # The bert optimizer will be bind later.
    elif optim == "none":
        optimizer = lambda x, y: None
    else:
        assert False, "Please add your optimizer %s in the list." % optim

    return optimizer


def parse_args():
    parser = argparse.ArgumentParser()

    # Data Splits ONLY USED IN PRETRAINING
    parser.add_argument("--train", default=None)
    parser.add_argument("--valid", default=None)

    # Test IS USED in all scripts
    parser.add_argument("--test", default=None)
    # new args
    parser.add_argument("--clevr_config", default="src/clevr_tasks/clevr_config.yaml")
    parser.add_argument(
        "--dataset",
        default="clevr_ho",
        choices=["clevr_ho", "clevr_ho_v2", "clevr", "clevr_atom_ho_exist"],
    )
    parser.add_argument("--ho_idx", type=int)
    parser.add_argument(
        "--max_questions",
        type=int,
        default=None,
        help="If provided, limit the dataset to *just* the first this many questions",
    )
    parser.add_argument(
        "--debug",
        action="store_const",
        default=False,
        const=True,
        help="Limit to 1000 train/val, set dataloader threads to 0, set val batch size for pretrain to only 256",
    )
    parser.add_argument(
        "--experiment_name",
        default="no_text_no_vis",
        choices=[
            "no_text_no_vis",
            "exp_text_no_vis",
            "no_text_exp_vis",
            "exp_text_exp_vis",
            "zero_shot_text",
        ],
        type=str,
        help="The type of experiment to run; only relevant for CLEVR-HO",
    )

    parser.add_argument(
        "--preemptable",
        action="store_const",
        default=False,
        const=True,
        help="Modify clevr.py behaviour so that it can be run safely on a system that may pre-empt & requeue the job",
    )
    parser.add_argument(
        "--load_clevr_pretrained",
        action="store_const",
        default=False,
        const=True,
        help="If true, make the required changes needed to load a checkpoint pretrained on CLEVR/CLEVR-HO; namely, a different answer table needs to be loaded.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=20,
        help="Max sequence length in clevr.py; largest for CLEVR appears to actually be 49. Default is 20 (used for GQA).",
    )

    # Training Hyper-parameters
    parser.add_argument("--batchSize", dest="batch_size", type=int, default=256)
    parser.add_argument("--optim", default="bert")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--train_steps",
        type=int,
        default=None,
        help="Max number of train steps. If both this and epochs are specified, then the min is used. Note! Default is 10 epochs.",
    )
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training configuration
    # Adjust the number of times that validation is performed
    parser.add_argument(
        "--num_val_evals",
        type=int,
        default=11,
        help="How many times validation should be performed. -1 for after every epoch. Validation is always run at the end of the 0th and last epochs, so must be at least 2 if non-negative",
    )
    parser.add_argument("--fp16", action="store_const", default=False, const=True)
    parser.add_argument("--multiGPU", action="store_const", default=False, const=True)
    parser.add_argument("--numWorkers", dest="num_workers", default=0)

    # Debugging
    parser.add_argument("--output", type=str, default="snap/test")
    parser.add_argument("--fast", action="store_const", default=False, const=True)
    parser.add_argument("--tiny", action="store_const", default=False, const=True)
    parser.add_argument("--tinier", action="store_const", default=False, const=True)
    parser.add_argument("--tqdm", action="store_const", default=False, const=True)

    # Model Loading
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="Load the model (usually the fine-tuned model).",
    )
    parser.add_argument(
        "--loadLXMERT",
        dest="load_lxmert",
        type=str,
        default=None,
        help="Load the pre-trained LXMERT model.",
    )
    parser.add_argument(
        "--loadLangLXMERT",
        dest="load_lang_lxmert",
        type=str,
        default=None,
        help="Load the pre-trained LXMERT language encoder, only. Safest to combine with --fromScratch to ensure rest of weights randomly initialized.",
    )
    parser.add_argument(
        "--loadVisLXMERT",
        dest="load_vis_lxmert",
        type=str,
        default=None,
        help="Load the pre-trained LXMERT vision encoder, only. Safest to combine with --fromScratch to ensure rest of weights randomly initialized.",
    )
    parser.add_argument(
        "--loadLXMERTQA",
        dest="load_lxmert_qa",
        type=str,
        default=None,
        help="Load the pre-trained LXMERT model with QA answer head.",
    )
    parser.add_argument(
        "--fromScratch",
        dest="from_scratch",
        action="store_const",
        default=False,
        const=True,
        help="If none of the --load, --loadLXMERT, --loadLXMERTQA is set, "
        "the model would be trained from scratch. If --fromScratch is"
        " not specified, the model would load BERT-pre-trained weights by"
        " default. ",
    )

    # Optimization
    parser.add_argument(
        "--mceLoss", dest="mce_loss", action="store_const", default=False, const=True
    )

    # LXRT Model Config
    # Note: LXRT = L, X, R (three encoders), Transformer
    parser.add_argument(
        "--llayers", default=9, type=int, help="Number of Language layers"
    )
    parser.add_argument(
        "--xlayers", default=5, type=int, help="Number of CROSS-modality layers."
    )
    parser.add_argument(
        "--rlayers", default=5, type=int, help="Number of object Relationship layers."
    )

    # LXMERT Pre-training Config
    parser.add_argument(
        "--taskMatched",
        dest="task_matched",
        action="store_const",
        default=False,
        const=True,
    )
    parser.add_argument(
        "--taskMaskLM",
        dest="task_mask_lm",
        action="store_const",
        default=False,
        const=True,
    )
    parser.add_argument(
        "--taskObjPredict",
        dest="task_obj_predict",
        action="store_const",
        default=False,
        const=True,
    )
    parser.add_argument(
        "--taskQA", dest="task_qa", action="store_const", default=False, const=True
    )
    parser.add_argument(
        "--visualLosses", dest="visual_losses", default="obj,attr,feat", type=str
    )
    parser.add_argument("--qaSets", dest="qa_sets", default=None, type=str)
    parser.add_argument(
        "--wordMaskRate", dest="word_mask_rate", default=0.15, type=float
    )
    parser.add_argument("--objMaskRate", dest="obj_mask_rate", default=0.15, type=float)

    # Parse the arguments.
    args = parser.parse_args()

    # Bind optimizer class.
    args.optimizer = get_optimizer(args.optim)

    print("NOT Setting any seeds.")
    if 0 <= args.num_val_evals < 2:
        raise ValueError("--num_val_evals must be either negative, or at least 2")

    return args


args = parse_args()
