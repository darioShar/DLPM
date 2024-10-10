import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import random

import torch
import numpy as np
import torch.utils.tensorboard as tb

from diffusion import Diffusion

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--dataset", type=str, default="data", help="Path for dataset."
    )
    parser.add_argument(
        "--exp", type=str, default="exp", help="Path for saving running related data."
    )
    parser.add_argument(
        "--doc",
        type=str,
        required=False,
        default="",
        help="A string for documentation purpose."
        "Will be the name of the log folder.",
    )
    parser.add_argument(
        "--comment", type=str, default="", help="A string for experiment comment"
    )
    parser.add_argument(
        "--verbose",
        type=str,
        default="info",
        help="Verbose level: info | debug | warning | critical",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Whether to produce samples from the model",
    )
    parser.add_argument(
        "--train_sample",
        action="store_true",
        help="Whether to make samples during training",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Whether to resume training"
    )
    parser.add_argument(
        "--bpd",
        default=None,
        choices=['train', 'test', None],
        help="type of bpd(log-likelihood(bits/dim)) computation",
    )
    parser.add_argument(
        "--bpd_nfe", type=int, default=0, help="nfe(sampling step) used in bpd evaluation"
    )
    parser.add_argument(
        "--imputation",
        action="store_true",
        help="Whether to perform imputation sampling",
    )
    parser.add_argument(
        "--fid",
        default=None,
        choices=['train', 'test', None],
        help="type of fid calculation",
    )
    parser.add_argument(
        "--sample_type",
        default='sde',
        choices=['sde', 'ode', 'sde_imputation', None],
        help="sampling method",
    )
    parser.add_argument(
        "--nfe", type=int, default=500, help="number of function evaluations"
    )
    parser.add_argument(
        "--ni",
        action="store_true",
        help="No interaction. Suitable for Slurm Job launcher",
    )    
    parser.add_argument(
        "-i",
        "--image_folder",
        type=str,
        default="images",
        help="The folder name of samples",
    )
    parser.add_argument(
        "--ddp",
        action="store_true",
        help="Whether to ddp train",
    )
    args = parser.parse_args()
    args.log_path = os.path.join(args.exp, "logs", args.doc)
    
    if args.ddp:
        args.local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(args.local_rank)
        torch.cuda.empty_cache()
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    else:
        args.local_rank = 0

    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    tb_path = os.path.join(args.exp, "tensorboard", args.doc)

    if not args.sample:
        if not args.resume:
            if True and args.local_rank == 0:
                if os.path.exists(args.log_path):
                    overwrite = False
                    if args.ni:
                        overwrite = True
                    else:
                        response = input("Folder already exists. Overwrite? (Y/N)")
                        if response.upper() == "Y":
                            overwrite = True

                    if overwrite:
                        shutil.rmtree(args.log_path)
                        os.makedirs(args.log_path)
                    else:
                        print("Folder exists. Program halted.")
                        sys.exit(0)
                else:
                    os.makedirs(args.log_path)
            try:
                with open(os.path.join(args.log_path, "config.yml"), "w") as f:
                    yaml.dump(new_config, f, default_flow_style=False)
            except:
                pass

        new_config.tb_logger = tb.SummaryWriter(log_dir=tb_path)
        # setup logger
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        try:
            handler2 = logging.FileHandler(os.path.join(args.log_path, "stdout.txt"))
        except:
            pass
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        try:
            handler2.setFormatter(formatter)
        except:
            pass
        logger = logging.getLogger()
        logger.addHandler(handler1)
        try:
            logger.addHandler(handler2)
        except:
            pass
        logger.setLevel(level)

    else:
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.setLevel(level)

        if args.sample:
            if args.bpd == None and args.local_rank == 0:
                os.makedirs(os.path.join(args.exp, "image_samples"), exist_ok=True)
                args.image_folder = os.path.join(
                    args.exp, "image_samples", args.image_folder
                )
                if not os.path.exists(args.image_folder):
                    os.makedirs(args.image_folder)
                else:
                    if not args.fid:
                        overwrite = False
                        if args.ni:
                            overwrite = True
                        else:
                            response = input(
                                f"Image folder {args.image_folder} already exists. Overwrite? (Y/N)"
                            )
                            if response.upper() == "Y":
                                overwrite = True

                        if overwrite:
                            shutil.rmtree(args.image_folder)
                            os.makedirs(args.image_folder)
                        else:
                            print("Output image folder exists. Program halted.")
                            sys.exit(0)
            else:
                args.image_folder = os.path.join(
                    args.exp, "image_samples", args.image_folder
                )
                pass

    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()
    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))

    try:
        runner = Diffusion(args, config)

        if args.sample:
            runner.sample()
        else:
            runner.train()
            
    except Exception:
        logging.error(traceback.format_exc())

    return 0


if __name__ == "__main__":
    sys.exit(main())