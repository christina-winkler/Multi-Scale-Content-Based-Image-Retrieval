import sys
sys.path.append("../../")

import argparse
import torch
import torch.nn as nn
from torchvision import datasets, transforms,models

# Dataset loading
from datasets import dataloading


# Utils
# import utils
import random
import numpy as np
import os

# Models
from models import densenet

# Optimization
from optimization import trainer

# import evaluate
import test
import pdb
from tensorboardX import SummaryWriter

import sys
sys.path.append("../../")


def main(args):

    print(torch.cuda.device_count())
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Initialize device on which to run the model
    if torch.cuda.is_available():
        args.device = torch.device("cuda")
        args.num_gpus = torch.cuda.device_count()
        args.parallel = False

    else:
        args.device = "cpu"

    # args.device = 'cpu'
    print("Device", args.device)
    # args.device = "cpu"

    # Build name of current model
    if args.modelname is None:
        args.modelname = "{}_{}_bsz{}_lr{:.4f}".format(args.modeltype, args.trainset,
                                                                   args.bsz, args.lr)

    # load data
    train_loader, valid_loader, test_loader, args = dataloading.load_data(args)
    in_channels = next(iter(test_loader))[0].shape[0]
    height, width = next(iter(train_loader))[0].shape[1], next(iter(train_loader))[0].shape[2]

    if args.modeltype == 'densenet121':
        
        model = densenet.DensNet(num_classes=3, num_channels=3)

        # adapt FC classification layer
        model.classifier = nn.Linear(in_features=1024, out_features=3)

        params = sum(x.numel() for x in model.parameters() if x.requires_grad)
        print('Nr of Trainable Params on {}:  '.format(args.device), params)

        trainer.train(args, train_loader, valid_loader, model, device=args.device)

    print("Start training {} on {}:".format(args.modeltype, args.trainset))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # train configs
    parser.add_argument("--modeltype", type=str, default="densenet121",
                        help="Specify modeltype you would like to train.")
    parser.add_argument("--model_path", type=str, default="runs/",
                        help="Directory where models are saved.")
    parser.add_argument("--modelname", type=str, default=None,
                        help="Sepcify modelname to be tested.")
    parser.add_argument("--epochs", type=int, default=20000,
                        help="number of epochs")
    parser.add_argument("--max_steps", type=int, default=2000000,
                        help="For training on a large dataset.")
    parser.add_argument("--log_interval", type=int, default=250,
                        help="Interval in which results should be logged.")
    parser.add_argument("--val_interval", type=int, default=250,
                        help="Interval in which model should be validated.")

    # runtime configs
    parser.add_argument("--visual", action="store_true",
                        help="Visualizing the samples at test time.")
    parser.add_argument("--noscaletest", action="store_true",
                        help="Disable scale in coupling layers only at test time.")
    parser.add_argument("--noscale", action="store_true",
                        help="Disable scale in coupling layers.")
    parser.add_argument("--testmode", action="store_true",
                        help="Model run on test set.")
    parser.add_argument("--resume", action="store_true",
                        help="If training should be resumed.")

    # hyperparameters
    parser.add_argument("--crop_size", type=int, default=500,
                        help="Crop size when random cropping is applied.")
    parser.add_argument("--patch_size", type=int, default=500,
                        help="Training patch size.")
    parser.add_argument("--bsz", type=int, default=16, help="batch size")
    parser.add_argument("--lr", type=float, default=0.0002,
                        help="learning rate")
    parser.add_argument("--filter_size", type=int, default=512//2,
                        help="filter size NN in Affine Coupling Layer")
    parser.add_argument("--nb", type=int, default=16,
                        help="# of residual-in-residual blocks LR network.")
    parser.add_argument("--condch", type=int, default=128//8,
                        help="# of residual-in-residual blocks in LR network.")

    # data
    parser.add_argument("--datadir", type=str, default="/home/christina/Documents/classification/datasets/")
    parser.add_argument("--trainset", type=str, default="resisc45",
                        help="Dataset to train the model on [fair1m, resisc45, sentinel2].")


    args = parser.parse_args()
    main(args)
