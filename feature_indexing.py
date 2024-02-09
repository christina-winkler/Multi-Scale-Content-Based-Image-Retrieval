import faiss

from datetime import datetime
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import cm

# Dataset loading
from datasets import dataloading

import random
import numpy as np
import os
import json
import pickle
import pdb
import torchvision
import argparse

# Models
from models import densenet

import wandb
os.environ["WANDB_SILENT"] = "true"
import sys
sys.path.append("../../")



def feature_indexing(args, train_loader, model, device):
    """
    Indexes features using FAISS (Facebook AI Similarity Search) indexing.
    https://www.pinecone.io/learn/series/faiss/composite-indexes/
    Parameters:
    - args (Namespace): Command-line arguments or configuration settings.
    - train_loader (DataLoader): DataLoader for the training dataset.
    - model (nn.Module): Neural network model that extracts features.
    - device (str): Device on which the model and indexing should be performed.

    Returns:
    - None: The function performs feature indexing using FAISS and does not return any value.
    """
    # adapted from: https://towardsdatascience.com/a-hands-on-introduction-to-image-retrieval-in-deep-learning-with-pytorch-651cd6dba61e
    dimension = 2048
    n_centroids = 3
    faiss_index = faiss.IndexFlatL2(dimension) # quantizer
    # faiss_index = faiss.IndexIVFPQ(quantizer, dimension, n_centroids, 16, 8)   # build the index

    im_indices = []

    # store image representations
    with torch.no_grad():
        for batch_idx, item in enumerate(train_loader):

            img = item[0].to(device)
            label = item[1].to(device)
            preds = model.features(img)
            preds = preds.view(-1)
            preds = np.array([preds.cpu().numpy()])
            faiss_index.add(preds) #add the representation to index
            im_indices.append(str(label.item()))   #store the image label to find it later on

            # test retrieval with a query
            _, I = faiss_index.search(preds, 5)
            print("Retrieved Image: {}".format(im_indices[I[0][0]]), batch_idx)

        # serialize faiss index
        with open('faiss_index.pickle','wb') as f:
            pickle.dump(faiss_index, f)

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

    # Build name of current model
    if args.modelname is None:
        args.modelname = "{}_{}_bsz{}_lr{:.4f}".format(args.modeltype, args.trainset,
                                                                   args.bsz, args.lr)

    # load data
    train_loader, valid_loader, test_loader, args = dataloading.load_data(args)
    in_channels = next(iter(test_loader))[0].shape[0]
    height, width = next(iter(train_loader))[0].shape[1], next(iter(train_loader))[0].shape[2]

    model = densenet.DensNet(num_classes=3, num_channels=3)

    # adapt FC classification layer
    model.classifier = nn.Linear(in_features=32, out_features=3)
    model.features[-2] = nn.Conv2d(512,32,3, padding=1)
    model.features[-1] = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    # load stored model
    modelname = 'model_epoch_10_step_2000'
    modelpath = '/home/christina/Documents/classification/runs/densenet121_resisc45__2024_02_03_17_31_54/model_checkpoints/{}.tar'.format(modelname)
    ckpt = torch.load(modelpath)

    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(args.device)

    params = sum(x.numel() for x in model.parameters() if x.requires_grad)
    print('Nr of Trainable Params on {}:  '.format(args.device), params)

    feature_indexing(args, train_loader, model, device=args.device)
