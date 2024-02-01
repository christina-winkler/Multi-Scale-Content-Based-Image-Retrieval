import argparse
import torch
import torch.nn as nn
from torchvision import datasets, transforms,models
import torchvision
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Dataset loading
from datasets import dataloading

# Utils
# import utils
import random
import numpy as np
import os
from datetime import datetime

# Models
from models import densenet

# Optimization
from optimization import trainer

from os.path import exists, join
import matplotlib.pyplot as plt
import pdb
import torch.nn as nn
from tensorboardX import SummaryWriter
import torch.nn.functional as F

import sys
sys.path.append("../../")

def calculate_multiclass_metrics(predictions, targets, average='weighted'):
    """
    Calculate precision, recall, and accuracy for multiclass classification.

    Parameters:
        predictions (torch.Tensor): Tensor containing predicted labels (integer values).
        targets (torch.Tensor): Tensor containing true labels (integer values).
        average (str): Type of averaging for precision, recall, and F1. Options: 'micro', 'macro', 'weighted'.

    Returns:
        float: Precision value.
        float: Recall value.
        float: Accuracy value.
    """
    # Convert predictions and targets to numpy arrays
    predictions_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()

    # Calculate precision, recall, and F1 score using scikit-learn functions
    precision, recall, _, _ = precision_recall_fscore_support(targets_np, predictions_np, average=average)
    accuracy = accuracy_score(targets_np, predictions_np)

    return precision, recall, accuracy

def compute_f1_score(predictions, targets, average='binary'):
    # only used in case of imbalanced dataset

    # Convert predictions and targets to numpy arrays
    predictions_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()

    # Use scikit-learn's f1_score function
    f1 = f1_score(targets_np, predictions_np, average=average)

    return f1

def get_densnet_features(model, images):

    feat_layers = model.features


    return features


def test(args, test_loader, model, device):

    loss_list=[]

    cross_entropy_loss = nn.CrossEntropyLoss()

    label_to_idx = {'airplane':0, 'ship':1, 'bridge':2}
    idx_to_label = {0: 'airplane', 1:'ship', 2:'bridge'}

    metric_dict = {'precision': [], 'recall': [], 'accuracy': []}
    args.exp_name = os.path.join('experiments/',
                                  args.modeltype + '_' + args.trainset + '_' + datetime.now().strftime("_%Y_%m_%d_%H_%M_%S"))
    os.makedirs(args.exp_name, exist_ok=True)

    model.eval()
    for batch_idx, item in enumerate(test_loader):

        img = item[0].to(args.device)
        true_labels = item[1].to(args.device)

        # forward loss
        logits = model.forward(img)
        prob_scores = torch.nn.functional.softmax(logits)

        # compute CE loss
        loss = cross_entropy_loss(prob_scores, true_labels)
        loss.backward()

        # store CE loss
        loss_list.append(loss.mean().detach().cpu().numpy())

        # retrieve predicted labels
        predicted_labels = torch.argmax(prob_scores, 1)

        # calculate metrics
        precision, recall, accuracy = calculate_multiclass_metrics(predicted_labels, true_labels)

        metric_dict['precision'].append(precision)
        metric_dict['recall'].append(recall)
        metric_dict['accuracy'].append(accuracy)

        savedir = "{}/snapshots/".format(args.exp_name)
        os.makedirs(savedir, exist_ok=True)

        # visualize training prediction and predicted label and GT
        pred_scores = model(img[0,...].unsqueeze(0))
        pred_probs = torch.nn.functional.softmax(pred_scores)
        pred_label = torch.argmax(pred_probs).item()

        # compute gradcam
        feats = model.features[:-1](img)
        feats.register_hook(model.gradients)

        a_k = model.gradients[0].mean(dim=-1).mean(dim=-1) # compute global pooling channel average over gradients
        pooled_gradients = torch.mean(model.gradients[0], dim=[0, 2, 3])

        # weight the activations channels by corresponding gradients
        activations = model.activations
        grad_cam = torch.sum(a_k[:,:,None,None] * activations, dim=1)

        plt.imshow(grad_cam.permute(1,2,0).detach().cpu().numpy())
        plt.show()

        # create superposition
        quit()

        grid = torchvision.utils.make_grid(img[0:9, :, :, :].cpu(), normalize=True, nrow=3)
        plt.figure()
        plt.imshow(grid.permute(1, 2, 0)[:,:,0])
        plt.axis('off')
        plt.title("Predicted Label: {} || GT Label: {}".format(idx_to_label[pred_label],idx_to_label[true_labels[0].item()]))
        plt.savefig(savedir + '/classification_validate_{}.png'.format(batch_idx), dpi=300, bbox_inches='tight')
        plt.close()

    # Print the results
    print(f"Precision: {np.mean(metric_dict['precision'])}")
    print(f"Recall: {np.mean(metric_dict['recall'])}")
    print(f"Accuracy: {np.mean(metric_dict['accuracy'])}")

    return None

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

        # load stored model
        modelname = 'model_epoch_2_step_500'
        modelpath = '/home/christina/Documents/classification/runs/densenet121_resisc45__2024_02_01_17_05_30/model_checkpoints/{}.tar'.format(modelname)
        ckpt = torch.load(modelpath)

        model.load_state_dict(ckpt['model_state_dict'])
        model = model.to(args.device)

        params = sum(x.numel() for x in model.parameters() if x.requires_grad)
        print('Nr of Trainable Params on {}:  '.format(args.device), params)
        print("Start testing {} on {}:".format(args.modeltype, args.trainset))
        test(args, test_loader, model, device=args.device)


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

# TODO: store features over test set (output before classifier layer) in dictionary
# TODO: then cluster them (KDTree ?)
