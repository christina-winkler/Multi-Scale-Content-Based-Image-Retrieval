import numpy as np
import torch
import random

import PIL
import os
import torchvision
from torchvision import transforms

import torch
from sklearn.metrics import f1_score
from torch.nn.functional import one_hot

import sys
sys.path.append("../../")

from os.path import exists, join
import matplotlib.pyplot as plt
import pdb
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def compute_f1_score(predictions, targets, average='binary'):
    # only used in case of imbalanced dataset

    # Convert predictions and targets to numpy arrays
    predictions_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()

    # Use scikit-learn's f1_score function
    f1 = f1_score(targets_np, predictions_np, average=average)

    return f1

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

def validate(model, val_loader, metric_dict, exp_name, logstep, args):

    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    loss_list=[]
    model.eval()
    cross_entropy_loss = nn.CrossEntropyLoss()

    label_to_idx = {'airplane':0, 'ship':1, 'bridge':2}
    idx_to_label = {0: 'airplane', 1:'ship', 2:'bridge'}

    metric_dict = {'precision': [], 'recall': [], 'accuracy': []}

    with torch.no_grad():
        for batch_idx, item in enumerate(val_loader):

            img = item[0].to(args.device)
            true_labels = item[1].to(args.device)

            # forward loss
            scores = model.forward(img)
            probs = torch.nn.functional.softmax(scores)

            # compute CE loss
            loss = cross_entropy_loss(probs, true_labels)

            # Store CE loss
            loss_list.append(loss.mean().detach().cpu().numpy())

            # retrieve predicted labels
            predicted_labels = torch.argmax(probs, 1)

            # calculate metrics
            precision, recall, accuracy = calculate_multiclass_metrics(predicted_labels, true_labels)

            metric_dict['precision'].append(precision)
            metric_dict['recall'].append(recall)
            metric_dict['accuracy'].append(accuracy)

            if batch_idx == 20:
                break

        savedir = "{}/snapshots/valid/".format(exp_name)

        os.makedirs(savedir, exist_ok=True)

        # visualize training prediction and predicted label and GT
        pred_scores = model(img[0,...].unsqueeze(0))
        pred_probs = torch.nn.functional.softmax(pred_scores)
        pred_label = torch.argmax(pred_probs).item()

        grid = torchvision.utils.make_grid(img[0:9, :, :, :].cpu(), normalize=True, nrow=3)
        plt.figure()
        plt.imshow(grid.permute(1, 2, 0)[:,:,0])
        plt.axis('off')
        plt.title("Predicted Label: {} || GT Label: {}".format(idx_to_label[pred_label],idx_to_label[true_labels[0].item()]))
        # plt.show()
        plt.savefig(savedir + '/classification_validate_{}.png'.format(logstep), dpi=300, bbox_inches='tight')
        plt.close()

    # Print the results
    print(f"Precision: {np.mean(metric_dict['precision'])}")
    print(f"Recall: {np.mean(metric_dict['recall'])}")
    print(f"Accuracy: {np.mean(metric_dict['accuracy'])}")

    return metric_dict, np.mean(loss_list)
