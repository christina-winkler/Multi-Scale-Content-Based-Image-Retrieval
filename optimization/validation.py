import numpy as np
import torch
import random

import PIL
import os
import torchvision
from torchvision import transforms

import sys
sys.path.append("../../")

from os.path import exists, join
import matplotlib.pyplot as plt
import pdb


def validate(model, val_loader, metric_dict, exp_name, logstep, args):

    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    loss_list=[]
    model.eval()

    with torch.no_grad():
        for batch_idx, item in enumerate(val_loader):

            img = item[0].to(device)
            label = item[1].to(device)

            # forward loss
            scores = model.forward(img)
            probs = torch.nn.functional.softmax(scores)

            # compute loss
            loss = cross_entropy_loss(probs, label)

            # Generative loss
            loss_list.append(loss.mean().detach().cpu().numpy())

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
        plt.title("Predicted Label: {} || GT Label: {}".format(idx_to_label[pred_label],idx_to_label[label[0].item()]))
        plt.show()
        plt.savefig(savedir + '/classification_validate_{}.png'.format(step), dpi=300, bbox_inches='tight')
        plt.close()

        with open(savedir + '/metric_dict.txt', 'w') as f:
            for key, value in metric_dict.items():
                f.write('%s:%s\n' % (key, value))

    return metric_dict, np.mean(nll_list)
