from datetime import datetime
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import cm
import torch.optim as optim
import os
import json
from skimage.transform import resize

import numpy as np
import random
import pdb
import torchvision
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR

from optimization.validation import validate
from typing import Tuple, Callable

import wandb
os.environ["WANDB_SILENT"] = "true"
import sys
sys.path.append("../../")

# seeding only for debugging
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def train(args, train_loader, valid_loader, model, device):

    args.experiment_dir = os.path.join('runs',
                                        args.modeltype + '_' + args.trainset + '_' + datetime.now().strftime("_%Y_%m_%d_%H_%M_%S"))

    os.makedirs(args.experiment_dir, exist_ok=True)
    config_dict = vars(args)
    with open(args.experiment_dir + '/configs.txt', 'w') as f:
        for key, value in config_dict.items():
            f.write('%s:%s\n' % (key, value))

    # set viz dir
    viz_dir = "{}/snapshots/trainset/".format(args.experiment_dir)
    os.makedirs(viz_dir, exist_ok=True)

    writer = SummaryWriter("{}".format(args.experiment_dir))
    prev_acc_epoch = np.inf
    logging_step = 0
    step = 0
    bpd_valid = 0
    optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=2 * 10 ** 5,
                                                gamma=0.5)
    model.to(device)

    cross_entropy_loss = nn.CrossEntropyLoss()
    metric_dict = {'MSE': [], 'RMSE': [], 'MAE': []}
    params = sum(x.numel() for x in model.parameters() if x.requires_grad)
    print('Nr of Trainable Params on {}:  '.format(device), params)

    # add hyperparameters to tensorboardX logger
    writer.add_hparams({'lr': args.lr, 'bsize':args.bsz}, {'nll_train': - np.inf})


    if torch.cuda.device_count() > 1 and args.train:
        print("Running on {} GPUs!".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
        args.parallel = True

    label_to_idx = {'airplane':0, 'ship':1, 'bridge':2}
    idx_to_label = {0: 'airplane', 1:'ship', 2:'bridge'}

    for epoch in range(args.epochs):
        for batch_idx, item in enumerate(train_loader):

            img = item[0].to(device)
            label = item[1].to(device)

            model.train()
            optimizer.zero_grad()

            # forward loss
            scores = model.forward(img)
            probs = torch.nn.functional.softmax(scores)

            # compute loss
            loss = cross_entropy_loss(probs, label)

            # Compute gradients
            loss.mean().backward()

            # Update model parameters using calculated gradients
            optimizer.step()
            scheduler.step()
            step = step + 1

            print("[{}] Epoch: {}, Train Step: {:01d}/{}, Bsz = {}, CE Loss {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"),
                    epoch, step,
                    args.max_steps,
                    args.bsz,
                    loss.mean().item()))

            if step % args.log_interval == 0:

                with torch.no_grad():

                    model.eval()

                    # visualize training prediction and predicted label and GT
                    pred_scores = model(img[0,...].unsqueeze(0))
                    pred_probs = torch.nn.functional.softmax(pred_scores)
                    pred_label = torch.argmax(pred_probs).item()

                    grid = torchvision.utils.make_grid(img[0:9, :, :, :].cpu(), normalize=True, nrow=3)
                    plt.figure()
                    plt.imshow(grid.permute(1, 2, 0)[:,:,0])
                    plt.axis('off')
                    plt.title("Predicted Label: {} || GT Label: {}".format(idx_to_label[pred_label],idx_to_label[label[0].item()]))
                    # plt.show()
                    plt.savefig(viz_dir + '/classification_train_{}.png'.format(step), dpi=300, bbox_inches='tight')
                    plt.close()


            if step % args.val_interval == 0:
                print('Validating model ... ')
                metric_dict, loss_valid = validate(model,
                                                    valid_loader,
                                                    metric_dict,
                                                    args.experiment_dir,
                                                    "{}".format(step),
                                                    args)

                # save checkpoint only when acc lower than previous model
                if np.mean(metric_dict['accuracy']) < prev_acc_epoch:
                    PATH = args.experiment_dir + '/model_checkpoints/'
                    os.makedirs(PATH, exist_ok=True)
                    torch.save({'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': np.mean(metric_dict['accuracy']).mean()}, PATH+ f"model_epoch_{epoch}_step_{step}.tar")
                    prev_acc_epoch = np.mean(metric_dict['accuracy'])

            logging_step += 1

            if step == args.max_steps:
                break

        if step == args.max_steps:
        #     print("Done Training for {} mini-batch update steps!".format(args.max_steps)
        #     )
        #
        #     if hasattr(model, "module"):
        #         model_without_dataparallel = model.module
        #     else:
        #         model_without_dataparallel = model

            utils.save_model(model,
                             epoch, optimizer, args, time=True)

            print("Saved trained model :)")
            wandb.finish()
            break
