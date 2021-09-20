import sys

sys.path.append('.')

import os
import torch
import argparse
import numpy as np
import pandas as pd
from model import NN
from test import test
from utils import SE3Loss
from tqdm import tqdm, trange
from datetime import datetime
import matplotlib.pyplot as plt
from dataset import get_station_dataset


def plot_lines(df, columns, colors, ax, alpha=0.25, show_range=True, window_size=1):
    for color, column in zip(colors, columns):
        agg_df = df.groupby('epoch')[column]

        if window_size > 1:
            agg_df = agg_df.rolling(window_size)

        means = agg_df.mean()
        ax.plot(np.arange(len(means)), means, c=color)

        if show_range:
            mins = agg_df.min()
            maxs = agg_df.max()
            ax.fill_between(x=np.arange(len(means)),
                            y1=mins, y2=maxs, alpha=alpha)

    ax.legend(columns)


def train():
    """Train the neural network model, save weights and plot loss over time."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, help='name of the model',
                        default='model_' + str(datetime.now()))
    parser.add_argument('-f', '--filename', type=str,
                        help='name of the dataset (.h5 file)', default='../data/station/random_dataset80m.h5')
    parser.add_argument('-e', '--epochs', type=int,
                        help='number of epochs of the training phase', default=60)
    parser.add_argument('-bs', '--batch-size', type=int,
                        help='size of the batches of the training data', default=64)
    parser.add_argument('-lr', '--learning-rate', type=float,
                        help='learning rate used for the training phase', default=1e-3)
    parser.add_argument('-lmb', '--lambda-loss', type=float,
                        help='lambda paramter used for the loss', default=10)
    parser.add_argument('-d', '--device', type=str, help=argparse.SUPPRESS,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    name = args.name
    epochs = args.epochs
    device = args.device
    lmbd = args.lambda_loss
    filename = args.filename
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    model_path = './station/model/' + name
    log_path = model_path + '/log'
    checkpoint_path = model_path + '/checkpoints'

    for k, v in vars(args).items():
        print('%s = "%s"' % (k, str(v)))

    if not os.path.exists(model_path):
        os.mkdir(model_path)

    if not os.path.exists(log_path):
        os.mkdir(log_path)

    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    # Dataset & Model

    train_dataset = get_station_dataset(filename, 'camera', 'ooi_noisy',
                                        split='train', augment=True, device=device)

    val_dataset = get_station_dataset(filename, 'camera', 'ooi_rel_pose',
                                      split='validation', augment=False, device=device)

    model = NN(in_channels=3, out_channels=7).to(device)

    # Optimizer & Loss

    loss_function = SE3Loss(lmbd, pos_loss=torch.nn.functional.l1_loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, cooldown=3, min_lr=1e-6)

    # Training

    history = pd.DataFrame()
    training_steps = len(train_dataset) // batch_size
    epochs_logger = trange(1, epochs + 1, desc='epoch')

    for epoch in epochs_logger:
        steps_logger = tqdm(train_dataset.batches(batch_size, shuffle=True),
                            desc='train step',
                            total=training_steps)
        val_metrics = [np.nan] * 8
        for step, (x, y) in enumerate(steps_logger, start=1):
            model.train()
            optimizer.zero_grad()

            pred = model(x)
            # todo: this sum() is temp and should be impemented within SE3Loss
            loss = loss_function(pred, y).sum()

            length = (x.size(0) // 2) * 2
            indices = torch.arange(length).view(-1, 2)

            sc_loss = loss_function(pred[indices[:, 0]],
                                    pred[indices[:, 1]]).sum()

            loss += sc_loss * 1

            loss.backward()
            optimizer.step()

            loss = loss.item()

            if step == training_steps:
                val_metrics = test(model, val_dataset, 4 * batch_size,
                                   loss_function.lmbd)
                scheduler.step(val_metrics[0])

            history = history.append({
                'epoch': epoch,
                'step': step,
                'loss': loss,
                'sc_loss': sc_loss.item(),
                'val_loss': val_metrics[0],
                'val_r2_x': val_metrics[1],
                'val_r2_y': val_metrics[2],
                'val_dquat': val_metrics[3]
            }, ignore_index=True)

            mean_values = history.query('epoch == ' + str(epoch)
                                        ).mean(axis=0, skipna=True)
            mean_metrics = mean_values[[
                'loss', 'sc_loss', 'val_loss',
                'val_r2_x', 'val_r2_y',
                'val_dquat'
            ]].tolist()

            log_str = 'L: %.4f SCL: %.4f' % tuple(mean_metrics[:2])
            steps_logger.set_postfix_str(log_str)

            if step == training_steps:
                log_str += ' VL: %.4f VR2x: %.4f VR2y: %.4f VDquat: %.4f' % tuple(
                    mean_metrics[2:])
                print
                epochs_logger.set_postfix_str(log_str)

        checkpoint_name = '%d_%.4f_state_dict.pth' % (
            epoch, mean_metrics[2])
        torch.save(model.state_dict(), checkpoint_path +
                   '/' + checkpoint_name)
        torch.save(model.state_dict(), checkpoint_path + '/last.pth')

    history.to_csv(log_path + '/history.csv')

    _, ax = plt.subplots(1, 1, figsize=(10, 10))
    plot_lines(history, ['loss', 'val_loss'], ['blue', 'orange'], ax)
    plt.savefig(log_path + '/loss.png')
    plt.savefig(log_path + '/loss.svg')
    plt.show()


if __name__ == '__main__':
    train()
