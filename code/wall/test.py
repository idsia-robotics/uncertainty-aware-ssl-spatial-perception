import sys
sys.path.append('.')

import os
import torch
import argparse
import numpy as np
from model import NN
from tqdm import tqdm
from utils import SE3Loss
from sklearn.metrics import r2_score
from dataset import get_wall_dataset


def test(model, dataset, batch_size, lmbd, verbose=True):
    '''Computes loss, *r2 and Dquat of the model applied to the dataset.

    Args:
            model: a pytorch neural network model.
            dataset: an HDF5Dataset.
            batch_size: size of the batches of the dataset.
            lmbd: the lambda scale used for the positional part of the loss.
            verbose: if set prints various information.

    Returns:
            loss, *r2 and Dquat of the model applied to the dataset.
    '''
    ys = []
    preds = []
    steps_logger = dataset.batches(batch_size)
    se3loss = SE3Loss(lmbd, pos_loss=torch.nn.functional.l1_loss)

    if verbose:
        steps_logger = tqdm(steps_logger,
                            desc=dataset.split + ' step',
                            total=len(dataset) // batch_size)
    model.eval()
    with torch.no_grad():
        for x, y, _ in steps_logger:
            y = y.cpu().numpy()
            pred = model(x).cpu().numpy()

            ys.append(y)
            preds.append(pred)

        y = np.concatenate(ys).astype(np.float32)
        pred = np.concatenate(preds).astype(np.float32)

        r2 = [r2_score(y_pred=pred[:, 0], y_true=y[:, 0])]

        y = torch.tensor(y)
        pred = torch.tensor(pred)

        # todo:
        loss = se3loss(pred, y).sum().item()
        drot = se3loss(pred, y)[1].item()

    return (loss,) + tuple(r2) + (drot,)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, help='name of the model',
                        default='model_uncert50_sc1')
    parser.add_argument('-f', '--filename', type=str,
                        help='name of the dataset (.h5 file)', default='../data/wall/dataset10m.h5')
    parser.add_argument('-s', '--split', type=str, help='dataset split, one of train, validation or test',
                        default='test')
    parser.add_argument('-bs', '--batch-size', type=int,
                        help='size of the batches of the training data', default=128)
    parser.add_argument('-v', '--verbose', help='if not set prints various information',
                        action='store_false', default=True)
    parser.add_argument('-d', '--device', type=str, help=argparse.SUPPRESS,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    name = args.name
    split = args.split
    device = args.device
    verbose = args.verbose
    filename = args.filename
    batch_size = args.batch_size

    model_path = './wall/model/' + name
    checkpoint_path = model_path + '/checkpoints'

    if verbose:
        for k, v in vars(args).items():
            print('%s = "%s"' % (k, str(v)))

    if not os.path.exists(checkpoint_path):
        raise ValueError(checkpoint_path + ' does not exist')

    # Dataset & Model

    dataset = get_wall_dataset(filename, 'proximity', 'ooi_rel_pose',
                               split=split, augment=False, device=device)

    model = NN(in_channels=7, out_channels=7).to(device)
    model.load_state_dict(
        torch.load(checkpoint_path + '/last.pth',
                   map_location=device))

    # Testing

    print(split + ': L=%.4f R2d=%.4f Ang=%.4f' %
          test(model, dataset, batch_size, lmbd=0, verbose=verbose))