import os
import h5py
import torch
import numpy as np
import albumentations as A
from itertools import repeat
from torchvision.transforms import Compose
from utils import to_tensor, apply_albumentations, reorder_dims


class HDF5Dataset(torch.utils.data.Dataset):

    def __init__(self, filename, start, end, input_col, target_col, transform=None):
        self.transform = transform

        if not os.path.exists(filename):
            raise FileNotFoundError(filename)

        self.h5f = h5py.File(filename, 'r', libver='latest')

        self.X = self.h5f[input_col]
        self.Y = self.h5f[target_col]

        self.start = start
        self.end = end

    def __len__(self):
        return self.end - self.start

    def _process_index(self, index, default):
        if index is None:
            index = default
        elif index < 0:
            index += self.end
        else:
            index += self.start

        return index

    def _process_slice(self, start=None, stop=None, step=None):
        return slice(self._process_index(start, self.start),
                     self._process_index(stop, self.end),
                     step)

    def __getitem__(self, slice):
        if isinstance(slice, int):
            slice = self._process_index(slice, self.start)
        else:
            slice = self._process_slice(slice.start, slice.stop, slice.step)

        xy = (self.X[slice], self.Y[slice])

        if self.transform is not None:
            xy = self.transform(xy)

        return xy

    def __del__(self):
        self.h5f.close()

    def batches(self, batch_size, shuffle=False):
        length = len(self)
        indices = torch.arange(0, length, batch_size)

        if shuffle:
            indices = indices[torch.randperm(indices.size(0))]

        for start in indices:
            end = min(start + batch_size, length)
            yield self[start:end]

    def check_finite(self, batch_size=32):
        for x, y in self.batches(batch_size):
            if not torch.isfinite(x).all():
                print('X is not finite')
            if not torch.isfinite(y).all():
                print('Y is not finite')


# Panda


def get_panda_dataset(filename, input_col, target_col, split='train',
                      only_visible=False, augment=False, device='cpu'):
    """Returns a HDF5Dataset of (input, target) instances.

    Args:
            filename: a filename of an HDF5 file.
            input_col: name of a dataset inside the specified file, to be used as input.
            target_col: name of a dataset inside the specified file, to be used as target.
            split: one of ("train", "validation", "test" or "all").
            only_visible: if set returns only frames when the object is visible (requires *visible*.h5).
            augment: if set applies data augmentation.
            device: device on which to perform computations (usually "cuda" or "cpu").

    Returns:
            the HDF5Dataset of (input, target) instances.
    """
    transforms = []

    if augment:
        transforms.append(lambda xy: apply_albumentations(xy, A.Compose([
            A.MotionBlur(3, p=.5),
            A.GaussNoise(.001, p=.5),
            A.RandomBrightnessContrast(.05, .05, p=.5),
            A.RandomResizedCrop(120, 160, [.95, .95], p=1),
            A.ToFloat()
        ])))

    transforms += [
        reorder_dims,
        lambda xy: to_tensor(xy, device=device)
    ]

    return PandaDataset(filename, split, only_visible, input_col, target_col, Compose(transforms))


class PandaDataset(HDF5Dataset):

    def __init__(self, filename, split, only_visible, input_col, target_col, transform=None):
        self.split = split

        if split not in ['train', 'validation', 'test', 'all']:
            raise ValueError(
                self.split + ' is not a valid split, use one of train, validation, test, or all')

        config = {
            True: {
                'train': (0, 58102),
                'validation': (58102, 68362),
                'test': (68362, 78025),
                'all': (0, 78025)
            },
            False: {
                'train': (0, 180518),
                'validation': (180518, 207142),
                'test': (207142, 236981),
                'all': (0, 236981)
            },
        }

        '''
        [   0   259   766   959  1684  2217  2617  3049  3732  3976  4353  4627
         5404  5851  6429  6758  7167  7750  8052  8640  8996  9457  9788 10567
        10762 11095 11485 11999 12470 13280 13727 14532 14990 15645 16218 16910
        17259 17729 18242 18989 19358 19731 20641 20822 21485 21744 22063 22411
        22690 23200 23323 23721 24333 24783 24961 25533 26170 26655 27301 27647
        28190 28922 29598 30272 30628 31346 32035 32778 33511 33897 34305 34877
        35234 35822 36070 36423 37039 37942 38266 38828 39246 39781 40159 40248
        40809 41030 41445 41951 42359 42863 43099 43675 43861 44418 44858 45329
        45789 46164 46750 47485 48053 48584 48791 49345 50154 50552 50978 51338
        51803 52113 52998 53335 53580 53972 55157 55910 56522 57316 57625 58102
        59006 59547 60100 60839 61200 61796 62448 63030 63344 64065 64352 64768
        65451 65866 66630 67205 67625 68362 68596 69043 69783 70604 70684 71451
        71837 71989 72749 72987 73430 73895 74268 74700 75122 75728 76345 77112
        77622 78025]
        '''

        start, end = config[only_visible][split]

        super(PandaDataset, self).__init__(filename, start, end,
                                           input_col, target_col, transform)


# Wall


def get_wall_dataset(filename, input_col, target_col, split='train',
                     augment=False, device='cpu'):
    """Returns a HDF5Dataset of (input, target) instances.

    Args:
            filename: a filename of an HDF5 file.
            input_col: name of a dataset inside the specified file, to be used as input.
            target_col: name of a dataset inside the specified file, to be used as target.
            split: one of ("train", "validation", "test" or "all").
            augment: if set applies data augmentation.
            device: device on which to perform computations (usually "cuda" or "cpu").

    Returns:
            the HDF5Dataset of (input, target) instances.
    """
    transforms = []

    if augment:
        transforms.append(lambda a: a)

    transforms += [
        lambda xyp: to_tensor(xyp, device=device)
    ]

    return WallDataset(filename, split, input_col, target_col, Compose(transforms))


class WallDataset(torch.utils.data.Dataset):

    def __init__(self, filename, split, input_col, target_col, transform=None):
        self.split = split
        self.input_col = input_col
        self.transform = transform
        self.target_col = target_col

        if split not in ['train', 'validation', 'test', 'all']:
            raise ValueError(
                self.split + ' is not a valid split, use one of train, validation, test, or all')

        if not os.path.exists(filename):
            raise FileNotFoundError(filename)

        self.h5f = h5py.File(filename, 'r', libver='latest')

        config = {
            'train': (0, 12),
            'validation': (12, 16),
            'test': (12, 16),
            'all': (0, 16)
        }

        start_idx, end_idx = config[split]

        self.Ws = [self.h5f['world_' + str(i).zfill(3)]
                   for i in range(start_idx, end_idx)]
        cum_lengths = [0]
        self.lengths = []
        for W in self.Ws:
            self.lengths.append(W[input_col].shape[0])

        cum_lengths += self.lengths
        self.cum_lengths = np.cumsum(cum_lengths)
        self.cum_length = self.cum_lengths[-1]

    def __len__(self):
        return self.cum_length

    def __del__(self):
        self.h5f.close()

    def __getitem__(self, args):
        wid, cid, slice = args
        x = self.Ws[wid][self.input_col][slice]

        if 'noisy' in self.target_col:
            Ys = [self.Ws[wid][self.target_col +
                               '_' + str(c)][slice] for c in cid]
            y = np.concatenate(Ys, 0)
        else:
            y = self.Ws[wid][self.target_col][slice]

        p = y.copy()
        y[:, [1, 2]] = 0

        xyp = (x, y, p)

        if self.transform is not None:
            xyp = self.transform(xyp)

        return xyp

    def batches(self, batch_size, shuffle=False):
        indices = [list(zip(repeat(i), torch.arange(0, self.lengths[i], batch_size)))
                   for i in range(len(self.lengths))]
        indices = torch.tensor(sum(indices, []))

        if shuffle:
            indices = indices[torch.randperm(indices.size(0))]

        for wid, start in indices:
            cid = torch.arange(50).numpy()
            end = min(start + batch_size, self.lengths[wid])
            yield self[wid, cid, start:end]


# Station


def get_station_dataset(filename, input_col, target_col, split='train',
                        augment=False, device='cpu'):
    """Returns a HDF5Dataset of (input, target) instances.

    Args:
            filename: a filename of an HDF5 file.
            input_col: name of a dataset inside the specified file, to be used as input.
            target_col: name of a dataset inside the specified file, to be used as target.
            split: one of ("train", "validation", "test" or "all").
            augment: if set applies data augmentation.
            device: device on which to perform computations (usually "cuda" or "cpu").

    Returns:
            the HDF5Dataset of (input, target) instances.
    """
    transforms = []

    if augment:
        transforms.append(lambda xy: apply_albumentations(xy, A.Compose([
            A.MotionBlur(3, p=.5),
            A.GaussNoise(.001, p=.5),
            A.RandomBrightnessContrast(.05, .05, p=.5),
            A.RandomResizedCrop(120, 160, [.99, .99], p=1),
            A.ToFloat()
        ])))

    transforms += [
        reorder_dims,
        lambda xy: to_tensor(xy, device=device)
    ]

    return StationDataset(filename, split, input_col, target_col, Compose(transforms))


class StationDataset(torch.utils.data.Dataset):

    def __init__(self, filename, split, input_col, target_col, transform=None):
        self.split = split
        self.input_col = input_col
        self.transform = transform
        self.target_col = target_col

        if split not in ['train', 'validation', 'test', 'all']:
            raise ValueError(
                self.split + ' is not a valid split, use one of train, validation, test, or all')

        if not os.path.exists(filename):
            raise FileNotFoundError(filename)

        self.h5f = h5py.File(filename, 'r', libver='latest')

        config = {
            'train': (0, 11),
            'validation': (11, 16),
            'test': (16, 21),
            'all': (0, 21)
        }

        '''
        [
            (0, 0), (1, 4569), (2, 8995), (3, 13644), (4, 15889), (5, 17803),
            (6, 21886), (7, 26129), (8, 30325), (9, 34624), (10, 37384), (11, 39731),
            (12, 44209), (13, 47065), (14, 49368), (15, 51688),(16, 54440),
            (17, 56471), (18, 58881), (19, 63174), (20, 66047), (21, 70222)
        ]
        '''

        start_idx, end_idx = config[split]

        self.Ws = [self.h5f['world_' + str(i).zfill(3)]
                   for i in range(start_idx, end_idx)]
        cum_lengths = [0]
        self.lengths = []
        for W in self.Ws:
            self.lengths.append(W[input_col].shape[0])

        cum_lengths += self.lengths
        self.cum_lengths = np.cumsum(cum_lengths)
        self.cum_length = self.cum_lengths[-1]

    def __len__(self):
        return self.cum_length

    def __del__(self):
        self.h5f.close()

    def __getitem__(self, args):
        wid, cid, slice = args
        X = self.Ws[wid][self.input_col]
        if 'noisy' in self.target_col:
            Y = self.Ws[wid][self.target_col + '_' + str(cid)]
        else:
            Y = self.Ws[wid][self.target_col]
        xy = (X[slice], Y[slice])

        if self.transform is not None:
            xy = self.transform(xy)

        return xy

    def batches(self, batch_size, shuffle=False):
        indices = [list(zip(repeat(i), torch.arange(0, self.lengths[i], batch_size)))
                   for i in range(len(self.lengths))]
        indices = torch.tensor(sum(indices, []))

        if shuffle:
            indices = indices[torch.randperm(indices.size(0))]

        for wid, start in indices:
            cid = torch.randint(0, 10, size=(1,)).item()
            end = min(start + batch_size, self.lengths[wid])
            yield self[wid, cid, start:end]
