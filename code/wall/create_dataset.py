import sys
sys.path.append('.')

import os
import h5py
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.spatial.transform import Rotation as R
from odometry_noise import parse_thymio_odometry, m2v


def se2tose3(v):
    length = v.shape[0]
    xyz = np.concatenate([v[:, :-1], np.zeros((length, 1))], -1)
    rot = R.from_euler('xyz', np.concatenate(
        [np.zeros((length, 2)), v[:, -1:]], -1)).as_quat()
    return np.concatenate([xyz, rot], -1)


def preprocess(path):
    '''Extracts data from HDF5, synchronizes it and saves everything into another HDF5 file.'''
    files = [file[:-3]for file in os.listdir(path)
             if file[-3:] == '.h5' and 'record' in file]

    if not files:
        print('no files found in "' + path + '"')
        return

    h5f = h5py.File(path + 'dataset_' +
                    str(datetime.now()) + '.h5', 'w')

    cumlen = [0]

    # for each file
    max_wid = 0
    for file in sorted(files):
        filename = path + file + '.h5'

        print('found ' + filename)

        with h5py.File(filename, 'r') as input_h5f:
            proximity = input_h5f['proximity'][...]
            # mask = proximity.sum(axis=-1) > 0
            mask = np.full((len(proximity),), True)
            # note: mask = np.full((len(proximity),), True)
            proximity = proximity[mask]
            odometry = input_h5f['odometry'][mask, ...]
            pose = input_h5f['pose'][mask, ...]
            time = input_h5f['time'][mask, ...][:, 0].astype(np.int64)

        wheel_odometry = odometry.copy()

        odometry = parse_thymio_odometry(odometry)
        odometry = np.array([m2v(o) for o in odometry])
        ooi_rel_pose = se2tose3(pose)
        ooi_rel_pose_odom = se2tose3(odometry)
        world_id = np.array([max_wid] * len(pose))

        df = pd.DataFrame({
            'time': time.tolist(),
            'world_id': world_id.tolist(),
            'proximity': (proximity.astype(np.float32) / 4096.0).tolist(),
            'ooi_rel_pose': ooi_rel_pose.tolist(),
            'ooi_rel_pose_odom': ooi_rel_pose_odom.tolist()
        })

        for idx in range(50):
            noisy_odom = parse_thymio_odometry(wheel_odometry, noise=0.35)
            noisy_odom = np.array([m2v(p) for p in noisy_odom])
            df['ooi_noisy_' + str(idx)] = se2tose3(noisy_odom).tolist()

        cumlen.append(len(df))

        df.set_index('time', inplace=True)
        df.index = pd.to_datetime(df.index)

        randomize = False

        if randomize:
            length = len(df)
            random_indices = np.random.permutation(length)
            df = df.iloc[random_indices]

        max_wid += 1

        columns = df.columns.tolist()

        print('saving...')

        for wid, world_df in df.groupby('world_id'):
            length = len(world_df)
            wid = str(int(wid)).zfill(3)
            group = h5f.create_group('world_' + wid)
            group.attrs['has_dect'] = np.array(
                world_df['ooi_rel_pose'].iloc[0]).ndim > 0
            for col in columns:
                shape = np.array(world_df[col].iloc[0]).shape
                store = group.create_dataset(col,
                                             shape=(length,) + shape,
                                             maxshape=(None,) + shape,
                                             dtype=np.float32,
                                             chunks=(128,) + shape,
                                             data=None)
                store[:] = np.stack(world_df[col].values)

    h5f.close()

    print(np.cumsum(cumlen))


if __name__ == '__main__':
    path = '../data/wall/'

    preprocess(path)
