import sys
sys.path.append('.')

import os
import h5py
import numpy as np
import pandas as pd
from datetime import datetime
from odometry_noise import v2m, m2v, Odometry, relrotnoise

path = '../data/station/benchmark/'

files = [file[:-7] for file in os.listdir(path) if file[-7:] == '.pickle']

if not files:
    print('no bag files found in "' + path + '"')
    exit(0)

h5f = h5py.File(path + 'dataset_' +
                str(datetime.now()) + '.h5', 'w')

# for each bagfile
max_wid = 0
for file in sorted(files):
    filename = path + file + '.pickle'

    print('found ' + filename)

    df = pd.read_pickle(filename)
    df.rename(columns={'ooi_pose': 'ooi_rel_pose'}, inplace=True)
    df['world_id'] = max_wid
    ooi_rel_pose_mat = df['ooi_rel_pose'].apply(v2m)
    noisy_odom = Odometry(ooi_rel_pose_mat,
                          lambda dp: relrotnoise(dp, fraction=0.1))

    for idx in range(10):
        odom = noisy_odom.sample_all()
        df['ooi_noisy_' + str(idx)] = [m2v(p) for p in odom]

    if False:
        length = len(df)
        indices = np.arange((length // 2) * 2).reshape(-1, 2)
        random_indices = indices[np.random.permutation(
            length // 2)].flatten()

        if length % 2 == 1:
            random_indices = np.append(random_indices, length - 1)

        df = df.iloc[random_indices]
        print(random_indices)

    max_wid += 1

    print('saving...')

    columns = df.columns.tolist()

    df.dropna(inplace=True)

    for wid, world_df in df.groupby('world_id'):
        length = len(world_df)
        wid = str(int(wid)).zfill(3)
        group = h5f.create_group('world_' + wid)
        group.attrs['has_dect'] = True
        group.attrs['filename'] = file
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
