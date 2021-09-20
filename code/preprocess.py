import os
import tqdm
import h5py
import rosbag
import numpy as np
import pandas as pd
from viz_utils import project
from datetime import datetime
from utils import jpeg_to_np, transf_matrix, se3_pose


def get_pose(m, col):
    p = m.position
    q = m.orientation
    q = np.array([q.x, q.y, q.z, q.w])
    q /= np.linalg.norm(q)
    return {col: np.array([p.x, p.y, p.z] + q.tolist())}


def get_transf_matrix(row, col):
    pose = row[col]

    if np.isnan(pose).any():
        return np.nan

    return transf_matrix(pose)


def get_transf_wrt(row, mat, wrt):
    if np.isnan(row[mat]).any() or np.isnan(row[wrt]).any():
        return np.nan

    wrt_inv = np.linalg.inv(row[wrt])
    return np.dot(wrt_inv, row[mat])


def get_transf_to_pose(row, mat):
    if np.isnan(row[mat]).any():
        return np.nan

    return se3_pose(row[mat])


def d_quat(q1, q2, normalize=True):
    if normalize:
        q1 = q1 / np.linalg.norm(q1, axis=-1)
        q2 = q2 / np.linalg.norm(q2, axis=-1)

    diff = np.linalg.norm(q1 - q2, axis=-1)
    summ = np.linalg.norm(q1 + q2, axis=-1)
    return np.stack([diff, summ], axis=-1).min(axis=-1)


def pose_avg(pos, axis=0):
    p = pos[:, :3]
    q = pos[:, 3:]
    q /= np.linalg.norm(q, axis=-1, keepdims=True)

    p = np.mean(p, axis=axis)

    diff = np.linalg.norm(q[:, None] - q[None, :], axis=-1)
    summ = np.linalg.norm(q[:, None] + q[None, :], axis=-1)
    dist = np.stack([diff, summ], axis=-1).min(axis=-1).sum(axis=-1)
    q = q[dist.argmin()]

    return np.concatenate([p, q])


def preprocess(path, extractors, tolerance='1s'):
    '''Extracts data from bagfiles and saves everything into an HDF5 file.

    Args:
            path: the path in which bag files are stored.
            extractors: a dictionary of functions associated to ROS topics that extracts the required values,
                                    composed of ROS topics as keys and functions as values.
            tolerance: a string representing the time tolerance used to merge the different ROS topics.
    '''
    files = [file[:-4] for file in os.listdir(path) if file[-4:] == '.bag']

    if not files:
        print('no bag files found in "' + path + '"')
        return

    h5f = h5py.File(path + 'dataset_' +
                    str(datetime.now()) + '.h5', 'w')

    # for each bagfile
    max_wid = 0
    for file in sorted(files):
        filename = path + file + '.bag'

        print('found ' + filename)

        # extract one dataframe per topic
        dfs = bag2dfs(rosbag.Bag(filename), extractors)

        dect_pose_df = mergedfs(
            {k: dfs[k] for k in ['/ooi_detect/pose', '/world/id']},
            tolerance=tolerance)

        gaz_pose_df = mergedfs(
            {k: dfs[k] for k in ['/ooi/pose', '/world/id']},
            tolerance=tolerance)

        del dfs['/ooi/pose']

        dfs['/ooi_detect/pose'].rename(columns={
            'ooi_dect_pose': 'ooi_tag_pose'
        }, inplace=True)

        gaz_pose = gaz_pose_df.groupby('world_id').first()['ooi_gaz_pose']
        mean_dect_pose = dect_pose_df.groupby('world_id')['ooi_dect_pose'].apply(
            lambda x: pose_avg(np.stack(x))) + np.array([0, 0, 0.03, 0, 0, 0, 0])

        # merge the dataframes based on the timeindex
        df = mergedfs(dfs, tolerance=tolerance, ignore=['/ooi_detect/pose'])

        df = df.join(mean_dect_pose, on='world_id')
        df = df.join(gaz_pose, on='world_id')

        # update world_id s.t. different bagfiles are seen as sequential
        wids = np.unique(df['world_id'])
        df['world_id'] += max_wid - wids.min()
        max_wid += np.unique(df['world_id']).shape[0]

        print('computing relative poses...')

        df['ee_transf_mat'] = df.apply(
            get_transf_matrix, axis=1, args=('ee_pose',))
        df['ooi_gaz_transf_mat'] = df.apply(
            get_transf_matrix, axis=1, args=('ooi_gaz_pose',))
        df['ooi_dect_transf_mat'] = df.apply(
            get_transf_matrix, axis=1, args=('ooi_dect_pose',))

        df['ooi_rel_transf_mat'] = df.apply(
            get_transf_wrt, axis=1, args=('ooi_dect_transf_mat', 'ee_transf_mat'))
        df['ooi_gaz_transf_mat'] = df.apply(
            get_transf_wrt, axis=1, args=('ooi_gaz_transf_mat', 'ee_transf_mat'))

        df['ooi_rel_pose'] = df.apply(
            get_transf_to_pose, axis=1, args=('ooi_rel_transf_mat',))

        df['ooi_gaz_pose'] = df.apply(
            get_transf_to_pose, axis=1, args=('ooi_gaz_transf_mat',))

        df['ooi_rel_proj2d_pose'] = df['ooi_rel_transf_mat'].apply(project)
        df['ooi_rel_proj2d_in_fov'] = df['ooi_rel_proj2d_pose'].apply(
            lambda x: 0 <= x[0] and x[0] <= 640 and 0 <= x[1] and x[1] <= 480
        )

        df.dropna(subset=[c for c in df.keys()
                          if c not in ['ooi_tag_pose']], inplace=True)

        print('saving...')

        columns = ['camera', 'ooi_rel_proj2d_in_fov'] + \
            list(filter(lambda x: 'pose' in x, df.columns))

        # fillna column with [-1 -1 -1 -1 -1 -1 -1]
        df['ooi_tag_pose'] = df['ooi_tag_pose'].apply(
            lambda e: e if not np.isnan(e).any() else [-1] * 7)

        df.dropna(inplace=True)

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


def bag2dfs(bag, extractors):
    '''Extracts data from a ROS bagfile and converts it to dataframes (one per topic).

    Args:
            bag: a ROS bagfile.
            extractors: a dictionary of functions associated to ros topics that extracts the required values,
                                    composed of ros topics as keys and functions as values.
    Returns:
            a dictionary of dataframes divided by ROS topic.
    '''
    result = {}

    for topic in tqdm.tqdm(extractors.keys(),
                           desc='extracting data from the bagfile'):
        timestamps = []
        values = []

        for subtopic, msg, t in bag.read_messages(topic):
            if subtopic == topic:
                timestamps.append(t.to_nsec())
                values.append(extractors[topic](msg))

        if not values:
            raise ValueError('Topic "' + topic +
                             '" not found in one of the bagfiles')

        df = pd.DataFrame(data=values, index=timestamps,
                          columns=values[0].keys())

        # note: avoid duplicated timestamps generated by ros
        df = df[~df.index.duplicated()]

        result[topic] = df

    return result


def mergedfs(dfs, tolerance='1s', ignore=[]):
    '''Merges different dataframes indexed by datetime into a synchronized dataframe.

    Args:
            dfs: a dictionary of dataframes.
            tolerance: a string representing the time tolerance used to merge the different dataframes.
            ignore: a list of dfs keys ignored for the selection of the time index.

    Returns:
            a single dataframe composed of the various dataframes synchronized.
    '''
    min_topic = None

    # find topic with fewest datapoints
    for topic, df in dfs.items():
        if topic not in ignore:
            if not min_topic or len(dfs[min_topic]) > len(df):
                min_topic = topic

    ref_df = dfs[min_topic]
    other_dfs = dfs
    other_dfs.pop(min_topic)

    # merge dfs with a time tolerance
    result = pd.concat(
        [ref_df] +
        [df.reindex(index=ref_df.index, method='nearest',
                    tolerance=pd.Timedelta(tolerance).value)
         for _, df in other_dfs.items()],
        axis=1)

    # columns to be ignored during dropna
    ignore_cols = [c for k, df in dfs.items()
                   for c in df.keys() if k in ignore]

    result.dropna(subset=[c for c in result.keys()
                          if c not in ignore_cols], inplace=True)

    result.index = pd.to_datetime(result.index)

    return result


if __name__ == '__main__':
    path = '../data/panda/'

    extractors = {
        '/d435_color/image_raw/compressed': lambda m:
        {'camera': jpeg_to_np(m.data, (160, 120), normalize=True)},
        '/world/id': lambda m: {'world_id': m.data},
        '/ee/pose': lambda m: get_pose(m, 'ee_pose'),
        '/ooi/pose': lambda m: get_pose(m, 'ooi_gaz_pose'),
        '/ooi_detect/pose': lambda m: get_pose(m, 'ooi_dect_pose')
    }

    res = preprocess(path=path, extractors=extractors, tolerance='0.5s')
