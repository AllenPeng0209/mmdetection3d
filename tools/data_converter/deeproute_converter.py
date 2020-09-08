""
import numpy as np
import pickle
from mmcv import track_iter_progress
from pathlib import Path

from IPython import embed
from mmdet3d.core.bbox import box_np_ops
from .deeproute_data_utils import get_deeproute_image_info


def convert_to_deeproute_info_version1(info):
    """convert deeproute info v1 

    Args:
        info (dict): Info of the input kitti data.
            - image (dict): image info
            - calib (dict): calibration info
            - point_cloud (dict): point cloud info
    """
    if 'image' not in info or 'calib' not in info or 'point_cloud' not in info:
        info['image'] = {
            'image_shape': info['img_shape'],
            'image_idx': info['image_idx'],
            'image_path': info['img_path'],
        }
        info['calib'] = {
            'R0_rect': info['calib/R0_rect'],
            'Tr_velo_to_cam': info['calib/Tr_velo_to_cam'],
            'P2': info['calib/P2'],
        }
        info['point_cloud'] = {
            'velodyne_path': info['velodyne_path'],
        }


def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]

def create_deeproute_info_file(data_path,
                           pkl_prefix='deeproute_',
                           save_path=None,
                           relative_path=True):
    """Create info file of deeproute dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        data_path (str): Path of the data root.
        pkl_prefix (str): Prefix of the info file to be generated.
        save_path (str): Path to save the info file.
        relative_path (bool): Whether to use relative path.
    """
    #imageset_folder = Path(data_path) / 'ImageSets'
    #train_img_ids = _read_imageset_file(str(imageset_folder / 'train.txt'))
    #val_img_ids = _read_imageset_file(str(imageset_folder / 'val.txt'))
    #test_img_ids = _read_imageset_file(str(imageset_folder / 'test.txt'))

    print('Generate info. this may take several minutes.')
    if save_path is None:
        save_path = Path(data_path)
    else:
        save_path = Path(save_path)
    deeproute_infos_train = get_deeproute_image_info(
        data_path,
        training=True,
        velodyne=True,
        calib=False,
        relative_path=relative_path,
        valid=False)
    #_calculate_num_points_in_gt(data_path, deeproute_infos_train, relative_path)
    filename = save_path / f'{pkl_prefix}_infos_train.pkl'
    print(f'Deeproute info train file is saved to {filename}')
    with open(filename, 'wb') as f:
        pickle.dump(deeproute_infos_train, f)
    deeproute_infos_val = get_deeproute_image_info(
        data_path,
        training=True,
        velodyne=True,
        calib=False,
        relative_path=relative_path,
        valid = True)
    #_calculate_num_points_in_gt(data_path, deeproute_infos_val, relative_path)
    filename = save_path / f'{pkl_prefix}_infos_val.pkl'
    print(f'Deeproute info val file is saved to {filename}')
    with open(filename, 'wb') as f:
        pickle.dump(deeproute_infos_val, f)
    filename = save_path / f'{pkl_prefix}_infos_trainval.pkl'
    print(f'Deeproute info trainval file is saved to {filename}')
    with open(filename, 'wb') as f:
        pickle.dump(deeproute_infos_train + deeproute_infos_val, f)

    deeproute_infos_test = get_deeproute_image_info(
        data_path,
        training=False,
        label_info=True,
        velodyne=True,
        calib=False,
        relative_path=relative_path,
        valid=False)
    filename = save_path / f'{pkl_prefix}_infos_test.pkl'
    print(f'Deeproute info test file is saved to {filename}')
    with open(filename, 'wb') as f:
        pickle.dump(deeproute_infos_test, f)
