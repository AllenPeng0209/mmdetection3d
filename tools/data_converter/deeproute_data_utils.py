""
import numpy as np
from collections import OrderedDict
from concurrent import futures as futures
from pathlib import Path
from skimage import io
from IPython import embed
import os
import json

def get_deeproute_info_path(idx,
                        prefix,
                        info_type='image',
                        file_tail='.png',
                        training=True,
                        relative_path=True,
                        exist_check=True):
    folder = idx[0]
    img_idx_str = idx[1]
    img_idx_str += file_tail
    prefix = Path(prefix)
    if training:
        file_path ='/home/yanlun/mmdetection3d/data/deeproute_mini'/ Path('training') / info_type / folder/ img_idx_str
    else:
        file_path ='/home/yanlun/mmdetection3d/data/deeproute_mini'/ Path('testing') / info_type / folder / img_idx_str
    if exist_check and not (prefix / file_path).exists():
        raise ValueError('file not exist: {}'.format(file_path))
    if relative_path:
        return str(file_path)
    else:
        return str(prefix / file_path)


def get_image_path(idx,
                   prefix,
                   training=True,
                   relative_path=True,
                   exist_check=True):
    return get_deeproute_info_path(idx, prefix, 'image', '.png', training,
                               relative_path, exist_check)


def get_label_path(idx,
                   prefix,
                   training=True,
                   relative_path=True,
                   exist_check=True):
    return get_deeproute_info_path(idx, prefix, 'label', '.txt', training,
                               relative_path, exist_check)


def get_velodyne_path(idx,
                      prefix,
                      training=True,
                      relative_path=True,
                      exist_check=True):
    return get_deeproute_info_path(idx, prefix, 'bin', '.bin', training,
                               relative_path, exist_check)


def get_calib_path(idx,
                   prefix,
                   training=True,
                   relative_path=True,
                   exist_check=True):
    return get_deeproute_info_path(idx, prefix, 'calib', '.txt', training,
                               relative_path, exist_check)


def get_label_anno(label_path):
    annotations = {}
    annotations.update({
        'id':[],
        'type': [],
        'state': [],
        'bbox2d': [],
	'location':[],
        'dimensions': [],
        'rotation_y': []
    })
    with open(label_path, 'r') as f:
         
         objects = json.load(f)
    # if len(lines) == 0 or len(lines[0]) < 15:
    #     content = []
    # else:
    ids, types, state, bbox2d, location, dimensions, rotation_y = [],[],[],[],[],[],[]
    
    num_objects = len(objects["objects"])
    for obj in objects["objects"]:
        ids.append(obj['id'])
        types.append(obj['type'])
        state.append(obj['state'])
        bbox2d.append(obj['bounding_box2d'])
        location.append(obj['position'])
        dimensions.append(obj['bounding_box'])
        rotation_y.append(obj['heading'])
        
    annotations['id'] = np.array(ids)
    annotations['type'] = np.array(types)
    annotations['state'] = np.array(state)
    annotations['bbox2d'] = np.array(bbox2d)
    annotations['location'] = np.array(location)
    annotations['dimensions'] = np.array(dimensions)
    annotations['rotation_y'] = np.array(rotation_y)
    return annotations


def _extend_matrix(mat):
    mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
    return mat


def get_deeproute_image_info(path,
                         training=True,
                         label_info=True,
                         velodyne=False,
                         calib=False,
                         image_ids=67166,
                         extend_matrix=True,
                         num_worker=8,
                         relative_path=True,
                         with_imageshape=False, 
                         valid = False):
    # image_infos = []
    """
    KITTI annotation format version 2:
    {
        [optional]points: [N, 3+] point cloud
        [optional, for kitti]image: {
            image_idx: ...
            image_path: ...
            image_shape: ...
        }
        point_cloud: {
            num_features: 4
            velodyne_path: ...
        }
        [optional, for kitti]calib: {
            R0_rect: ...
            Tr_velo_to_cam: ...
            P2: ...
        }
        annos: {
            location: [num_gt, 3] array
            dimensions: [num_gt, 3] array
            rotation_y: [num_gt] angle array
            name: [num_gt] ground truth name array
            [optional]difficulty: kitti difficulty
            [optional]group_ids: used for multi-part object
        }
    }
    """
    root_path = Path(path)
    image_ids = []
    if training:
        if valid:
            folders= os.listdir(root_path/'training'/'pcd')
            for folder in folders[:2]: 
                idx_in_folder = os.listdir(root_path/'training'/'label'/folder)
                for idx in idx_in_folder:
                    image_ids.append([folder, idx[:-4]]) 
        else:
            folders= os.listdir(root_path/'training'/'pcd') 
            for folder in folders:
                idx_in_folder = os.listdir(root_path/'training'/'label'/folder)
                for idx in idx_in_folder:
                    image_ids.append([folder, idx[:-4]])
    else:
            folders= os.listdir(root_path/'testing'/'pcd') 
            for folder in folders:
                folders= os.listdir(root_path/'testing'/'pcd')
                idx_in_folder = os.listdir(root_path/'testing'/'label'/folder) 
                for idx in idx_in_folder: 
                    image_ids.append([folder, idx[:-4]]) 
    

    def map_func(idx):
        info = {}
        pc_info = {'num_features': 4}
        calib_info = {}

        image_info = {'image_idx': idx}
        annotations = None
        if velodyne:
            pc_info['velodyne_path'] = get_velodyne_path(
                idx, path, training, relative_path)
        #image_info['image_path'] = get_image_path(idx, path, training,                                          relative_path)
        if with_imageshape:
            img_path = image_info['image_path']
            if relative_path:
                img_path = str(root_path / img_path)
            image_info['image_shape'] = np.array(
                io.imread(img_path).shape[:2], dtype=np.int32)
        if label_info:
            label_path = get_label_path(idx, path, training, relative_path)
            if relative_path:
                label_path = str(root_path / label_path)
            annotations = get_label_anno(label_path)
        info['image'] = image_info
        info['point_cloud'] = pc_info
        if calib:
            calib_path = get_calib_path(
                idx, path, training, relative_path=False)
            with open(calib_path, 'r') as f:
                lines = f.readlines()
            P0 = np.array([float(info) for info in lines[0].split(' ')[1:13]
                           ]).reshape([3, 4])
            P1 = np.array([float(info) for info in lines[1].split(' ')[1:13]
                           ]).reshape([3, 4])
            P2 = np.array([float(info) for info in lines[2].split(' ')[1:13]
                           ]).reshape([3, 4])
            P3 = np.array([float(info) for info in lines[3].split(' ')[1:13]
                           ]).reshape([3, 4])
            if extend_matrix:
                P0 = _extend_matrix(P0)
                P1 = _extend_matrix(P1)
                P2 = _extend_matrix(P2)
                P3 = _extend_matrix(P3)
            R0_rect = np.array([
                float(info) for info in lines[4].split(' ')[1:10]
            ]).reshape([3, 3])
            if extend_matrix:
                rect_4x4 = np.zeros([4, 4], dtype=R0_rect.dtype)
                rect_4x4[3, 3] = 1.
                rect_4x4[:3, :3] = R0_rect
            else:
                rect_4x4 = R0_rect

            Tr_velo_to_cam = np.array([
                float(info) for info in lines[5].split(' ')[1:13]
            ]).reshape([3, 4])
            Tr_imu_to_velo = np.array([
                float(info) for info in lines[6].split(' ')[1:13]
            ]).reshape([3, 4])
            if extend_matrix:
                Tr_velo_to_cam = _extend_matrix(Tr_velo_to_cam)
                Tr_imu_to_velo = _extend_matrix(Tr_imu_to_velo)
            calib_info['P0'] = P0
            calib_info['P1'] = P1
            calib_info['P2'] = P2
            calib_info['P3'] = P3
            calib_info['R0_rect'] = rect_4x4
            calib_info['Tr_velo_to_cam'] = Tr_velo_to_cam
            calib_info['Tr_imu_to_velo'] = Tr_imu_to_velo
            info['calib'] = calib_info

        if annotations is not None:
            info['annos'] = annotations
        return info

    with futures.ThreadPoolExecutor(num_worker) as executor:
        image_infos = executor.map(map_func, image_ids)


    return list(image_infos)


def kitti_anno_to_label_file(annos, folder):
    folder = Path(folder)
    for anno in annos:
        image_idx = anno['metadata']['image_idx']
        label_lines = []
        for j in range(anno['bbox'].shape[0]):
            label_dict = {
                'name': anno['name'][j],
                'alpha': anno['alpha'][j],
                'bbox': anno['bbox'][j],
                'location': anno['location'][j],
                'dimensions': anno['dimensions'][j],
                'rotation_y': anno['rotation_y'][j],
                'score': anno['score'][j],
            }
            label_line = kitti_result_line(label_dict)
            label_lines.append(label_line)
        label_file = folder / f'{get_image_index_str(image_idx)}.txt'
        label_str = '\n'.join(label_lines)
        with open(label_file, 'w') as f:
            f.write(label_str)




def kitti_result_line(result_dict, precision=4):
    prec_float = '{' + ':.{}f'.format(precision) + '}'
    res_line = []
    all_field_default = OrderedDict([
        ('name', None),
        ('truncated', -1),
        ('occluded', -1),
        ('alpha', -10),
        ('bbox', None),
        ('dimensions', [-1, -1, -1]),
        ('location', [-1000, -1000, -1000]),
        ('rotation_y', -10),
        ('score', 0.0),
    ])
    res_dict = [(key, None) for key, val in all_field_default.items()]
    res_dict = OrderedDict(res_dict)
    for key, val in result_dict.items():
        if all_field_default[key] is None and val is None:
            raise ValueError('you must specify a value for {}'.format(key))
        res_dict[key] = val

    for key, val in res_dict.items():
        if key == 'name':
            res_line.append(val)
        elif key in ['truncated', 'alpha', 'rotation_y', 'score']:
            if val is None:
                res_line.append(str(all_field_default[key]))
            else:
                res_line.append(prec_float.format(val))
        elif key == 'occluded':
            if val is None:
                res_line.append(str(all_field_default[key]))
            else:
                res_line.append('{}'.format(val))
        elif key in ['bbox', 'dimensions', 'location']:
            if val is None:
                res_line += [str(v) for v in all_field_default[key]]
            else:
                res_line += [prec_float.format(v) for v in val]
        else:
            raise ValueError('unknown key. supported key:{}'.format(
                res_dict.keys()))
    return ' '.join(res_line)
