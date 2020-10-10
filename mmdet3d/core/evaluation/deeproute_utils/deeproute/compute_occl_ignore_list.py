import sys
sys.path.append("/home/yanlun/mmdetection3d/mmdet3d/core/evaluation/deeproute_utils/")
import numpy as np
import time
import os
from collections import defaultdict
from google.protobuf import text_format
from tqdm import tqdm
import multiprocessing
import pickle
import core.box_ops as box_ops
from core.occl_dist_assign import get_occlusion
from core.read_info import get_all_txt_names, readInfo, readInfo_per_frame

def sub_compute_occl_ignore(gt_path, info_path, pcd_path, idx, occlusion_thresh):
    timeStamp = info_text_names[idx]
    # compute occlusion attribute of gt
    gt_loc, gt_dims, gt_yaws, gt_ids, gt_names = readInfo_per_frame(gt_path, timeStamp, name_to_label, interest_labels, False)

    gt_boxes = np.concatenate([gt_loc, gt_dims, gt_yaws[..., np.newaxis]], axis=1)
    box_ops.change_box3d_center_(gt_boxes, src=[0.5, 0.5, 0], dst=[0.5, 0.5, 0.5])

    gt_corners = box_ops.get_corners(gt_boxes[:, :2], gt_boxes[:, 3:5], gt_boxes[:, 6]) if gt_boxes.shape[0] > 0 else np.array([])

    if info_path == None:
        info_path = gt_path
        info_corners = gt_corners.copy()
    else:
        info_loc, info_dims, info_yaws, info_ids, info_names = readInfo_per_frame(info_path, timeStamp, name_to_label, interest_labels, False)

        info_boxes = np.concatenate([info_loc, info_dims, info_yaws[..., np.newaxis]], axis=1)
        box_ops.change_box3d_center_(info_boxes, src=[0.5, 0.5, 0], dst=[0.5, 0.5, 0.5])
        info_corners = box_ops.get_corners(info_boxes[:, :2], info_boxes[:, 3:5], info_boxes[:, 6]) if info_boxes.shape[0] > 0 else np.array([])

    info_occlusion = get_occlusion(info_corners, gt_corners, occlusion_thresh)

    # compute ignore attribute of gt
    info_loc, info_ids, info_names, info_surfaces = readInfo(info_path, timeStamp, name_to_label, interest_labels, False)
    points = box_ops.get_points_per_box(pcd_path, timeStamp, info_surfaces)

    info_ignore = []
    for idx in range(len(points)):
        number = name_to_label[info_names[idx].lower()]
        if points[idx].shape[0] >= filter_points_number[number]:
            info_ignore.append(False)
        else:
            info_ignore.append(True)

    return (timeStamp, info_occlusion, info_ignore, info_ids)

def compute_occl_ignore(info_path, pcd_path, save_path, config_path, pool_num = 10, detection_occl = True, save_flag = True, tracker_path = None):
    global info_text_names
    global filter_points_number
    global interest_labels
    global name_to_label
    global num_features_for_pc

    if detection_occl:
        if isinstance(config_path, str):
            from protos import detection_eval_pb2
            config = detection_eval_pb2.DeeprouteDetectionEvalConfig()
            with open(config_path, "r") as f:
                proto_str = f.read()
                text_format.Merge(proto_str, config)
        else:
            config = config_path
    else:
        if isinstance(config_path, str):
            from protos import tracking_eval_pb2
            config = tracking_eval_pb2.DeeprouteDetectionEvalConfig()
            with open(config_path, "r") as f:
                proto_str = f.read()
                text_format.Merge(proto_str, config)
        else:
            config = config_path

    occlusion_thresh = config.deeproute_eval_input_reader.occlusion_thresh
    interest_labels = config.deeproute_eval_input_reader.interest_labels
    filter_points_number = list(config.deeproute_eval_input_reader.filter_points_number)
    num_features_for_pc = int(config.deeproute_eval_input_reader.num_features_for_pc)
    name_to_label = {}
    for sample_group in config.deeproute_eval_input_reader.name_to_label.sample_groups:
        name_to_label.update(dict(sample_group.name_to_num))

    assert info_path is not None
    assert save_path is not None
    # info_text_names = get_all_txt_names(info_path)

    if detection_occl:
        occl_relation = defaultdict(list)
        gt_ignores = defaultdict(list)
    else:
        occl_relation = defaultdict(dict)
        gt_ignores = defaultdict(dict)

    pool = multiprocessing.Pool(pool_num)
    res = []
    for idx in range(len(info_text_names)):
        res.append(pool.apply_async(sub_compute_occl_ignore, (info_path, tracker_path, pcd_path, idx, occlusion_thresh)))

    for i in tqdm(range(len(res))):
        if detection_occl:
            occl_relation[res[i].get()[0]].extend(res[i].get()[1])
            gt_ignores[res[i].get()[0]].extend(res[i].get()[2])
        else:
            for idx2, id in enumerate(res[i].get()[3]):
                occl_relation[res[i].get()[0]][id] = not res[i].get()[1][idx2]
                gt_ignores[res[i].get()[0]][id] = res[i].get()[2][idx2]

    pool.close()
    pool.join()

    if save_flag and detection_occl:
        with open(os.path.join(save_path, 'occlusion_detection.pkl'), 'wb') as f:
            pickle.dump((occlusion_thresh, occl_relation), f)
        with open(os.path.join(save_path, 'ignore_detection.pkl'), 'wb') as f:
            pickle.dump((filter_points_number, gt_ignores), f)
    elif save_flag and (not detection_occl):
        with open(os.path.join(save_path, 'occlusion_tracking.pkl'), 'wb') as f:
            pickle.dump((occlusion_thresh, occl_relation), f)
        with open(os.path.join(save_path, 'ignore_tracking.pkl'), 'wb') as f:
            pickle.dump((filter_points_number, gt_ignores), f)

    return occl_relation, gt_ignores


def main():
    list_path = "/home/yanlun/mmdetection3d/data/deeproute_mini/test.list.20191216"
    #list_path = "/home/data/deeproute/20200420_long_cone_test.list"
    info_path = "/home/yanlun/mmdetection3d/data/deeproute_mini/testing/label"
    pcd_path = "/home/yanlun/mmdetection3d/data/deeproute_mini/testing/pcd"
    save_path = "/home/yanlun/mmdetection3d/data/deeproute_mini/testing/evaluation/occl"
    config_path = "/home/yanlun/mmdetection3d/mmdet3d/core/evaluation/deeproute_utils/configs/detection.eval.config"
    #config_path = "/home/yanlun/mmdetection3d/mmdet3d/core/evaluation/deeproute_utils/configs/detection.long.eval.config"

    if len(sys.argv) > 1:
        pool_num = int(sys.argv[1])
    else:
        pool_num = 10

    global info_text_names
    with open(list_path) as hd:
        info_text_names = [line.strip() + ".txt" for line in hd.readlines()]
    print("Have {} files".format(len(info_text_names)))

    t = time.time()
    compute_occl_ignore(info_path, pcd_path, save_path, config_path, pool_num)
    print("The total evaluation time: {}".format(time.time()-t))
    print("occlusion relation of gt is placed in: {}".format(save_path))

if __name__ == '__main__':
    main()
