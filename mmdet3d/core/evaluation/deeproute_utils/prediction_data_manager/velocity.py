import sys
sys.path.append("..")
import numpy as np, os
from tqdm import tqdm
from collections import defaultdict
import pickle
import time
import mpi4py.MPI as MPI
import open3d as o3d
from core.read_info import get_all_txt_names, readLidarConfigFile, readPoseConfigFile, readInfo
from core.box_ops import get_points_per_box
from core.calib_trans import pointsSensingToMap_v2, pointsMapToSensing_v2
from google.protobuf import text_format
from protos import prediction_eval_pb2

def loadPredictConfig(config_path):
    if isinstance(config_path, str):
        config = prediction_eval_pb2.DeeproutePredictionEvalConfig()
        with open(config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, config)
    else:
        config = config_path

    interest_labels = config.deeproute_eval_input_reader.interest_labels
    weight = config.deeproute_eval_input_reader.weight
    overlap_thresh = config.deeproute_eval_input_reader.overlap_thresh
    frequency = int(config.deeproute_eval_input_reader.frequency)
    eval_frequency = int(config.deeproute_eval_input_reader.eval_frequency)
    predict_time = int(config.deeproute_eval_input_reader.predict_time)
    cal_iou_bev = config.deeproute_eval_input_reader.cal_iou_bev
    num_modal = config.deeproute_eval_input_reader.num_modal
    filter_points_number = config.deeproute_eval_input_reader.filter_points_number

    internal_gap = int(frequency / eval_frequency)
    future_length = predict_time * frequency + 1
    selected = list(range(future_length - 1, 0, -internal_gap))
    selected.sort()

    name_to_label = {}
    for sample_group in config.deeproute_eval_input_reader.name_to_label.sample_groups:
        name_to_label.update(dict(sample_group.name_to_num))

    velocity_accpet_error = float(config.deeproute_eval_input_reader.velocity_accpet_error)
    stop_thresh = float(config.deeproute_eval_input_reader.stop_thresh)
    acclerate_thresh = float(config.deeproute_eval_input_reader.acclerate_thresh)

    return (interest_labels, weight, overlap_thresh, future_length, selected, cal_iou_bev, num_modal, \
            name_to_label, internal_gap, filter_points_number, velocity_accpet_error, stop_thresh, acclerate_thresh, frequency, config)

def sub_compute_velocity(frame_id, pcd_path):

    timeStamp_cur = info_text_names[frame_id]
    pose_info_cur = pose_infos[str(int(timeStamp_cur.split(".")[0]))]
    
    loc_cur, ids_cur, names_cur, surfaces_cur = readInfo(gt_path, timeStamp_cur, name_to_label, interest_labels)
    points_cur = get_points_per_box(pcd_path, timeStamp_cur, surfaces_cur)
    if frame_id == 0:
        velocity_confidence = {}
        for idx_cur, id in enumerate(ids_cur):
            velocity_confidence[id] = (-1, [0, 0, 0], [0, 0, 0])
        return (timeStamp_cur, velocity_confidence)

    timeStamp_pre = info_text_names[frame_id-1]
    pose_info_pre = pose_infos[str(int(info_text_names[frame_id-1].split(".")[0]))]
    loc_pre, ids_pre, names_pre, surfaces_pre = readInfo(gt_path, timeStamp_pre, name_to_label, interest_labels)
    points_pre = get_points_per_box(pcd_path, timeStamp_pre, surfaces_pre)

    velocity_confidence = {}
    for idx_cur, id in enumerate(ids_cur):
        if name_to_label[names_cur[idx_cur].lower()] != 0:
            continue
        if id not in ids_pre:
            velocity_confidence[id] = (-1, [0, 0, 0], [0, 0, 0])
            continue
        idx_pre = ids_pre.index(id)
        if name_to_label[names_pre[idx_pre].lower()] != 0:
            continue

        # compute velocity with respect to box center
        global_pre = pointsSensingToMap_v2(pose_info_pre, loc_pre[idx_pre].reshape(-1, 3), lidar_config)[0]
        global_cur = pointsSensingToMap_v2(pose_info_cur, loc_cur[idx_cur].reshape(-1, 3), lidar_config)[0]

        interval = (int(timeStamp_cur.split(".")[0]) - int(timeStamp_pre.split(".")[0])) / 1000000
        velocity_box_x = (global_cur[0] - global_pre[0]) * (1/interval)
        velocity_box_y = (global_cur[1] - global_pre[1]) * (1/interval)
        velocity_box = [velocity_box_x, velocity_box_y, 0.0]
        #print(velocity_box)
        # compute velocity with respect to pointCloud
        local_point_pre = points_pre[idx_pre]
        local_point_cur = points_cur[idx_cur]
        if local_point_pre.shape[0] < filter_points_number[0] or local_point_cur.shape[0] < filter_points_number[0]:
            velocity_confidence[id] = (-1, velocity_box, [0, 0, 0])
            continue

        global_point_pre = pointsSensingToMap_v2(pose_info_pre, local_point_pre[:,0:3].reshape(-1, 3), lidar_config)
        local_point_pre_under_cur = pointsMapToSensing_v2(pose_info_cur, global_point_pre[:, 0:3].reshape(-1, 3), lidar_config)

        pcd_pre_under_cur = o3d.geometry.PointCloud()
        pcd_pre_under_cur.points = o3d.utility.Vector3dVector(local_point_pre_under_cur[:, 0:3])
        pcd_cur = o3d.geometry.PointCloud()
        pcd_cur.points = o3d.utility.Vector3dVector(local_point_cur[:, 0:3])
        #print("id_cur", id)
        trans_init = np.eye(4)
        ret = o3d.registration.registration_icp(pcd_pre_under_cur, pcd_cur, 1000, trans_init,
                                                    o3d.registration.TransformationEstimationPointToPoint(),
                                                    criteria = o3d.registration.ICPConvergenceCriteria(max_iteration=50000))
        transform = ret.transformation
        local_center_pre_under_cur = np.mean(local_point_pre_under_cur, axis=0).reshape(1, -1)
        local_center_pre_under_cur = np.concatenate([local_center_pre_under_cur, np.array([1]).reshape(-1, 1)], axis = 1)
        local_center_cur = np.dot(local_center_pre_under_cur, transform.T)
        local_center_pre_under_cur = local_center_pre_under_cur[:, 0:3]
        local_center_cur = local_center_cur[:, 0:3]

        global_center_pre = pointsSensingToMap_v2(pose_info_cur, local_center_pre_under_cur, lidar_config)[0]
        global_center_cur = pointsSensingToMap_v2(pose_info_cur, local_center_cur, lidar_config)[0]

        velocity_pointCloud_x = (global_center_cur[0] - global_center_pre[0]) * (1/interval)
        velocity_pointCloud_y = (global_center_cur[1] - global_center_pre[1]) * (1/interval)
        velocity_pointCloud = [velocity_pointCloud_x, velocity_pointCloud_y, 0.0]

        if ret.inlier_rmse < 0.2 and abs(velocity_pointCloud_x - velocity_box_x) < 1 and abs(velocity_pointCloud_y - velocity_box_y) < 1:
            velocity_confidence[id] = (0, velocity_box, velocity_pointCloud)
        elif ret.inlier_rmse < 0.2 and abs(velocity_pointCloud_x - velocity_box_x) < 2 and abs(velocity_pointCloud_y - velocity_box_y) < 2:
            velocity_confidence[id] = (1, velocity_box, velocity_pointCloud)
        else:
            velocity_confidence[id] = (2, velocity_box, velocity_pointCloud)
    return (int(timeStamp_cur.split(".")[0]), velocity_confidence)

def compute_velocity(gt_path, pcd_path, config_path, poseConfigFile, lidarConfigFile, velocity_file):
    global info_text_names
    global pose_infos
    global lidar_config
    global name_to_label
    global filter_points_number
    global interest_labels

    interest_labels, weight, overlap_thresh, future_length, selected, cal_iou_bev, num_modal, \
    name_to_label, internal_gap, filter_points_number, velocity_accpet_error, stop_thresh, \
    change_thresh, frequency, predict_config = loadPredictConfig(config_path)

    info_text_names = get_all_txt_names(gt_path)
    pose_infos = readPoseConfigFile(poseConfigFile)
    lidar_config = readLidarConfigFile(lidarConfigFile)

    start_time = time.time()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        data = range(len(info_text_names))
        start_time = time.time()
    else:
        data = None

    data = comm.bcast(data, root=0)

    length = len(data)//size
    if rank != size-1:
        data_range = range(rank*length, (rank+1)*length)
    else:
        data_range = range(rank*length, len(data))

    sub_velocity_confidence = []
    for frame_id in tqdm(data_range):
        sub_velocity_confidence.append(sub_compute_velocity(frame_id, pcd_path))
    velocity_confidence = comm.gather(sub_velocity_confidence, root=0)
    if rank == 0:
        velocity_confidences = defaultdict(dict)
        for sub_velocity_confidence in velocity_confidence:
            for sub_sub_velocity_confidence in sub_velocity_confidence:
                velocity_confidences[sub_sub_velocity_confidence[0]] = sub_sub_velocity_confidence[1]

        filename = os.path.join(velocity_file)
        with open(filename, "wb") as f:
            pickle.dump(velocity_confidences, f)

        print(time.time() - start_time)


if __name__ == '__main__':
    gt_path = "/home/jiamiaoxu/data/code/detection_evaluation/tracking_Jiamiao/groundtruth"
    pcd_path = "/home/jiamiaoxu/data/code/detection_evaluation/tracking_Jiamiao/pcd"
    velocity_file = "/home/jiamiaoxu/code/attribute_pkl/gt_velocity.pkl"
    eval_path = "/home/jiamiaoxu/code/prediction_evaluation/evaluations/eval_prediction"
    config_path = "/home/jiamiaoxu/code/detection_evaluation/configs/prediction.eval.config"
    poseConfigFile = "/home/jiamiaoxu/data/code/detection_evaluation/tracking_Jiamiao/configs/pose.csv"
    lidarConfigFile = "/home/jiamiaoxu/data/code/detection_evaluation/tracking_Jiamiao/configs/lidars_mkz.cfg"

    velocity_dir = "/".join(velocity_file.split("/")[:-1])
    if not os.path.exists(velocity_dir):
        os.makedirs(velocity_dir)

    compute_velocity(gt_path, pcd_path, config_path, poseConfigFile, lidarConfigFile, velocity_file)