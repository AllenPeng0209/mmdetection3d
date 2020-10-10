import sys
sys.path.append("..")
import os
import multiprocessing
import time
from shapely.geometry.polygon import Polygon
from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from collections import defaultdict
import math
import json
import pickle
import torch
import iou_cuda
import numpy as np
from core.munkres import Munkres
import core.box_ops as box_ops
from core.read_info import get_all_txt_names
from protos import prediction_eval_pb2
from core.occl_dist_assign import angle
from core.calib_trans import pointsMapToSensing_v2
from prediction_data_manager.prediction_data_manager import PredictionDataManager
from google.protobuf import text_format
from protos import prediction_eval_path_pb2

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


def assign(vehicle_error, pedestrian_error, bicycle_error, error, cls, name_to_label, attribute):
    if len(error) != 0:
        error = np.array(error)
        if name_to_label[cls] == 0:
            vehicle_error.append((attribute, error))
        elif name_to_label[cls] == 1:
            pedestrian_error.append(error)
        elif name_to_label[cls] == 2:
            bicycle_error.append(error)


def save(info, eval_path):
    filename = os.path.join(eval_path, "information.pkl")
    with open(filename, "wb") as f:
        pickle.dump(info, f)

#
def compute_results(predict_len, errors, attribute="non_vehicle"):
    count = 0
    data = np.zeros(predict_len)
    size = np.zeros(predict_len)

    if attribute == "non_vehicle":
        for error in errors:
            data[:len(error)] += error
            size[:len(error)] += 1
            count += 1
    elif attribute == "car_all":
        for error in errors:
            data[:len(error[1])] += error[1]
            size[:len(error[1])] += 1
            count += 1
    else:
        for error in errors:
            flag = True
            for i in range(len(attribute)):
                if attribute[i] != -100:
                    flag = flag and (error[0][i] == attribute[i])
            if flag:
                data[:len(error[1])] += error[1]
                size[:len(error[1])] += 1
                count += 1

    mask = np.zeros(predict_len)
    index = np.where(mask == size)
    size[index] = 1
    avg_error = data/size.reshape(1, -1)
    avg_error = np.concatenate((np.array([count]).reshape(1,1), avg_error), axis = 1).reshape(1,-1)
    return avg_error

def draw_table(eval_path, data, filename, predict_len, rows):
    plt.cla()
    columns = ["num"]
    col_tmp = np.round((np.arange(predict_len)+1)*0.5, 1)
    col_tmp = col_tmp.tolist()
    columns += [str(time) + "s" for time in col_tmp]
    the_table = plt.table(cellText=data,
                            colWidths=[0.1] * (predict_len+1),
                            rowLabels=rows,
                            colLabels=columns,
                            loc='center')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(24)
    the_table.scale(4, 4)
    plt.title(filename.split('.')[0], fontsize = 24)
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
    for pos in ['right', 'top', 'bottom', 'left']:
        plt.gca().spines[pos].set_visible(False)
    plt.savefig(Path(eval_path) / filename, bbox_inches='tight', pad_inches=0.05)

def generate_data(vehicle_error, pedestrian_error, bicycle_error, type, eval_path, weight, predict_len, has_priority):
    data_ped = compute_results(predict_len, pedestrian_error)
    data_bic = compute_results(predict_len, bicycle_error)

    # do not consider attribute
    rows = ["car", "pedestrian", "cyclist", "total"]
    data_car_all = compute_results(predict_len, vehicle_error, "car_all")
    data_total_all = weight[0] * data_car_all + weight[1] * data_ped + weight[2] * data_bic
    data = np.concatenate((data_car_all, data_ped, data_bic, data_total_all), axis=0)
    data = np.round(data, 2)
    filename = type + "_eval_result.png"
    draw_table(eval_path, data, filename, predict_len, rows)

    # del low-confidence or high-error velocity
    data_car_nice_velocity_all = compute_results(predict_len, vehicle_error, [1])
    data_total_nice_velocity_all = weight[0] * data_car_nice_velocity_all + weight[1] * data_ped + weight[2] * data_bic
    data = np.concatenate((data_car_nice_velocity_all, data_ped, data_bic, data_total_nice_velocity_all), axis=0)
    data = np.round(data, 2)
    filename = type + "_nice_velocity_eval_result.png"
    draw_table(eval_path, data, filename, predict_len, rows)

    # velocity_based category
    rows = ["stop", "normal_driving", "accelerate", "decelerate"]
    data_car_stop = compute_results(predict_len, vehicle_error, [-100, 0])
    data_car_normal_driving = compute_results(predict_len, vehicle_error, [-100, 1])
    data_car_accelerate = compute_results(predict_len, vehicle_error, [-100, 2])
    data_car_decelerate = compute_results(predict_len, vehicle_error, [-100, 3])
    data = np.concatenate((data_car_stop, data_car_normal_driving, data_car_accelerate, data_car_decelerate), axis=0)
    data = np.round(data, 2)
    filename = type + "_eval_classify_result.png"
    draw_table(eval_path, data, filename, predict_len, rows)

    # velocity_based category and correct velocity
    data_car_stop = compute_results(predict_len, vehicle_error, [1, 0])
    data_car_normal_driving = compute_results(predict_len, vehicle_error, [1, 1])
    data_car_accelerate = compute_results(predict_len, vehicle_error, [1, 2])
    data_car_decelerate = compute_results(predict_len, vehicle_error, [1, 3])
    data = np.concatenate((data_car_stop, data_car_normal_driving, data_car_accelerate, data_car_decelerate), axis=0)
    data = np.round(data, 2)
    filename = type + "_nice_velocity_eval_classify_result.png"
    draw_table(eval_path, data, filename, predict_len, rows)

    # turn or no-turn except stop object
    rows = ["turn", "no-turn"]
    data_car_turn = compute_results(predict_len, vehicle_error, [-100, -100, 1])
    data_car_no_turn = compute_results(predict_len, vehicle_error, [-100, -100, 0])
    data = np.concatenate((data_car_turn, data_car_no_turn), axis=0)
    data = np.round(data, 2)
    filename = type + "_eval_turn_result.png"
    draw_table(eval_path, data, filename, predict_len, rows)

    # turn or no-turn and nice velocity except stop object
    data_car_turn = compute_results(predict_len, vehicle_error, [1, -100, 1])
    data_car_no_turn = compute_results(predict_len, vehicle_error, [1, -100, 0])
    data = np.concatenate((data_car_turn, data_car_no_turn), axis=0)
    data = np.round(data, 2)
    filename = type + "_nice_velocity_eval_turn_result.png"
    draw_table(eval_path, data, filename, predict_len, rows)

    # cut-in or no-cut-in except stop object
    rows = ["cut-in", "no-cut-in"]
    data_car_cut_in = compute_results(predict_len, vehicle_error, [-100, -100, -100, 1])
    data_car_no_cut_in = compute_results(predict_len, vehicle_error, [-100, -100, -100, 0])
    data = np.concatenate((data_car_cut_in, data_car_no_cut_in), axis=0)
    data = np.round(data, 2)
    filename = type + "_eval_cut_in_result.png"
    draw_table(eval_path, data, filename, predict_len, rows)

    # cut-in or no-cut-in and nice velocity except stop object
    data_car_cut_in = compute_results(predict_len, vehicle_error, [1, -100, -100, 1])
    data_car_no_cut_in = compute_results(predict_len, vehicle_error, [1, -100, -100, 0])
    data = np.concatenate((data_car_cut_in, data_car_no_cut_in), axis=0)
    data = np.round(data, 2)
    filename = type + "_nice_velocity_eval_cut_in_result.png"
    draw_table(eval_path, data, filename, predict_len, rows)

    # on lane or off-lane except stop object
    rows = ["on-lane", "off-lane"]
    data_car_on_lane = compute_results(predict_len, vehicle_error, [-100, -100, -100, -100, 1])
    data_car_off_lane = compute_results(predict_len, vehicle_error, [-100, -100, -100, -100, 0])
    data = np.concatenate((data_car_on_lane, data_car_off_lane), axis=0)
    data = np.round(data, 2)
    filename = type + "_eval_on_lane_result.png"
    draw_table(eval_path, data, filename, predict_len, rows)

    # on lane or off lane and nice velocity except stop object
    data_car_on_lane = compute_results(predict_len, vehicle_error, [1, -100, -100, -100, 1])
    data_car_off_lane = compute_results(predict_len, vehicle_error, [1, -100, -100, -100, 0])
    data = np.concatenate((data_car_on_lane, data_car_off_lane), axis=0)
    data = np.round(data, 2)
    filename = type + "_nice_velocity_eval_on_lane_result.png"
    draw_table(eval_path, data, filename, predict_len, rows)

    if has_priority:
        # high and low priority except stop object
        rows = ["high-priority", "low-priority"]
        data_car_high_priority = compute_results(predict_len, vehicle_error, [-100, 1, -100, -100, -100, 1])
        data_car_low_priority = compute_results(predict_len, vehicle_error, [-100, 1, -100, -100, -100, 0])
        data = np.concatenate((data_car_high_priority, data_car_low_priority), axis=0)
        data = np.round(data, 2)
        filename = type + "_eval_priority_result.png"
        draw_table(eval_path, data, filename, predict_len, rows)

        # on lane or off lane and nice velocity except stop object
        data_car_high_priority = compute_results(predict_len, vehicle_error, [1, 1, -100, -100, -100, 1])
        data_car_low_priority = compute_results(predict_len, vehicle_error, [1, 1, -100, -100, -100, 0])
        data = np.concatenate((data_car_high_priority, data_car_low_priority), axis=0)
        data = np.round(data, 2)
        filename = type + "_nice_velocity_eval_priority_result.png"
        draw_table(eval_path, data, filename, predict_len, rows)


def subMergeTable(eval_path, suffix):
    filename1 = "/L1" + suffix
    filename2 = "/L2" + suffix
    filename3 = "/L2_along" + suffix
    filename4 = "/L2_across" + suffix
    filename = "/total" + suffix
    img1 = mpimg.imread(eval_path + filename1)
    img2 = mpimg.imread(eval_path + filename2)
    img3 = mpimg.imread(eval_path + filename3)
    img4 = mpimg.imread(eval_path + filename4)
    img = np.concatenate((img1, img2, img3, img4), axis=0)
    plt.imsave(eval_path + filename, img)
    os.remove(eval_path + filename1)
    os.remove(eval_path + filename2)
    os.remove(eval_path + filename3)
    os.remove(eval_path + filename4)

def MergeTable(eval_path, has_priority):
    suffix  = "_eval_result.png"
    subMergeTable(eval_path, suffix)
    suffix = "_nice_velocity_eval_result.png"
    subMergeTable(eval_path, suffix)
    suffix = "_eval_classify_result.png"
    subMergeTable(eval_path, suffix)
    suffix = "_nice_velocity_eval_classify_result.png"
    subMergeTable(eval_path, suffix)
    suffix = "_eval_turn_result.png"
    subMergeTable(eval_path, suffix)
    suffix = "_nice_velocity_eval_turn_result.png"
    subMergeTable(eval_path, suffix)
    suffix = "_eval_cut_in_result.png"
    subMergeTable(eval_path, suffix)
    suffix = "_nice_velocity_eval_cut_in_result.png"
    subMergeTable(eval_path, suffix)
    suffix = "_eval_on_lane_result.png"
    subMergeTable(eval_path, suffix)
    suffix = "_nice_velocity_eval_on_lane_result.png"
    subMergeTable(eval_path, suffix)

    if has_priority:
        suffix = "_eval_priority_result.png"
        subMergeTable(eval_path, suffix)
        suffix = "_nice_velocity_eval_priority_result.png"
        subMergeTable(eval_path, suffix)

def l2_error(data1, data2):
    return np.sqrt(np.sum((data1 - data2) ** 2))

def l1_error(data1, data2):
    return abs(data1[0] - data2[0]) + abs(data1[1] - data2[1])

def l2_across_along_error(pre_data1, data1, data2):
    direction = [abs(data1[0] - pre_data1[0]), abs(data1[1] - pre_data1[1])]
    delta = [abs(data2[0] - data1[0]), abs(data2[1] - data1[1])]

    angle1 = angle(direction)
    angle2 = angle(delta)
    degree = abs(angle1 - angle2)

    dist_error = math.sqrt(delta[0] ** 2 + delta[1] ** 2)
    across = abs(dist_error * math.sin(degree / 180 * math.pi))
    along = abs(dist_error * math.cos(degree / 180 * math.pi))

    return across, along

def get_match_pair(gt_num, dt_num, gt_names, dt_names, iou, overlap_thresh, name_to_label):

    hm = Munkres()
    max_cost = 1e9

    cost_matrix = []
    for i in range(gt_num):
        cost_row = []
        for j in range(dt_num):
            if(iou[i][j] > overlap_thresh[name_to_label[gt_names[i].lower()]]):
                cost_row.append(1-iou[i][j])
            else:
                cost_row.append(max_cost)
        cost_matrix.append(cost_row)

    if gt_num == 0:
        cost_matrix = [[]]

    association_matrix = hm.compute(cost_matrix)

    match_pairs = {}
    gt_match_index = []
    pre_match_index = []
    for row, col in association_matrix:
        # apply gating on boxoverlap
        c = cost_matrix[row][col]
        if c < max_cost:
            gt_match_index.append(row)
            pre_match_index.append(col)
            match_pairs.update({col:row})

    return match_pairs, gt_match_index, pre_match_index

def get_iou_3D(gt_num, dt_num, gt_boxes, dt_boxes, gt_corners, dt_corners):
    """return intersections over ground truths and detections of a image

    Args:
        gt_boxes: numpy ndarrays including location dimension and rotation_y of ground truth boxes
        dt_boxes: numpy ndarrays including location dimension and rotation_y of detection boxes

    Returns:
        a numpy ndarray of iou values between gt and dt boxes,
        the number of columns if the number of detection boxes
        for example:

        iou [[0.         0.         0.4792311  0.         ]
                [0.8098454  0.         0.         0.         ]
                [0.         0.         0.         0.7279645  ]
    """
    if gt_num <= 0 or dt_num <= 0:
        return None
    iou = np.zeros([gt_num, dt_num], dtype=np.float32)
    for i in range(gt_num):
        for j in range(dt_num):
            iw = (min(gt_boxes[i, 2] + gt_boxes[i, 5],
                        dt_boxes[j, 2] + dt_boxes[j, 5]) - max(gt_boxes[i, 2], dt_boxes[j, 2]))
            if iw > 0:
                p1 = Polygon(gt_corners[i])
                p2 = Polygon(dt_corners[j])
                # first get the intersection of the undersides, then times it with the min height
                inc = p1.intersection(p2).area * iw
                if inc > 0:
                    iou[i, j] = inc / (p1.area * gt_boxes[i, 5] +
                                        p2.area * dt_boxes[j, 5] - inc)

    return iou

def get_iou_bev(gt_num, dt_num, gt_corners, dt_corners):
    if gt_num <= 0 or dt_num <= 0:
        return None

    iou = iou_cuda.iou_forward(torch.cuda.FloatTensor(gt_corners), torch.cuda.FloatTensor(dt_corners))
    iou = iou.cpu().numpy().reshape(gt_num, dt_num)

    return iou

def readPredict(pre_path, timestamp, predict_len, has_priority):
    try:
        with open(os.path.join(pre_path, timestamp), "rb") as f:
            objects = json.load(f)
    except:
        objects = {}
        objects["objects"] = []

    pre_names = []
    pre_loc = []
    pre_dims = []
    pre_yaws = []
    pre_velocities = []
    pre_all_pos = []
    pre_ids = []
    pre_priorities = []
    for obj_idx, obj in enumerate(objects["objects"]):
        pre_pos = np.array(obj["prediction"]).reshape(-1, predict_len, 2)
        pre_all_pos.append(pre_pos)
        pre_names.append(obj['type'].lower())
        pre_loc.append([obj["position"]["x"], obj["position"]["y"], obj["position"]["z"]])
        pre_dims.append([obj["bounding_box"]["length"], obj["bounding_box"]["width"], obj["bounding_box"]["height"]])
        pre_velocities.append([obj["velocity"]["x"], obj["velocity"]["y"], obj["velocity"]["z"]])
        pre_yaws.append(obj["heading"])
        pre_ids.append(obj["id"])
        if has_priority:
            pre_priorities.append(obj["priority"])

    pre_loc = np.array(pre_loc).reshape(-1, 3)
    pre_dims = np.array(pre_dims).reshape(-1, 3)
    pre_yaws = np.array(pre_yaws)
    pre_boxes = np.concatenate([pre_loc, pre_dims, pre_yaws[..., np.newaxis]], axis=1)
    box_ops.change_box3d_center_(pre_boxes, src=[0.5, 0.5, 0], dst=[0.5, 0.5, 0.5])
    pre_corners = box_ops.get_corners(pre_boxes[:, :2], pre_boxes[:, 3:5], pre_boxes[:, 6])

    return pre_all_pos, pre_boxes, pre_corners, pre_names, pre_ids, pre_velocities, pre_priorities

def best_match(errors1, errors2, errors3, errors4):
    error1 = errors1[errors1.mean(axis=1).argmin(), :]
    error2 = errors2[errors2.mean(axis=1).argmin(), :]
    error3 = errors3[errors3.mean(axis=1).argmin(), :]
    error4 = errors4[errors4.mean(axis=1).argmin(), :]
    return error1, error2, error3, error4

def get_attribute(gt_velocity_confidence_cur, gt_id, pre_velocity_cur, pre_priority_cur, has_priority,
                  gt_accelerate_cur, gt_turn_cur, gt_cutIn_cur, gt_onLane_cur):
    attribute = [0] * 6  #
    gt_velocity_cur = gt_velocity_confidence_cur[gt_id][2]
    if gt_velocity_confidence_cur[gt_id][0] == 0 or gt_velocity_confidence_cur[gt_id][0] == 1:   # confidence
        # if gt_velocity_cur[0]-pre_velocity_cur[0] < velocity_accpet_error \
        #         and gt_velocity_cur[1]-pre_velocity_cur[1] < velocity_accpet_error:
        if ((gt_velocity_cur[0]-pre_velocity_cur[0])**2+(gt_velocity_cur[1]-pre_velocity_cur[1])**2)**0.5 < velocity_accpet_error:
            attribute[0] = 1

    abs_gt_velocity_cur = (gt_velocity_cur[0]**2 + gt_velocity_cur[1]**2)**0.5
    if abs_gt_velocity_cur < stop_thresh:   # stop or low speed
        attribute[1] = 0
    elif gt_accelerate_cur[gt_id] == 1:
        attribute[1] = 2    # accelerate
    elif gt_accelerate_cur[gt_id] == -1:
        attribute[1] = 3    # decelerate
    else:
        attribute[1] = 1    # normal driving

    if attribute[1] == 0:
        attribute[2] = -1   # do not consider
    elif gt_turn_cur[gt_id] == 1:
        attribute[2] = 1    # turn
    else:
        attribute[2] = 0    # no turn

    if attribute[1] == 0:
        attribute[3] = -1   # do not consider
    elif gt_cutIn_cur[gt_id] == 1:
        attribute[3] = 1    # cut in
    else:
        attribute[3] = 0    # no cut in

    if attribute[1] == 0:
        attribute[4] = -1   # do not consider
    elif gt_onLane_cur[gt_id] == 1:
        attribute[4] = 1    # on lane
    else:
        attribute[4] = 0    # off lane

    if has_priority:
        if attribute[1] == 0:
            attribute[5] = -1   # do not consider
        elif pre_priority_cur == 1:    # high priority
            attribute[5] = 1
        else:
            attribute[5] = 0    # low priority

    return attribute

def perform_sub_evaluation(gt_corners, gt_boxes, gt_names, gt_ids, \
                           pre_corners, pre_boxes, pre_names, pre_all_pos, pre_ids, pre_velocities,
                           pre_priorities, has_priority, timestamp):
    # get match pairs
    gt_obj_num = len(gt_names)
    dt_obj_num = len(pre_names)
    iou = get_iou_bev(gt_obj_num, dt_obj_num, gt_corners, pre_corners) if cal_iou_bev \
        else get_iou_3D(gt_obj_num, dt_obj_num, gt_boxes, pre_boxes, gt_corners, pre_corners)
    match_pairs, gt_match_index, pre_match_index = get_match_pair(gt_obj_num, dt_obj_num, \
                                gt_names, pre_names, iou, overlap_thresh, name_to_label)


    # compute error
    gt_match_id = []
    pre_match_id = []
    gt_pos_tmp = []
    pre_pos_tmp = []
    vehicle_L1_error = []
    pedestrian_L1_error = []
    bicycle_L1_error = []
    vehicle_L2_error = []
    pedestrian_L2_error = []
    bicycle_L2_error = []
    vehicle_across_error = []
    pedestrian_across_error = []
    bicycle_across_error = []
    vehicle_along_error = []
    pedestrian_along_error = []
    bicycle_along_error = []

    gt_pre_dict = {}
    for obj_idx in (pre_match_index):
        # get gt trajectory
        target_id = gt_ids[match_pairs[obj_idx]]
        gt_trajectory = predictionDataManager.getFutureTrajectory(target_id, timestamp, future_length)
        gt_trajectory = np.array(gt_trajectory)[:, 1:4]
        if gt_trajectory.shape[0] == future_length:
            gt_trajectory = gt_trajectory[selected, :]    # global coord
        elif gt_trajectory.shape[0] > 5:
            part_selected = list(range(5, gt_trajectory.shape[0], internal_gap))
            gt_trajectory = gt_trajectory[part_selected, :]
        else:
            continue

        pose_info = predictionDataManager.getPose(timestamp)
        config_lidars = predictionDataManager._config_lidars
        gt_trajectory = pointsMapToSensing_v2(pose_info, gt_trajectory, config_lidars)[:, :2] # local coord

        # get pre trajectory
        pre_trajectory = pre_all_pos[obj_idx]

        # get attribute
        attribute = []
        if name_to_label[gt_names[match_pairs[obj_idx]].lower()] == 0:  # for vehicle
            gt_velocity_confidence_cur = predictionDataManager._gt_velocity[timestamp]
            gt_accelerate_cur = predictionDataManager._gt_accelerate[timestamp]
            gt_turn_cur = predictionDataManager._gt_turn[timestamp]
            gt_cutIn_cur = predictionDataManager._gt_cut_in[timestamp]
            gt_onLane_cur = predictionDataManager._gt_on_lane[timestamp]
            pre_velocity = pre_velocities[obj_idx]
            if has_priority:
                pre_priority = pre_priorities[obj_idx]
            else:
                pre_priority = []
            attribute = get_attribute(gt_velocity_confidence_cur, target_id, pre_velocity, pre_priority, has_priority,
                                      gt_accelerate_cur, gt_turn_cur, gt_cutIn_cur, gt_onLane_cur)

        # error
        L1_errors, L2_errors, across_errors, along_errors = [], [], [], []
        for num in range(num_modal):
            L1_errors.append([l1_error(gt_trajectory[i, :], pre_trajectory[num, i, :]) for i in range(gt_trajectory.shape[0])])
            L2_errors.append([l2_error(gt_trajectory[i, :], pre_trajectory[num, i, :]) for i in range(gt_trajectory.shape[0])])

            across_error, along_error = [], []
            for i in range(gt_trajectory.shape[0]):
                if i == 0:
                    res = l2_across_along_error(gt_boxes[match_pairs[obj_idx], 0:2], gt_trajectory[i, :], pre_trajectory[num, i, :])
                else:
                    res = l2_across_along_error(gt_trajectory[i-1, :], gt_trajectory[i, :], pre_trajectory[num, i, :])
                across_error.append(res[0])
                along_error.append(res[1])
            across_errors.append(across_error)
            along_errors.append(along_error)

        L1_errors, L2_errors = np.array(L1_errors), np.array(L2_errors)
        across_errors, along_errors = np.array(across_errors), np.array(along_errors)
        L1_error, L2_error, across_error, along_error = best_match(L1_errors, L2_errors, across_errors, along_errors)

        cls = gt_names[match_pairs[obj_idx]]

        # save information for visualization
        # Todo:
        gt_match_id.append(target_id)
        pre_match_id.append(pre_ids[obj_idx])
        gt_pre_dict[target_id] = pre_ids[obj_idx]
        gt_pos_tmp.append([])
        pre_pos_tmp.append([])

        assign(vehicle_L1_error, pedestrian_L1_error, bicycle_L1_error, L1_error, cls, name_to_label, attribute)
        assign(vehicle_L2_error, pedestrian_L2_error, bicycle_L2_error, L2_error, cls, name_to_label, attribute)
        assign(vehicle_across_error, pedestrian_across_error, bicycle_across_error, across_error, cls, name_to_label, attribute)
        assign(vehicle_along_error, pedestrian_along_error, bicycle_along_error, along_error, cls, name_to_label, attribute)

    return (timestamp, gt_match_id, pre_match_id, gt_pre_dict, pre_pos_tmp,
            vehicle_L1_error, pedestrian_L1_error, bicycle_L1_error,
            vehicle_L2_error, pedestrian_L2_error, bicycle_L2_error,
            vehicle_across_error, pedestrian_across_error, bicycle_across_error,
            vehicle_along_error, pedestrian_along_error, bicycle_along_error)

def perform_evaluation(predict_config, pre_text_names, has_priority, pre_path, eval_path, pool_num):
    global interest_labels, overlap_thresh, future_length, selected, cal_iou_bev, num_modal, name_to_label, internal_gap, \
            velocity_accpet_error, stop_thresh, acclerate_thresh, frequency

    # load config parameters
    (interest_labels, weight, overlap_thresh, future_length, selected, cal_iou_bev, num_modal, name_to_label,\
     internal_gap, filter_points_number, velocity_accpet_error, stop_thresh, acclerate_thresh, frequency, config) = predict_config

    pre_length = len(pre_text_names)

    print("start prediction...")
    pool = multiprocessing.Pool(pool_num)
    res = []
    for i in tqdm(range(pre_length)):
        # loading prediction results
        timestamp = pre_text_names[i]
        pre_all_pos, pre_boxes, pre_corners, pre_names, pre_ids, pre_velocities, pre_priorities \
            = readPredict(pre_path, timestamp, len(selected), has_priority)

        # loading gt
        timestamp = int(''.join(timestamp.split(".")[0].split("_")))
        gt_corners, gt_boxes, gt_names, gt_ids, gt_yaws = \
            predictionDataManager.getObjectInfos(timestamp)

        # evaluation per frame
        #perform_sub_evaluation(gt_corners, gt_boxes, gt_names, gt_ids, \
        #                     pre_corners, pre_boxes, pre_names, pre_all_pos, pre_ids, pre_velocities,
        #                     pre_priorities, timestamp)
        res.append(pool.apply_async(perform_sub_evaluation, (gt_corners, gt_boxes, gt_names, gt_ids, \
                             pre_corners, pre_boxes, pre_names, pre_all_pos, pre_ids, pre_velocities,
                             pre_priorities, has_priority, timestamp)))

    # generate results
    info = defaultdict(list)
    vehicle_L1_error = []
    pedestrian_L1_error = []
    bicycle_L1_error = []
    vehicle_L2_error = []
    pedestrian_L2_error = []
    bicycle_L2_error = []
    vehicle_across_error = []
    pedestrian_across_error = []
    bicycle_across_error = []
    vehicle_along_error = []
    pedestrian_along_error = []
    bicycle_along_error = []
    for i in tqdm(range(len(res))):
        timeStamp = res[i].get()[0]
        # save information for visualization
        info[timeStamp].append(res[i].get()[1])
        info[timeStamp].append(res[i].get()[2])
        info[timeStamp].append(res[i].get()[3])
        # info[timeStamp].append(res[i].get()[4])
        vehicle_L1_error += res[i].get()[5]
        pedestrian_L1_error += res[i].get()[6]
        bicycle_L1_error += res[i].get()[7]
        vehicle_L2_error += res[i].get()[8]
        pedestrian_L2_error += res[i].get()[9]
        bicycle_L2_error += res[i].get()[10]
        vehicle_across_error += res[i].get()[11]
        pedestrian_across_error += res[i].get()[12]
        bicycle_across_error += res[i].get()[13]
        vehicle_along_error += res[i].get()[14]
        pedestrian_along_error += res[i].get()[15]
        bicycle_along_error += res[i].get()[16]

    pool.close()
    pool.join()

    generate_data(vehicle_L1_error, pedestrian_L1_error, bicycle_L1_error, "L1", eval_path, weight, len(selected), has_priority)
    generate_data(vehicle_L2_error, pedestrian_L2_error, bicycle_L2_error, "L2", eval_path, weight, len(selected), has_priority)
    generate_data(vehicle_across_error, pedestrian_across_error, bicycle_across_error, "L2_across", eval_path, weight, len(selected), has_priority)
    generate_data(vehicle_along_error, pedestrian_along_error, bicycle_along_error, "L2_along", eval_path, weight, len(selected), has_priority)
    save(info, eval_path)

    MergeTable(eval_path, has_priority)

def evaluation(gt_path, pre_path, pcd_path, map_path, gt_config_path, trajectory_file, velocity_file,
               accelerate_file, turn_file, cut_in_file, on_lane_file, eval_path, config_path,
               has_priority, pool_num=10):
    global predictionDataManager

    # load config
    predict_config = loadPredictConfig(config_path)

    # load gt
    predictionDataManager = PredictionDataManager(gt_path, pcd_path, map_path, gt_config_path,
                                                  trajectory_file, velocity_file, accelerate_file,
                                                  turn_file, cut_in_file, on_lane_file, predict_config[-1])

    # create results directory
    if not os.path.exists(eval_path):
        os.makedirs(eval_path)

    # load pre timestamps
    pre_text_names = get_all_txt_names(pre_path)

    perform_evaluation(predict_config, pre_text_names, has_priority, pre_path, eval_path, pool_num)

def main(params_config_path):
    if isinstance(params_config_path, str):
        config = prediction_eval_path_pb2.PredictionEvalPathConfig()
        with open(params_config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, config)
    else:
        config = params_config_path

    gt_path = config.prediction_eval_path_reader.gt_path
    pre_path = config.prediction_eval_path_reader.pre_path
    pcd_path = config.prediction_eval_path_reader.pcd_path
    map_path = config.prediction_eval_path_reader.map_path
    trajectory_file = config.prediction_eval_path_reader.trajectory_file
    velocity_file = config.prediction_eval_path_reader.velocity_file
    accelerate_file = config.prediction_eval_path_reader.accelerate_file
    turn_file = config.prediction_eval_path_reader.turn_file
    cut_in_file = config.prediction_eval_path_reader.cut_in_file
    on_lane_file = config.prediction_eval_path_reader.on_lane_file
    gt_config_path = config.prediction_eval_path_reader.gt_config_path
    eval_path = config.prediction_eval_path_reader.eval_path
    config_path = config.prediction_eval_path_reader.config_path
    has_priority = config.prediction_eval_path_reader.has_priority
    pool_num = config.prediction_eval_path_reader.pool_num

    t = time.time()

    evaluation(gt_path, pre_path, pcd_path, map_path, gt_config_path, trajectory_file, \
               velocity_file, accelerate_file, turn_file, cut_in_file, on_lane_file,
               eval_path, config_path, has_priority, pool_num)
    print("Evaluation Finished!")
    print("The total evaluation time:%f" % (time.time() - t))
    print("All evaluation results are placed in the path: {}".format(eval_path))


if __name__ == '__main__':
    params_config_path = "/home/jiamiaoxu/code/detection_evaluation/configs/prediction.eval.path.config"
    main(params_config_path)
    """
    gt_path = "/home/jiamiaoxu/data/code/detection_evaluation/tracking_Jiamiao/groundtruth"
    pre_path = "/home/jiamiaoxu/format_fix_cutinaccv2"
    pcd_path = "/home/jiamiaoxu/data/code/detection_evaluation/tracking_Jiamiao/pcd"
    map_path = "/home/jiamiaoxu/data/our_data/prediction_data/hdmap/longhua_map_0422.bin"
    trajectory_file = "/home/jiamiaoxu/code/attribute_pkl/gt_trajectories.pkl"
    velocity_file = "/home/jiamiaoxu/code/attribute_pkl/gt_velocity.pkl"
    accelerate_file = "/home/jiamiaoxu/code/attribute_pkl/gt_accelerate.pkl"
    turn_file = "/home/jiamiaoxu/code/attribute_pkl/gt_turn.pkl"
    cut_in_file = "/home/jiamiaoxu/code/attribute_pkl/gt_cutIn.pkl"
    on_lane_file = "/home/jiamiaoxu/code/attribute_pkl/gt_onLane.pkl"
    gt_config_path = "/home/jiamiaoxu/data/our_data/prediction_data/20190412_rain_1/config"
    eval_path = "/home/jiamiaoxu/code/detection_evaluation/evaluations/eval_prediction"
    config_path = "/home/jiamiaoxu/code/detection_evaluation/configs/prediction.eval.config"

    has_priority = False

    t = time.time()

    if len(sys.argv) > 1:
        pool_num = int(sys.argv[1])
    else:
        pool_num = 10

    evaluation(gt_path, pre_path, pcd_path, map_path, gt_config_path, trajectory_file, \
               velocity_file, accelerate_file, turn_file, cut_in_file, on_lane_file,
               eval_path, config_path, has_priority, pool_num)
    print("Evaluation Finished!")
    print("The total evaluation time:%f" %(time.time()-t))
    print("All evaluation results are placed in the path: {}".format(eval_path))
    """


