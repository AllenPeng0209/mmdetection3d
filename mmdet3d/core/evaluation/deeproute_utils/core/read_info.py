import os
import json
import numpy as np
import pickle
from protos.config_lidar_pb2 import LidarArray
from google.protobuf import text_format
from core.box_ops import get_corners_v2
from IPython import embed
def readInfo_per_frame(info_path, timeStamp, name_to_label, interest_labels, velocity = True):
    # compute occlusion attribute of gt
    try:
        with open(os.path.join(info_path, timeStamp), "rb") as f:
            objects = json.load(f)
    except:
        objects = {}
        objects["objects"] = []

    info_names = []
    info_ids = []
    info_loc = []
    info_dims = []
    info_yaws = []
    info_scores = []
  
    if len(objects["objects"]) > 0 :
        for obj in objects["objects"]:
            if velocity and name_to_label[obj['type'].lower()] < len(interest_labels) and interest_labels[name_to_label[obj['type'].lower()]].lower() != "car":
                continue
            if (not velocity) and obj['type'].lower() == "cone" and obj["position"]["x"] < 0:
                continue

            info_names.append(obj['type'].lower())
            if 'id' in obj:
                info_ids.append(obj['id'])
            info_loc.append([obj["position"]["x"], obj["position"]["y"], obj["position"]["z"]])
            info_dims.append([obj["bounding_box"]["length"], obj["bounding_box"]["width"], obj["bounding_box"]["height"]])
            info_yaws.append(obj["heading"])
            if 'score' in obj:
                info_scores.append(obj['score'])

    info_loc = np.array(info_loc).reshape(-1, 3)
    info_dims = np.array(info_dims).reshape(-1, 3)
    info_yaws = np.array(info_yaws)
    info_scores = np.array(info_scores)
    return info_loc, info_dims, info_yaws, info_ids, info_names, info_scores

def readInfo(info_path, timeStamp, name_to_label, interest_labels, velocity = True):

    loc, dims, yaws, ids, names = readInfo_per_frame(info_path, timeStamp, name_to_label, interest_labels, velocity)

    gt_boxes = np.concatenate([loc, dims, yaws[..., np.newaxis]], axis=1)

    gt_corners = get_corners_v2(gt_boxes[:, :3], gt_boxes[:, 3:6], gt_boxes[:, 6]) if gt_boxes.shape[0] > 0 else np.array([])
    gt_corners = gt_corners.reshape(-1, 8, 3)

    surface = []
    surface.append(gt_corners[:, [3, 2, 1, 0], :].reshape(-1, 1, 4, 3))
    surface.append(gt_corners[:, [6, 7, 4, 5], :].reshape(-1, 1, 4, 3))
    surface.append(gt_corners[:, [1, 5, 4, 0], :].reshape(-1, 1, 4, 3))
    surface.append(gt_corners[:, [3, 7, 6, 2], :].reshape(-1, 1, 4, 3))
    surface.append(gt_corners[:, [2, 6, 5, 1], :].reshape(-1, 1, 4, 3))
    surface.append(gt_corners[:, [4, 7, 3, 0], :].reshape(-1, 1, 4, 3))

    surfaces = np.concatenate(surface, axis=1)
    return loc, ids, names, surfaces, scores


def get_all_txt_names(path):
    text_names = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1] == '.txt':
                text_names.append(file)

    text_names.sort()
    return text_names

def get_timeStamps(det_path):
    timeStamps = []
    for root, dirs, files in os.walk(det_path):
        for file in files:
            if os.path.splitext(file)[1] == '.txt':
                timeStamps.append(file.split('.')[0])

    timeStamps.sort()
    return timeStamps

def readPoseConfigFile(pose_config_file):
    with open(pose_config_file, 'r') as f:
        lines = f.readlines()

    pose_infos = {}
    for i, line in enumerate(lines):
        if i == 0:
            continue
        fields = line.strip().split(", ")
        if len(fields) != 7:
            print("Load pose error: {}, in line: {}".format(pose_config_file, line))
            continue
        timestamp = fields[0]
        translation = (float(fields[1]), float(fields[2]), float(fields[3]))
        roll_pitch_yaw = (float(fields[4]), float(fields[5]), float(fields[6]))
        pose_infos[timestamp] = (translation, roll_pitch_yaw)
    return pose_infos

def readLidarConfigFile(config_file):
    configs = LidarArray()
    if not os.path.exists(config_file):
        print("Warn: {} not exists!".format(config_file))
        return configs
    with open(config_file, 'r') as f:
        text_format.Parse(f.read(), configs)
    return configs

def readPickleFile(filename):
    try:
        with open(filename, "rb") as f:
            info = pickle.load(f)
        return info
    except:
        print("fail to open " + filename)
        raise

def savePickleFile(info, filename):
    try:
        with open(filename, "rb") as f:
            pickle.dump(info, f)
    except:
        print("fail to save " + filename)


