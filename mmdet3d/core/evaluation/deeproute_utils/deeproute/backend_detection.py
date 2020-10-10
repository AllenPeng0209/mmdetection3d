import sys
sys.path.append('..')
import os
import numpy as np
import base64
import pcl
from flask import Flask, jsonify, request
from flask_cors import CORS
import pickle
from pathlib import Path
import fire
import core.box_ops as box_ops
from protos import detection_eval_pb2
from google.protobuf import text_format
from core.calib_trans import *
from core.read_info import *
app = Flask("second")
CORS(app)

class SecondBackend:
    def __init__(self):
        self.pcd_path = None
        self.gt_path = None
        self.det_path = None
        self.eval_path = None
        self.kitti_infos = None
        self.image_idxes = None
        self.error_image_idxes = None
        self.det_path = None
        self.pose_config_file = None
        self.data_index_file = None
        self.camera_config_file = None
        self.lidar_config_file = None
        self.inference_ctx = None
        self.eval_info = None
        self.PCformat = True
        self.occluded_info = {}
        self.occlusionPath = None
        self.interest_labels = None
        self.name_to_label = None
        self.distance_thresh = None

BACKEND = SecondBackend()

def error_response(msg):
    response = {}
    response["status"] = "error"
    response["message"] = "[ERROR]" + msg
    print("[ERROR]" + msg)
    return response

@app.route('/api/read_all', methods=['POST'])
def read_all():
    global BACKEND
    instance = request.json
    occlusionPath = Path(instance["occlusion_path"])
    gt_path = Path(instance["gt_path"])
    det_path = Path(instance["det_path"])
    pcd_path = Path(instance["pcd_path"])
    eval_path = Path(instance["eval_path"])
    config_path = instance["config_path"]
    type = instance["type"]

    BACKEND.occlusionPath = occlusionPath
    BACKEND.gt_path = gt_path
    BACKEND.det_path = det_path
    BACKEND.pcd_path = pcd_path
    BACKEND.eval_path = eval_path
    BACKEND.PCformat = instance["pc_format"]

    response = {"status": "normal"}

    BACKEND.timeStamps = get_timeStamps(gt_path)
    response["timeStamps"] = BACKEND.timeStamps

    try:
        with open(occlusionPath, 'rb') as fp:
            BACKEND.occluded_info = pickle.load(fp)[1]
    except:
        BACKEND.occluded_info = {}

    if isinstance(config_path, str):
        config = detection_eval_pb2.DeeprouteDetectionEvalConfig()
        with open(config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, config)
    else:
        config = config_path
    interest_labels = config.deeproute_eval_input_reader.interest_labels
    name_to_label = {}
    for sample_group in config.deeproute_eval_input_reader.name_to_label.sample_groups:
        name_to_label.update(dict(sample_group.name_to_num))

    distance_thresh = list(config.deeproute_eval_input_reader.distance_thresh)

    BACKEND.interest_labels = interest_labels
    BACKEND.name_to_label = name_to_label
    BACKEND.distance_thresh = distance_thresh

    focus_range = instance["focus_range"]
    if instance["removeOccluded"]:
        if focus_range == "full":
            if type == "total":
                filename = eval_path / 'total_full_range_rm_occluded.pkl'
            else:
                filename = eval_path / (type + "_full_range_rm_occluded.pkl")
        else:
            if type == "total":
                filename = eval_path / ("total_within_" + str(focus_range) + "_rm_occluded.pkl")
            else:
                filename = eval_path / (type + "_within_" + str(focus_range) + "_rm_occluded.pkl")
    else:
        if focus_range == "full":
            if type == "total":
                filename = eval_path / 'total_full_range.pkl'
            else:
                filename = eval_path / (type + "_full_range.pkl")
        else:
            if type == "total":
                filename = eval_path / ("total_within_" + str(focus_range) + ".pkl")
            else:
                filename = eval_path / (type + "_within_" + str(focus_range) + ".pkl")

    with open(filename, 'rb') as fp:
        eval_info = pickle.load(fp)
        error_image_idxes = list(eval_info.keys())
        BACKEND.eval_info = eval_info
    BACKEND.error_image_idxes = [key.split('.')[0] for key in error_image_idxes]
    response["error_image_indexes"] = BACKEND.error_image_idxes

    response = jsonify(results=[response])
    response.headers['Access-Control-Allow-Headers'] = '*'
    return response

@app.route('/api/get_pointcloud', methods=['POST'])
def get_pointcloud():
    global BACKEND
    instance = request.json
    response = {"status": "normal"}
    if BACKEND.pcd_path is None:
        return error_response("pcd path is not set")

    timeStamp = instance["timeStamp"]

    path = os.path.join(BACKEND.pcd_path, timeStamp+".pcd")

    if BACKEND.PCformat:
        points = np.fromfile(path, dtype=np.float32)
        points = points.reshape([-1, 4])
        response["num_features"] = points.shape[1]
    else:
        pc = pcl.load(path)
        points = np.array(pc)
        response["num_features"] = points.shape[1]

    enable_int16 = instance["enable_int16"]
    
    if enable_int16:
        int16_factor = instance["int16_factor"]
        points *= int16_factor
        points = points.astype(np.int16)
    pc_str = base64.b64encode(points.tobytes())
    response["pointcloud"] = pc_str.decode("utf-8")


    gt_loc, gt_dims, gt_yaws, gt_names, dt_loc, dt_dims, dt_yaws, dt_names = \
        get_lidar_box_per_frame(BACKEND.gt_path, BACKEND.det_path, timeStamp)

    response["gt_loc"] = gt_loc
    response["gt_dims"] = gt_dims
    response["gt_yaws"] = gt_yaws
    response["gt_names"] = gt_names
    response["dt_loc"] = dt_loc
    response["dt_dims"] = dt_dims
    response["dt_yaws"] = dt_yaws
    response["dt_names"] = dt_names
    response["distance"] = BACKEND.distance_thresh

    gt_name_trans = []
    for name in gt_names:
        if BACKEND.name_to_label[name.lower()] >= len(BACKEND.interest_labels):
            gt_name_trans.append(name)
            continue
        name = BACKEND.interest_labels[BACKEND.name_to_label[name.lower()]].lower()
        gt_name_trans.append(name)

    dt_name_trans = []
    for name in dt_names:
        if BACKEND.name_to_label[name.lower()] >= len(BACKEND.interest_labels):
            dt_name_trans.append(name)
            continue
        name = BACKEND.interest_labels[BACKEND.name_to_label[name.lower()]].lower()
        dt_name_trans.append(name)

    for name_idx in range(len(gt_names)):
        gt_names[name_idx] = gt_names[name_idx] + "/gt"

    for name_idx in range(len(dt_names)):
        dt_names[name_idx] = dt_names[name_idx] + "/dt"
    response["gt_name_trans"] = gt_name_trans
    response["dt_name_trans"] = dt_name_trans

    response['missing_boxes'] = []
    if BACKEND.eval_path is not None:
        if(timeStamp+".txt") in BACKEND.eval_info:
            response["missing_boxes"] = list([x for x in BACKEND.eval_info[timeStamp+".txt"]["missing_boxes"]])
            print(list([x for x in BACKEND.eval_info[timeStamp+".txt"]["missing_boxes"]]))

    response['fp_boxes'] = []
    if BACKEND.eval_path is not None:
        if (timeStamp+".txt") in BACKEND.eval_info:
            response["fp_boxes"] = list([x for x in BACKEND.eval_info[timeStamp + ".txt"]["fp_boxes"]])
            print(list([x for x in BACKEND.eval_info[timeStamp+".txt"]["fp_boxes"]]))

    response["dt_occlusion"] = []

    if (timeStamp+".txt") in BACKEND.occluded_info:
        gt_occlusion = BACKEND.occluded_info[timeStamp+".txt"]
    else:
        gt_occlusion = []

    for occl_idx in range(len(gt_occlusion)):
        if gt_occlusion[occl_idx] == 0:
            gt_occlusion[occl_idx] = "True"
        elif gt_occlusion[occl_idx] == 1:
            gt_occlusion[occl_idx] = "False"

    response["gt_occlusion"] = gt_occlusion

    response = jsonify(results=[response])
    response.headers['Access-Control-Allow-Headers'] = '*'
    return response


def main(port=16666):
    app.run(host='127.0.0.1', threaded=True, port=port)

if __name__ == '__main__':
    fire.Fire()
