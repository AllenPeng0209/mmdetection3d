import sys
sys.path.append('..')
import base64
import pcl
from flask import Flask, jsonify, request
from flask_cors import CORS
from pathlib import Path
import fire
from core.calib_trans import *
from core.read_info import *
from prediction_data_manager.prediction_data_manager import PredictionDataManager
app = Flask("second")
CORS(app)

from protos import prediction_eval_pb2
from core.calib_trans import pointsMapToSensing_v2
from prediction_data_manager.prediction_data_manager import PredictionDataManager
from google.protobuf import text_format

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

class SecondBackend:
    def __init__(self):
        self.PCformat = True
        self.gtPath = None
        self.prePath = None
        self.pcdPath = None
        self.mapPath = None
        self.trajectoryFile = None
        self.velocityFile = None
        self.accelerateFile = None
        self.turnFile = None
        self.cutInFile = None
        self.onLaneFile = None
        self.gtConfigPath = None
        self.evalPath = None
        self.configPath = None
        self.behavior = None
        self.predict_config = None
        self.gtDataManager = None
        self.attribute = None
        self.infos = None

BACKEND = SecondBackend()

def readPredict(pre_path, timeStamp, predict_len):
    try:
        with open(os.path.join(pre_path, timeStamp), "rb") as f:
            objects = json.load(f)
    except:
        objects = {}
        objects["objects"] = []

    pre_names = []
    pre_loc = []
    pre_dims = []
    pre_yaws = []
    pre_ids = []
    pre_trajectories = []
    for obj_idx, obj in enumerate(objects["objects"]):
        trajectory = np.array(obj["prediction"]).reshape(-1, predict_len, 2)
        pre_trajectories.append(trajectory[0, :, :])
        pre_ids.append(obj["id"])
        pre_names.append(obj['type'].lower())
        pre_loc.append([obj["position"]["x"], obj["position"]["y"], obj["position"]["z"]])
        pre_dims.append([obj["bounding_box"]["length"], obj["bounding_box"]["width"], obj["bounding_box"]["height"]])
        pre_yaws.append([0.0, 0.0, obj["heading"]])

    return pre_loc, pre_dims, pre_yaws, pre_names, pre_ids, pre_trajectories

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
    response = {"status": "normal"}

    # assign
    BACKEND.gtPath = instance["gtPath"]
    BACKEND.prePath = instance["prePath"]
    BACKEND.pcdPath = instance["pcdPath"]
    BACKEND.mapPath = instance["mapPath"]
    BACKEND.trajectoryFile = instance["trajectoryFile"]
    BACKEND.velocityFile = instance["velocityFile"]
    BACKEND.accelerateFile = instance["accelerateFile"]
    BACKEND.turnFile = instance["turnFile"]
    BACKEND.cutInFile = instance["cutInFile"]
    BACKEND.onLaneFile = instance["onLaneFile"]
    BACKEND.gtConfigPath = instance["gtConfigPath"]
    BACKEND.evalPath = instance["evalPath"]
    BACKEND.configPath = instance["configPath"]
    BACKEND.behavior = instance["behavior"]

    # load config
    BACKEND.predict_config = loadPredictConfig(BACKEND.configPath)

    # load gt
    BACKEND.gtDataManager = PredictionDataManager(BACKEND.gtPath, BACKEND.pcdPath, BACKEND.mapPath, BACKEND.gtConfigPath,
                                                  BACKEND.trajectoryFile, BACKEND.velocityFile, BACKEND.accelerateFile,
                                                  BACKEND.turnFile, BACKEND.cutInFile, BACKEND.onLaneFile, BACKEND.predict_config[-1])

    if BACKEND.behavior == "turn":
        BACKEND.attribute = BACKEND.gtDataManager.loadPickleFile(BACKEND.turnFile)
    elif BACKEND.behavior == "accelerate" or BACKEND.behavior == "decelerate":
        BACKEND.attribute = BACKEND.gtDataManager.loadPickleFile(BACKEND.accelerateFile)
    elif BACKEND.behavior == "cut_in":
        BACKEND.attribute = BACKEND.gtDataManager.loadPickleFile(BACKEND.cutInFile)
    elif BACKEND.behavior == "off_lane" or BACKEND.behavior == "on_lane":
        BACKEND.attribute = BACKEND.gtDataManager.loadPickleFile(BACKEND.onLaneFile)

    BACKEND.infos = BACKEND.gtDataManager.loadPickleFile(os.path.join(BACKEND.evalPath, "information.pkl"))

    if BACKEND.attribute == None:
        response["timeStamps"] = BACKEND.gtDataManager._timestamps

    elif BACKEND.behavior == "turn" or BACKEND.behavior == "cut_in" or \
            BACKEND.behavior == "accelerate" or BACKEND.behavior == "on_lane":
        timestamps = []
        for timestamp in BACKEND.attribute.keys():
            for id in BACKEND.attribute[timestamp].keys():
                if BACKEND.attribute[timestamp][id] == 1:
                    timestamps.append(timestamp)
                    break
        response["timeStamps"] = timestamps

    elif BACKEND.behavior == "decelerate":
        timestamps = []
        for timestamp in BACKEND.attribute.keys():
            for id in BACKEND.attribute[timestamp].keys():
                if BACKEND.attribute[timestamp][id] == -1:
                    timestamps.append(timestamp)
                    break
        response["timeStamps"] = timestamps

    elif BACKEND.behavior == "off_lane":
        timestamps = []
        for timestamp in BACKEND.attribute.keys():
            for id in BACKEND.attribute[timestamp].keys():
                if BACKEND.attribute[timestamp][id] == 0:
                    timestamps.append(timestamp)
                    break
        response["timeStamps"] = timestamps

    response["start_index"] = 10
    response = jsonify(results=[response])
    response.headers['Access-Control-Allow-Headers'] = '*'
    return response

@app.route('/api/get_pointcloud', methods=['POST'])
def get_pointcloud():
    global BACKEND
    instance = request.json
    response = {"status": "normal"}
    if BACKEND.pcdPath is None:
        return error_response("pcd path is not set")

    timestamp = instance["timeStamp"]

    # load pcd
    points = BACKEND.gtDataManager.getPcd(timestamp)
    response["num_features"] = points.shape[1]
    enable_int16 = instance["enable_int16"]
    if enable_int16:
        int16_factor = instance["int16_factor"]
        points *= int16_factor
        points = points.astype(np.int16)
    pc_str = base64.b64encode(points.tobytes())
    response["pointcloud"] = pc_str.decode("utf-8")

    gt_match_id = BACKEND.infos[timestamp][0]
    pre_match_id = BACKEND.infos[timestamp][1]
    gt_pre_dict = BACKEND.infos[timestamp][2]

    # config
    stop_thresh = float(BACKEND.predict_config[-1].deeproute_eval_input_reader.stop_thresh)
    frequency = int(BACKEND.predict_config[-1].deeproute_eval_input_reader.frequency)
    eval_frequency = int(BACKEND.predict_config[-1].deeproute_eval_input_reader.eval_frequency)
    predict_time = int(BACKEND.predict_config[-1].deeproute_eval_input_reader.predict_time)
    internal_gap = int(frequency / eval_frequency)
    future_length = predict_time * frequency + 1
    selected = list(range(future_length - 1, 0, -internal_gap))
    selected.sort()

    # load gt and pre
    pre_timestamp = ''.join(list(str(timestamp)))[0:-6] + "_" + ''.join(list(str(timestamp)))[-6:] + ".txt"
    loc2, dims2, yaws2, names2, ids2, trajectories2 = \
        readPredict(BACKEND.prePath, pre_timestamp, len(selected))

    corners, boxes, names, ids, yaws = BACKEND.gtDataManager.getObjectInfos(timestamp)
    loc = boxes[:, :3].tolist()
    dims = boxes[:, 3:6].tolist()
    gt_loc, pre_loc = [], []
    gt_dims, pre_dims = [], []
    gt_yaws, pre_yaws = [], []
    gt_names, pre_names = [], []
    gt_trajectories, pre_trajectories = [], []
    for idx, id in enumerate(ids):
        if id in gt_match_id:
            if BACKEND.attribute == None or \
              (BACKEND.behavior == "turn" and id in BACKEND.attribute[timestamp] and BACKEND.attribute[timestamp][id] == 1) or\
              (BACKEND.behavior == "accelerate" and id in BACKEND.attribute[timestamp] and BACKEND.attribute[timestamp][id] == 1) or\
              (BACKEND.behavior == "decelerate" and id in BACKEND.attribute[timestamp] and BACKEND.attribute[timestamp][id] == -1) or\
              (BACKEND.behavior == "cut_in" and id in BACKEND.attribute[timestamp] and BACKEND.attribute[timestamp][id] == 1) or\
              (BACKEND.behavior == "off_lane" and id in BACKEND.attribute[timestamp] and BACKEND.attribute[timestamp][id] == 0) or\
              (BACKEND.behavior == "on_lane" and id in BACKEND.attribute[timestamp] and BACKEND.attribute[timestamp][id] == 1):
                gt_trajectory = BACKEND.gtDataManager.getFutureTrajectory(id, timestamp, future_length)
                gt_trajectory = np.array(gt_trajectory)[:, 1:4]
                if gt_trajectory.shape[0] == future_length:
                    gt_trajectory = gt_trajectory[selected, :]  # global coord
                elif gt_trajectory.shape[0] > 5:
                    part_selected = list(range(5, gt_trajectory.shape[0], internal_gap))
                    gt_trajectory = gt_trajectory[part_selected, :]
                else:
                    continue
                pose_info = BACKEND.gtDataManager.getPose(timestamp)
                config_lidars = BACKEND.gtDataManager._config_lidars
                gt_trajectory = pointsMapToSensing_v2(pose_info, gt_trajectory, config_lidars)[:, :2]  # local coord

                gt_loc.append(loc[idx])
                gt_dims.append(dims[idx])
                gt_yaws.append([0, 0, yaws[idx].tolist()])
                gt_names.append(names[idx])
                gt_trajectory = gt_trajectory.tolist()
                gt_trajectory.insert(0, [loc[idx][0], loc[idx][1]])
                gt_trajectories.append(gt_trajectory)


                pre_id = gt_pre_dict[id]
                pre_idx = ids2.index(pre_id)
                pre_loc.append(loc2[pre_idx])
                pre_dims.append(dims2[pre_idx])
                pre_yaws.append(yaws2[pre_idx])
                pre_names.append(names2[pre_idx])
                pre_trajectory = trajectories2[pre_idx].tolist()
                pre_trajectory.insert(0, [loc2[pre_idx][0], loc2[pre_idx][1]])
                pre_trajectories.append(pre_trajectory)

    for name_idx in range(len(gt_names)):
        gt_names[name_idx] = str(gt_names[name_idx]) + "/gt"
    response["gt_loc"] = gt_loc
    response["gt_dims"] = gt_dims
    response["gt_yaws"] = gt_yaws
    response["gt_names"] = gt_names
    response["gt_trajectories"] = gt_trajectories

    for name_idx in range(len(pre_names)):
        pre_names[name_idx] = pre_names[name_idx] + "/dt"
    response["dt_loc"] = pre_loc
    response["dt_dims"] = pre_dims
    response["dt_yaws"] = pre_yaws
    response["dt_names"] = pre_names
    response["pre_trajectories"] = pre_trajectories

    response = jsonify(results=[response])
    response.headers['Access-Control-Allow-Headers'] = '*'
    return response

def main(port=16666):
    app.run(host='127.0.0.1', threaded=True, port=port)

if __name__ == '__main__':
    fire.Fire()
