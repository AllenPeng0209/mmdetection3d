import sys

sys.path.append('..')
import base64
import pcl
from flask import Flask, jsonify, request
from flask_cors import CORS
import pickle
from pathlib import Path
import fire
from core.calib_trans import *
from protos import tracking_eval_pb2
from google.protobuf import text_format
from core.read_info import get_timeStamps, readPoseConfigFile

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
        self.translation_v2s = None
        self.evaluate_interval = None
        # self.sequence = "0000"
        self.interest_labels = None
        self.name_to_label = None
        self.distance_thresh = None
        self.fp = None
        self.missing = None
        self.ids = None
        self.error_ids = []
        self.error_timeStamp = []
        self.error_dist = []
        self.show_mode = True
        self.colorList = {}


BACKEND = SecondBackend()

def readInfo(info_path, timeStamp, tracker=True):
    timeStamp = timeStamp + ".txt"

    try:
        with open(os.path.join(info_path, timeStamp)) as f:
            objects = json.load(f)
    except:
        objects = {}
        objects["objects"] = []

    info_names = []
    info_loc = []
    info_dims = []
    info_yaws = []
    info_ids = []
    info_velocities = []
    for obj in objects["objects"]:
        info_names.append(obj['type'].lower())
        info_loc.append([obj["position"]["x"], obj["position"]["y"], obj["position"]["z"]])
        info_dims.append([obj["bounding_box"]["length"], obj["bounding_box"]["width"], obj["bounding_box"]["height"]])
        info_yaws.append([0.0, 0.0, obj["heading"]])
        info_ids.append(obj["id"])
        if tracker:
            if "velocity" in obj:
                info_velocities.append([obj["velocity"]["x"], obj["velocity"]["y"], obj["velocity"]["z"]])
            else:
                info_velocities.append([0.0, 0.0, 0.0])

    if tracker:
        return info_loc, info_dims, info_yaws, info_names, info_ids, info_velocities
    else:
        return info_loc, info_dims, info_yaws, info_names, info_ids

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
    gt_path = Path(instance["gt_path"])
    det_path = Path(instance["det_path"])
    pcd_path = Path(instance["pcd_path"])
    eval_path = Path(instance["eval_path"])
    config_path = instance["config_path"]
    GtVelocityFile = Path(instance["GtVelocityFile"])
    show_mode = instance["show_mode"]
    BACKEND.lidar_config_file = Path(instance["lidarConfigFile"])

    if isinstance(config_path, str):
        config = tracking_eval_pb2.DeeprouteTrackingEvalConfig()
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

    BACKEND.gt_path = gt_path
    BACKEND.det_path = det_path
    BACKEND.pcd_path = pcd_path
    BACKEND.eval_path = eval_path
    BACKEND.PCformat = instance["pc_format"]
    BACKEND.interest_labels = interest_labels
    BACKEND.name_to_label = name_to_label
    BACKEND.distance_thresh = distance_thresh
    BACKEND.show_mode = show_mode

    with open(GtVelocityFile, "rb") as f:
        BACKEND.gt_velocities = pickle.load(f)

    pose_config_file = Path(instance["pose_config_file"])
    BACKEND.pose_infos = readPoseConfigFile(pose_config_file)

    response = {"status": "normal"}

    BACKEND.timeStamps = get_timeStamps(gt_path)
    response["timeStamps"] = BACKEND.timeStamps
    focus_range = instance["focus_range"]

    if show_mode:
        if instance["removeOccluded"]:
            endName = "_rm_occluded.pkl"
        else:
            endName = ".pkl"
        type = instance["type"]
        if type == "total":
            if focus_range == "full":
                filename_missing = eval_path / Path('total_missing_full_range' + endName)
                filename_fp = eval_path / Path('total_fp_full_range' + endName)
                filename_ids = eval_path / Path('total_ids_full_range' + endName)
            else:
                pklname = Path("total_missing_within_" + str(focus_range) + endName)
                filename_missing = eval_path / pklname
                pklname = Path("total_fp_within_" + str(focus_range) + endName)
                filename_fp = eval_path / pklname
                pklname = Path("total_ids_within_" + str(focus_range) + endName)
                filename_ids = eval_path / pklname

            with open(filename_missing, "rb") as f:
                BACKEND.missing = pickle.load(f)
            with open(filename_fp, "rb") as f:
                BACKEND.fp = pickle.load(f)
            with open(filename_ids, "rb") as f:
                BACKEND.ids = pickle.load(f)

            ids_timestamp = []
            for key in BACKEND.ids.keys():
                ids_timestamp.append(key.split(".")[0])
                a = list(str(int(key.split(".")[0]) - 100000))
                a.insert(-6, "_")
                b = "".join(a)
                ids_timestamp.append(b)

            ids_timestamp.sort()

        else:
            eval_path = os.path.join(eval_path, type)
            if focus_range == "full":
                filename_missing = eval_path / Path('missing_full_range' + endName)
                filename_fp = eval_path / Path('fp_full_range' + endName)
                filename_ids = eval_path / Path('ids_full_range' + endName)
            else:
                pklname = Path("missing_within_" + str(focus_range) + endName)
                filename_missing = eval_path / pklname
                pklname = Path("fp_within_" + str(focus_range) + endName)
                filename_fp = eval_path / pklname
                pklname = Path("ids_within_" + str(focus_range) + endName)
                filename_ids = eval_path / pklname

            with open(filename_missing, "rb") as f:
                BACKEND.missing = pickle.load(f)
            with open(filename_fp, "rb") as f:
                BACKEND.fp = pickle.load(f)
            with open(filename_ids, "rb") as f:
                BACKEND.ids = pickle.load(f)

            ids_timestamp = []
            for key in BACKEND.ids.keys():
                ids_timestamp.append(key.split(".")[0])
                a = list(str(int(key.split(".")[0]) - 100000))
                a.insert(-6, "_")
                b = "".join(a)
                ids_timestamp.append(b)

            ids_timestamp.sort()
    else:
        ids_timestamp = []

    response["id_switches"] = ids_timestamp

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

    path = os.path.join(BACKEND.pcd_path, timeStamp + ".pcd")

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

    gt_velocity = BACKEND.gt_velocities[timeStamp + ".txt"]
    gt_loc, gt_dims, gt_yaws, gt_names, gt_ids = readInfo(BACKEND.gt_path, timeStamp, False)
    pre_loc, pre_dims, pre_yaws, pre_names, dt_ids, dt_velocity = readInfo(BACKEND.det_path, timeStamp, True)

    gt_velocity = velocityglobal2local(gt_velocity, gt_loc, gt_ids, timeStamp, BACKEND.pose_infos,
                                       BACKEND.lidar_config_file, tracker=False)
    dt_velocity = velocityglobal2local(dt_velocity, pre_loc, dt_ids, timeStamp, BACKEND.pose_infos,
                                       BACKEND.lidar_config_file, tracker=True)

    response["gt_velocity"] = gt_velocity
    response["dt_velocity"] = dt_velocity

    if BACKEND.show_mode:
        missing = BACKEND.missing[timeStamp + ".txt"]
        fp = BACKEND.fp[timeStamp + ".txt"]
        ids = []
        if timeStamp + ".txt" in BACKEND.ids:
            ids = BACKEND.ids[timeStamp + ".txt"]
    else:
        missing = []
        fp = []
        ids = []

    response["missing"] = missing
    response["fp"] = fp
    response["ids"] = ids

    response["gt_loc"] = gt_loc
    response["gt_dims"] = gt_dims
    response["gt_yaws"] = gt_yaws
    response["gt_names"] = gt_names
    response["gt_ids"] = gt_ids

    response["dt_loc"] = pre_loc
    response["dt_dims"] = pre_dims
    response["dt_yaws"] = pre_yaws
    response["dt_names"] = pre_names
    response["dt_ids"] = dt_ids

    response["distance"] = BACKEND.distance_thresh

    dt_colorList = []
    for id in dt_ids:
        if id in BACKEND.colorList:
            dt_colorList.append(BACKEND.colorList[id])
        else:
            color = hex(np.random.randint(0xffffff))
            color = "#" + color[2:]
            BACKEND.colorList[id] = color
            dt_colorList.append(color)

    response["color_list"] = dt_colorList

    gt_name_trans = []
    for name in gt_names:
        if BACKEND.name_to_label[name.lower()] >= len(BACKEND.interest_labels):
            gt_name_trans.append(name)
            continue
        name = BACKEND.interest_labels[BACKEND.name_to_label[name.lower()]].lower()
        gt_name_trans.append(name)

    dt_name_trans = []
    for name in pre_names:
        if BACKEND.name_to_label[name.lower()] >= len(BACKEND.interest_labels):
            dt_name_trans.append(name)
            continue
        name = BACKEND.interest_labels[BACKEND.name_to_label[name.lower()]].lower()
        dt_name_trans.append(name)

    response["gt_name_trans"] = gt_name_trans
    response["dt_name_trans"] = dt_name_trans

    response = jsonify(results=[response])
    response.headers['Access-Control-Allow-Headers'] = '*'
    return response


def main(port=16666):
    app.run(host='127.0.0.1', threaded=True, port=port)


if __name__ == '__main__':
    fire.Fire()
