import sys
sys.path.append("..")
import os
from tqdm import tqdm
import cv2
import json
import numpy as np
from prediction_data_manager.prediction_data_manager import PredictionDataManager
from prediction_visualization.scene_render import SceneRender
from functools import partial
from core.calib_trans import pointsMapToSensing
from google.protobuf import text_format
from protos import prediction_eval_path_pb2
from prediction_visualization.gt_visualizer import loadPredictConfig

def readPredict(pre_path, timeStamp, predict_len, has_priority):
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
    pre_priorities = []
    pre_trajectories = []
    for obj_idx, obj in enumerate(objects["objects"]):
        trajectory = np.array(obj["prediction"]).reshape(-1, predict_len, 2)
        pre_trajectories.append(trajectory[0, :, :])
        pre_ids.append(obj["id"])
        pre_names.append(obj['type'].lower())
        pre_loc.append([obj["position"]["x"], obj["position"]["y"], obj["position"]["z"]])
        pre_dims.append([obj["bounding_box"]["length"], obj["bounding_box"]["width"], obj["bounding_box"]["height"]])
        pre_yaws.append([0.0, 0.0, -obj["heading"]])
        if has_priority:
            pre_priorities.append(obj["priority"])

    return (pre_loc, pre_dims, pre_yaws, pre_names, pre_ids, pre_trajectories, pre_priorities)

def getTransformerMapToSensing(position, heading):
    pose = (position, [0, 0, -heading])
    return partial(pointsMapToSensing, pose)

def hasObjects(gt_match_id, attribute, show_mode):
    if show_mode == 0 and len(gt_match_id) > 0:
        return gt_match_id

    if show_mode == 1 or show_mode == 2 or show_mode == 3 or show_mode == 6:
        ids = []
        for id in gt_match_id:
            if id in attribute and attribute[id] == 1:
                ids.append(id)
        return ids

    if show_mode == 4:
        ids = []
        for id in gt_match_id:
            if id in attribute and attribute[id] == -1:
                ids.append(id)
        return ids

    if show_mode == 5:
        ids = []
        for id in gt_match_id:
            if id in attribute and attribute[id] == 0:
                ids.append(id)
        return ids

    return gt_match_id

def perform_visualizer(frame_id, predictionDataManager, sceneRender, selected, infos, care_attribute, \
                       pre_path, save_path, show_mode, save_img, has_priority):

    timestamp = predictionDataManager._timestamps[frame_id]
    pose_info = predictionDataManager._pose_buffer[timestamp]
    ego_position, ego_rotation  = pose_info
    ego_heading = -ego_rotation[2]

    # load info
    info = infos[timestamp]
    gt_match_id = info[0]
    pre_match_id = info[1]
    gt_pre_dict = info[2]
    if care_attribute != None:
        attribute = care_attribute[timestamp]
    else:
        attribute = []
    gt_ids = hasObjects(gt_match_id, attribute, show_mode)
    if len(gt_ids) <= 0:
        return [], False

    # load pre
    pre_timestamp = ''.join(list(str(timestamp)))[0:-6] + "_" + ''.join(list(str(timestamp)))[-6:] + ".txt"
    pre_objects = readPredict(pre_path, pre_timestamp, len(selected), has_priority)

    transformer = getTransformerMapToSensing(ego_position, ego_heading)
    map_layers, flag = sceneRender.render(pre_objects, gt_ids, gt_pre_dict,\
                                    predictionDataManager, timestamp, ego_position, ego_heading, \
                                    transformer, has_priority)
    if flag == False:
        return [], False

    rgb_image = sceneRender.MapLayersToImage(map_layers)
    if save_img:
        anchor_png = save_path + "/" + str(frame_id) + ".png"
        cv2.imwrite(anchor_png, rgb_image)
    return rgb_image, True

def loadAttribute(predictionDataManager, show_mode):
    attribute = None
    if show_mode == 0:
        return attribute
    elif show_mode == 1:
        attribute = predictionDataManager._gt_cut_in
    elif show_mode == 2:
        attribute = predictionDataManager._gt_turn
    elif show_mode == 3 or show_mode == 4:
        attribute = predictionDataManager._gt_accelerate
    elif show_mode == 5 or show_mode == 6:
        attribute = predictionDataManager._gt_on_lane
    else:
        print("no this mode, please check!")
        raise
    return attribute

def visualizer(gt_path, pre_path, pcd_path, map_path, gt_config_path, trajectory_file, velocity_file,
               accelerate_file, turn_file, cut_in_file, on_lane_file, config_path,
               eval_path, save_path, show_mode, show_time, show_error, save_img, has_priority):
    # load config
    predict_config = loadPredictConfig(config_path)
    selected = predict_config[4]

    # load gt
    predictionDataManager = PredictionDataManager(gt_path, pcd_path, map_path, gt_config_path,
                                                  trajectory_file, velocity_file, accelerate_file,
                                                  turn_file, cut_in_file, on_lane_file, predict_config[-1])
    sceneRender = SceneRender(map_path, predict_config[-1], show_time, show_error)

    # load attribute
    care_attribute = loadAttribute(predictionDataManager, show_mode)

    # load infos
    infos = predictionDataManager.loadPickleFile(os.path.join(eval_path, "information.pkl"))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    length = len(predictionDataManager._timestamps)
    for frame_id in tqdm(range(length)):
        rgb_image, flag = perform_visualizer(frame_id, predictionDataManager, sceneRender, selected, infos, \
                                       care_attribute, pre_path, save_path, show_mode, save_img, has_priority)
        if not flag:
            continue
        cv2.imshow("groundtruth", rgb_image)
        if (int(cv2.waitKey(1)) == 27):
            cv2.destroyAllWindows()
            break

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
    save_path = config.prediction_eval_path_reader.save_path


    has_priority = config.prediction_eval_path_reader.has_priority
    show_mode = config.prediction_eval_path_reader.show_mode
    show_time = config.prediction_eval_path_reader.show_time
    show_error = config.prediction_eval_path_reader.show_error
    save_img = config.prediction_eval_path_reader.save_img

    visualizer(gt_path, pre_path, pcd_path, map_path, gt_config_path, trajectory_file, \
               velocity_file, accelerate_file, turn_file, cut_in_file, on_lane_file,
               config_path, eval_path, save_path, show_mode, show_time, show_error, save_img, has_priority)

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
    config_path = "/home/jiamiaoxu/code/detection_evaluation/configs/prediction.eval.config"
    eval_path = "/home/jiamiaoxu/code/detection_evaluation/evaluations/eval_prediction"
    save_path = "/home/jiamiaoxu/code/gt_dt_images"

    # 0 show all objects
    # 1 show cut-in objects
    # 2 show turn objects
    # 3 show accelerate objects
    # 4 show decelerate objects
    # 5 show off-lane objects
    # 6 show on-lane objects
    show_mode = 1

    show_time = 3 # 3 second
    show_error = 0 # show objects that have > 3 error in show_time second
    has_priority = False
    save_img = True

    visualizer(gt_path, pre_path, pcd_path, map_path, gt_config_path, trajectory_file, \
               velocity_file, accelerate_file, turn_file, cut_in_file, on_lane_file,
               config_path, eval_path, save_path, show_mode, show_time, show_error, save_img, has_priority)
    
    """


