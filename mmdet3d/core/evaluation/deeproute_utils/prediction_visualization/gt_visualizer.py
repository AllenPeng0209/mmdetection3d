import sys
sys.path.append("..")
import os
from tqdm import tqdm
import cv2
from prediction_data_manager.prediction_data_manager import PredictionDataManager
from prediction_visualization.gt_scene_render import SceneRender
from functools import partial
from core.calib_trans import pointsMapToSensing
from google.protobuf import text_format
from protos import prediction_eval_path_pb2
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

def getTransformerMapToSensing(position, heading):
    pose = (position, [0, 0, -heading])
    return partial(pointsMapToSensing, pose)

def perform_visualizer(frame_id, predictionDataManager, sceneRender, save_path, show_mode, save_img):
    timestamp = predictionDataManager._timestamps[frame_id]
    pose_info = predictionDataManager._pose_buffer[timestamp]
    ego_position, ego_rotation  = pose_info
    ego_heading = -ego_rotation[2]

    transformer = getTransformerMapToSensing(ego_position, ego_heading)
    map_layers = sceneRender.render(predictionDataManager, timestamp, ego_position, ego_heading, transformer, show_mode)
    rgb_image = sceneRender.MapLayersToImage(map_layers)
    if save_img:
        anchor_png = save_path + "/" + str(frame_id) + ".png"
        cv2.imwrite(anchor_png, rgb_image)
    return rgb_image

def visualizer(gt_path, pcd_path, map_path, gt_config_path, trajectory_file, velocity_file,
               accelerate_file, turn_file, cut_in_file, on_lane_file, config_path, save_path, show_mode, save_img):
    # load config
    predict_config = loadPredictConfig(config_path)

    # load gt
    predictionDataManager = PredictionDataManager(gt_path, pcd_path, map_path, gt_config_path,
                                                  trajectory_file, velocity_file, accelerate_file,
                                                  turn_file, cut_in_file, on_lane_file, predict_config[-1])
    sceneRender = SceneRender(map_path, predict_config[-1])

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    length = len(predictionDataManager._timestamps)
    for frame_id in tqdm(range(length)):
        rgb_image = perform_visualizer(frame_id, predictionDataManager, sceneRender, save_path, show_mode, save_img)
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
    pcd_path = config.prediction_eval_path_reader.pcd_path
    map_path = config.prediction_eval_path_reader.map_path
    trajectory_file = config.prediction_eval_path_reader.trajectory_file
    velocity_file = config.prediction_eval_path_reader.velocity_file
    accelerate_file = config.prediction_eval_path_reader.accelerate_file
    turn_file = config.prediction_eval_path_reader.turn_file
    cut_in_file = config.prediction_eval_path_reader.cut_in_file
    on_lane_file = config.prediction_eval_path_reader.on_lane_file
    gt_config_path = config.prediction_eval_path_reader.gt_config_path
    config_path = config.prediction_eval_path_reader.config_path
    save_path = config.prediction_eval_path_reader.save_path


    show_mode = config.prediction_eval_path_reader.show_mode
    save_img = config.prediction_eval_path_reader.save_img

    visualizer(gt_path, pcd_path, map_path, gt_config_path, trajectory_file, \
               velocity_file, accelerate_file, turn_file, cut_in_file, on_lane_file,
               config_path, save_path, show_mode, save_img)

if __name__ == '__main__':
    params_config_path = "/home/jiamiaoxu/code/detection_evaluation/configs/prediction.eval.path.config"
    main(params_config_path)

    """
    gt_path = "/home/jiamiaoxu/data/code/detection_evaluation/tracking_Jiamiao/groundtruth"
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
    save_path = "/home/jiamiaoxu/code/gt_images"

    show_mode = 3  # 1 for cut-in, 2 for turn, 3 for off-lane
    save_img = True

    visualizer(gt_path, pcd_path, map_path, gt_config_path, trajectory_file, \
               velocity_file, accelerate_file, turn_file, cut_in_file, on_lane_file,
               config_path, save_path, show_mode, save_img)
    """


