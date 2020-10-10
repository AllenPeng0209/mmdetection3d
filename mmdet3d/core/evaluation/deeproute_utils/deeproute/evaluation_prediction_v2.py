import multiprocessing
import sys
sys.path.append("..")
import time
from shapely.geometry.polygon import Polygon
from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from collections import defaultdict
import math
import pickle
import torch
import iou_cuda
from core.calib_trans import *
from core.munkres import Munkres
import core.box_ops as box_ops
from core.read_info import *
from core.occl_dist_assign import angle
from protos import prediction_eval_pb2
from google.protobuf import text_format

def save(info, eval_path):
    filename = os.path.join(eval_path, "information.pkl")
    with open(filename, "wb") as f:
        pickle.dump(info, f)

def compute_results(predict_len, errors):
    data = np.zeros(predict_len)
    size = np.zeros(predict_len)

    for error in errors:
        data[:len(error)] += error
        size[:len(error)] += 1

    mask = np.zeros(predict_len)
    index = np.where(mask == size)
    size[index] = 1
    return (data / size).reshape(1, -1)

def draw_table(eval_path, data, filename, predict_len):
    rows = ["car", "pedestrian", "cyclist", "total"]
    col_tmp = np.round((np.arange(predict_len)+1)*0.5, 1)
    col_tmp = col_tmp.tolist()
    columns = [str(time) + "s" for time in col_tmp]
    the_table = plt.table(cellText=data,
                            colWidths=[0.08] * predict_len,
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

def generate_data(vehicle_error, pedestrian_error, bicycle_error, type, eval_path, weight, predict_len):
    data_car = compute_results(predict_len, vehicle_error)
    data_ped = compute_results(predict_len, pedestrian_error)
    data_bic = compute_results(predict_len, bicycle_error)
    data_total = weight[0] * data_car + weight[1] * data_ped + weight[2] * data_bic

    data = np.concatenate((data_car, data_ped, data_bic, data_total), axis=0)
    data = np.round(data, 2)

    filename = type + "_eval_result.png"
    draw_table(eval_path, data, filename, predict_len)


def MergeTable(eval_path):
    img1 = mpimg.imread(eval_path + "/L1_eval_result.png")
    img2 = mpimg.imread(eval_path + "/L2_eval_result.png")
    img3 = mpimg.imread(eval_path + "/L2_along_eval_result.png")
    img4 = mpimg.imread(eval_path + "/L2_across_eval_result.png")
    img = np.concatenate((img1, img2, img3, img4), axis = 0)
    plt.imsave(eval_path + "/total_eval_result.png", img)
    os.remove(eval_path + "/L1_eval_result.png")
    os.remove(eval_path + "/L2_eval_result.png")
    os.remove(eval_path + "/L2_along_eval_result.png")
    os.remove(eval_path + "/L2_across_eval_result.png")

def get_gt_pos_per_id(gt_all_pos, gt_all_ids):
    gt_pos_per_id = defaultdict(list)

    for i in range(0, len(gt_all_ids)):
        for j in range(len(gt_all_ids[i])):
            gt_pos_per_id[gt_all_ids[i][j]].append(gt_all_pos[i][j])

    return gt_pos_per_id

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
    gt_match_pos = []
    pre_match_pos = []
    for row, col in association_matrix:
        # apply gating on boxoverlap
        c = cost_matrix[row][col]
        if c < max_cost:
            gt_match_pos.append(row)
            pre_match_pos.append(col)
            match_pairs.update({col:row})

    return match_pairs, gt_match_pos, pre_match_pos

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

def readPredict(pre_path, pre_text_names, index, predict_len, remove_pre_condition):
    try:
        with open(os.path.join(pre_path, pre_text_names[index]), "rb") as f:
            objects = json.load(f)
    except:
        objects = {}
        objects["objects"] = []

    pre_names = []
    pre_loc = []
    pre_dims = []
    pre_yaws = []
    pre_all_pos = []
    for obj_idx, obj in enumerate(objects["objects"]):
        pre_pos = []
        if obj_idx == 0 and predict_len * 2 > len(obj["prediction"]):
            print("The predict_len that you choose is invalid!")
        for i in range(0, predict_len * 2, 2):
            pre_pos.append(np.array([obj["prediction"][i], obj["prediction"][i + 1]]))

        # if np.sqrt(np.sum((pre_pos[0] - pre_pos[1])**2)) < remove_pre_condition:
        #    continue

        pre_all_pos.append(pre_pos)
        pre_names.append(obj['type'].lower())
        pre_loc.append([obj["position"]["x"], obj["position"]["y"], obj["position"]["z"]])
        pre_dims.append([obj["bounding_box"]["length"], obj["bounding_box"]["width"], obj["bounding_box"]["height"]])
        pre_yaws.append(obj["heading"])

    pre_loc = np.array(pre_loc).reshape(-1, 3)
    pre_dims = np.array(pre_dims).reshape(-1, 3)
    pre_yaws = np.array(pre_yaws)
    pre_boxes = np.concatenate([pre_loc, pre_dims, pre_yaws[..., np.newaxis]], axis=1)
    box_ops.change_box3d_center_(pre_boxes, src=[0.5, 0.5, 0], dst=[0.5, 0.5, 0.5])
    pre_corners = box_ops.get_corners(pre_boxes[:, :2], pre_boxes[:, 3:5], pre_boxes[:, 6])

    return pre_all_pos, pre_boxes, pre_corners, pre_names

def readGT(gt_path, gt_text_names, index):
    try:
        with open(os.path.join(gt_path, gt_text_names[index])) as f:
            objects = json.load(f)
    except:
        objects = {}
        objects["objects"] = []

    gt_names = []
    gt_loc = []
    gt_dims = []
    gt_yaws = []
    gt_ids = []
    for obj in objects["objects"]:
        if obj["id"] in gt_ids:
            print("there is the same id: ", obj["id"], " in this timestamp", gt_text_names[index], ". please check!")
            continue

        gt_names.append(obj['type'].lower())
        gt_loc.append([obj["position"]["x"], obj["position"]["y"], obj["position"]["z"]])
        gt_dims.append([obj["bounding_box"]["length"], obj["bounding_box"]["width"], obj["bounding_box"]["height"]])
        gt_yaws.append(obj["heading"])
        gt_ids.append(obj["id"])

    gt_loc = np.array(gt_loc).reshape(-1, 3)
    gt_dims = np.array(gt_dims).reshape(-1, 3)
    gt_yaws = np.array(gt_yaws)
    gt_boxes = np.concatenate([gt_loc, gt_dims, gt_yaws[..., np.newaxis]], axis=1)
    box_ops.change_box3d_center_(gt_boxes, src=[0.5, 0.5, 0], dst=[0.5, 0.5, 0.5])
    gt_corners = box_ops.get_corners(gt_boxes[:, :2], gt_boxes[:, 3:5], gt_boxes[:, 6])
    gt_pos = gt_boxes[:,:3]

    return gt_pos, gt_boxes, gt_corners, gt_names, gt_ids, gt_yaws

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

def assign(vehicle_error, pedestrian_error, bicycle_error, error, cls, name_to_label):
    if len(error) != 0:
        error = np.array(error)
        if name_to_label[cls] == 0:
            vehicle_error.append(error)
        elif name_to_label[cls] == 1:
            pedestrian_error.append(error)
        elif name_to_label[cls] == 2:
            bicycle_error.append(error)

def loadConfig(config_path):
    if isinstance(config_path, str):
        config = prediction_eval_pb2.DeeprouteDetectionEvalConfig()
        with open(config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, config)
    else:
        config = config_path

    interest_labels = config.deeproute_eval_input_reader.interest_labels
    weight = config.deeproute_eval_input_reader.weight
    overlap_thresh = config.deeproute_eval_input_reader.overlap_thresh
    gt_frameFrequency = config.deeproute_eval_input_reader.gt_frameFrequency
    predict_frameFrequency = config.deeproute_eval_input_reader.predict_frameFrequency
    training_time = config.deeproute_eval_input_reader.training_time
    predict_len = config.deeproute_eval_input_reader.predict_len
    cal_iou_bev = config.deeproute_eval_input_reader.cal_iou_bev
    remove_pre_condition = config.deeproute_eval_input_reader.remove_pre_condition

    name_to_label = {}
    for sample_group in config.deeproute_eval_input_reader.name_to_label.sample_groups:
        name_to_label.update(dict(sample_group.name_to_num))

    return interest_labels, weight, overlap_thresh, gt_frameFrequency, predict_frameFrequency, \
           training_time, predict_len, cal_iou_bev, remove_pre_condition, name_to_label

def sub_compute_evaluate(gt_all_pos, gt_all_boxes, gt_all_corners, gt_all_names,
             gt_all_ids, pre_all_pos, pre_boxes, pre_corners, pre_names, timeStamp, cutIn):

    gt_all_pos = local2global2local(gt_all_pos, timeStamp, gt_text_names, pose_infos, evaluate_interval)
    gt_object_num = len(gt_all_names[0])
    dt_object_num = len(pre_names)

    # get occlusion and distance attribute of each object in gt
    iou = get_iou_bev(gt_object_num, dt_object_num, gt_all_corners[0], pre_corners) if cal_iou_bev  else get_iou_3D(gt_object_num, dt_object_num, gt_all_boxes[0], pre_boxes, gt_all_corners[0], pre_corners)
    match_pairs, gt_match_pos, pre_match_pos = get_match_pair(gt_object_num, dt_object_num, gt_all_names[0], pre_names, iou, overlap_thresh, name_to_label)

    # calculate error
    gt_pos_per_id = get_gt_pos_per_id(gt_all_pos[1:], gt_all_ids[1:])

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
    for obj_idx in (pre_match_pos):

        target_id = gt_all_ids[0][match_pairs[obj_idx]]
        gt_pos = gt_pos_per_id[target_id]

        if cutIn:
            if gt_cut_in_attribute[timeStamp][target_id] == 0:
                continue

        try:
            L1_error = [l1_error(gt_pos[i], pre_all_pos[obj_idx][i-1]) for i in range(1, len(gt_pos))]
        except:
            print(len(gt_pos))
            print("timeStamp: ", timeStamp, "target_id: ", target_id)
        L2_error = [l2_error(gt_pos[i], pre_all_pos[obj_idx][i-1]) for i in range(1, len(gt_pos))]

        across_error = []
        along_error = []
        for i in range(1, len(gt_pos)):
            res = l2_across_along_error(gt_pos[i-1], gt_pos[i], pre_all_pos[obj_idx][i-1])
            across_error.append(res[0])
            along_error.append(res[1])

        cls = gt_all_names[0][match_pairs[obj_idx]]

        # save information for visualization
        gt_pos_tmp.append(np.array(gt_pos).tolist())
        tmp = np.concatenate([pre_boxes[obj_idx][:2].reshape(1,2), np.array(pre_all_pos[obj_idx][:len(gt_pos)])], axis = 0)
        pre_pos_tmp.append(tmp.tolist())

        assign(vehicle_L1_error, pedestrian_L1_error, bicycle_L1_error, L1_error, cls, name_to_label)
        assign(vehicle_L2_error, pedestrian_L2_error, bicycle_L2_error, L2_error, cls, name_to_label)
        assign(vehicle_across_error, pedestrian_across_error, bicycle_across_error, across_error, cls, name_to_label)
        assign(vehicle_along_error, pedestrian_along_error, bicycle_along_error, along_error, cls, name_to_label)


    return (timeStamp, gt_match_pos, pre_match_pos, gt_pos_tmp, pre_pos_tmp,
            vehicle_L1_error, pedestrian_L1_error, bicycle_L1_error,
            vehicle_L2_error, pedestrian_L2_error, bicycle_L2_error,
            vehicle_across_error, pedestrian_across_error, bicycle_across_error,
            vehicle_along_error, pedestrian_along_error, bicycle_along_error)


def compute_evaluate(gt_path, pre_path, eval_path, config_path, pose_config_file, cut_in_file, pool_num = 10, cutIn=False):
    global interest_labels, overlap_thresh, gt_frameFrequency, predict_frameFrequency
    global training_time, predict_len, cal_iou_bev, remove_pre_condition, start_index, evaluate_interval
    global gt_text_names, pose_infos, name_to_label
    global gt_cut_in_attribute

    assert eval_path is not None
    Path(eval_path).mkdir(parents=True, exist_ok=True)

    if cutIn:
        try:
            with open(cut_in_file, "rb") as f:
                gt_cut_in_attribute = pickle.load(f)
        except:
            print("no cut in file, run cut_in_attribute.py first!")
            raise
    else:
        gt_cut_in_attribute = []

    pose_infos = readPoseConfigFile(pose_config_file)

    # loading configuration for prediction
    interest_labels, weight, overlap_thresh, gt_frameFrequency, predict_frameFrequency, \
    training_time, predict_len, cal_iou_bev, remove_pre_condition, name_to_label = loadConfig(config_path)

    start_index = int(training_time * gt_frameFrequency)
    evaluate_interval = int(gt_frameFrequency / predict_frameFrequency)

    # loading all timestamps of gt and prediction
    assert gt_path is not None
    assert pre_path is not None
    pre_text_names = get_all_txt_names(pre_path)
    gt_text_names = get_all_txt_names(gt_path)
    assert len(pre_text_names) == len(gt_text_names)

    gt_frame_num = len(gt_text_names)

    gt_all_pos = []
    gt_all_boxes = []
    gt_all_corners = []
    gt_all_names = []
    gt_all_ids = []
    print("loading gt information...")
    for i in tqdm(range(gt_frame_num)):
        gt_pos, gt_boxes, gt_corners, gt_names, gt_ids, gt_yaws = readGT(gt_path, gt_text_names, i)
        gt_all_pos.append(gt_pos)
        gt_all_boxes.append(gt_boxes)
        gt_all_corners.append(gt_corners)
        gt_all_names.append(gt_names)
        gt_all_ids.append(gt_ids)

    print("start prediction...")
    pool = multiprocessing.Pool(pool_num)
    res = []
    for i in tqdm(range(start_index, gt_frame_num-evaluate_interval-1)):
        # loading prediction results
        timeStamp = pre_text_names[i]
        pre_all_pos, pre_boxes, pre_corners, pre_names = readPredict(pre_path, pre_text_names, i, predict_len, remove_pre_condition)

        # loading gt
        gt_part_pos = [gt_all_pos[i].copy()]
        gt_part_boxes = [gt_all_boxes[i].copy()]
        gt_part_corners = [gt_all_corners[i].copy()]
        gt_part_names = [gt_all_names[i].copy()]
        gt_part_ids = [gt_all_ids[i].copy()]

        for j in range(i, min(i+predict_len*evaluate_interval+1, gt_frame_num-1), evaluate_interval):
            gt_part_pos.append(gt_all_pos[j])
            gt_part_boxes.append(gt_all_boxes[j])
            gt_part_corners.append(gt_all_corners[j])
            gt_part_names.append(gt_all_names[j])
            gt_part_ids.append(gt_all_ids[j])

        #gt_part_pos = local2global2local(gt_part_pos, timeStamp, gt_text_names, pose_infos, evaluate_interval)
        res.append(pool.apply_async(sub_compute_evaluate, (gt_part_pos.copy(), gt_part_boxes.copy(), gt_part_corners.copy(), gt_part_names.copy(), gt_part_ids.copy(), pre_all_pos.copy(), pre_boxes.copy(), pre_corners.copy(), pre_names.copy(), timeStamp, cutIn)))

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
        info[timeStamp].append(res[i].get()[4])
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

    generate_data(vehicle_L1_error, pedestrian_L1_error, bicycle_L1_error, "L1", eval_path, weight, predict_len)
    generate_data(vehicle_L2_error, pedestrian_L2_error, bicycle_L2_error, "L2", eval_path, weight, predict_len)
    generate_data(vehicle_across_error, pedestrian_across_error, bicycle_across_error, "L2_across", eval_path, weight, predict_len)
    generate_data(vehicle_along_error, pedestrian_along_error, bicycle_along_error, "L2_along", eval_path, weight, predict_len)
    save(info, eval_path)

    MergeTable(eval_path)

def evaluation(config_path):
    if isinstance(config_path, str):
        config = prediction_eval_pb2.DeeprouteEvalPathConfig()
        with open(config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, config)
    else:
        config = config_path

    gt_path = config.deeproute_eval_path_reader.gt_path
    pre_path = config.deeproute_eval_path_reader.dt_path
    eval_config = config.deeproute_eval_path_reader.eval_config
    save_path = config.deeproute_eval_path_reader.save_path
    pose_config_file = config.deeproute_eval_path_reader.pose_config_file
    pool_num = config.deeproute_eval_path_reader.pool_num

    compute_evaluate(gt_path, pre_path, save_path, eval_config, pose_config_file, pool_num)
    print("Evaluation Finished!")


if __name__ == '__main__':
    # root_path = "/home/jiamiaoxu/data/code/detection_evaluation/prediction_Jiamiao/"
    # gt_path = os.path.join(root_path, "our_data/20190412_rain_1_old")
    # pre_path = os.path.join(root_path, "our_data/prediction_result")
    # eval_path = os.path.join(root_path, "evaluations/eval_prediction")
    # config_path = os.path.join(root_path, "configs/deeproute.eval.config")
    # pose_config_file = os.path.join(root_path, "configs/pose.csv")
    cut_in_file = "/home/jiamiaoxu/Downloads/gt_cut_in.pkl"
    root_path = "/home/jiamiaoxu/data/code/detection_evaluation/prediction_Jiamiao/"
    gt_path = os.path.join(root_path, "our_data/20190412_rain_1_old")
    pre_path = os.path.join(root_path, "our_data/prediction_result")
    eval_path = os.path.join(root_path, "evaluations/eval_prediction_new")
    config_path = os.path.join(root_path, "configs/deeproute.eval.config")
    pose_config_file = os.path.join(root_path, "configs/pose.csv")
    pose_config_file = "/home/jiamiaoxu/data/code/detection_evaluation/tracking_Jiamiao/configs/pose.csv"
    attribute_file = "/home/jiamiaoxu/gt_prediction_state.pkl"
    # pre_path = "/home/jiamiaoxu/prediction_result_new/"
    pre_path = "/home/jiamiaoxu/dr_pp_predict_v2_cls_loc"
    config_path = "/home/jiamiaoxu/data/code/detection_evaluation/perception_evaluation/configs/prediction.eval.config"
    pose_config_file = "/home/jiamiaoxu/data/code/detection_evaluation/tracking_Jiamiao/configs/pose.csv"

    root_path = "/home/deeproute/workspace/deeproute/detection_evaluation"

    eval_path = os.path.join(root_path, "eval_prediction")
    config_path = os.path.join(root_path, "configs/prediction.eval.config")

    gt_path = "/media/deeproute/HDISK/dataset/lidar/label/20190412_rain_1"
    pre_path = "/home/deeproute/workspace/perception/perception/test/format/"
    pre_path = "/media/deeproute/HDISK/dataset/prediction_data/20190412_rain_1/prediction/"
    pose_config_file = "/media/deeproute/HDISK/dataset/lidar/config/20190412_rain_1/pose.csv"
    
    t = time.time()

    if len(sys.argv) > 1:
        pool_num = int(sys.argv[1])
    else:
        pool_num = 10

    compute_evaluate(gt_path, pre_path, eval_path, config_path, pose_config_file, cut_in_file, pool_num, False)
    print("Evaluation Finished!")
    print("The total evaluation time:%f" %(time.time()-t))
    print("All evaluation results are placed in the path: {}".format(eval_path))
