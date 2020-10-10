import sys
sys.path.append("..")
import multiprocessing
import pickle
from tqdm import tqdm
from core.hdmap import RoadMap
from protos.config_lidar_pb2 import LidarArray

def compute_agent_lane_id(frame_id, point_cloud_range):
    timestamp = predictionDataManager._timestamps[frame_id]
    gt_corners, gt_boxes, gt_names, gt_ids, gt_yaws = predictionDataManager.getObjectInfos(timestamp)
    pose_info = predictionDataManager._pose_buffer[timestamp]
    position, rotation = pose_info
    heading = -rotation[2]

    ego_car_lane = roadMap.get_lane_by_point(position[:2])
    near_lanes = roadMap.get_lane_around_ego_car(position, point_cloud_range, heading)

    near_lanes_ids = []
    for lane in near_lanes:
        near_lanes_ids.append(lane.id)
    successor_ids = roadMap.get_roi_successor_predecessor_id(ego_car_lane, near_lanes_ids)

    lane_ids = {}
    for id in gt_ids:
        gt_loc = predictionDataManager.getFutureTrajectory(id, timestamp, 1)[0][1:3]
        lane_ids[id] = roadMap.get_lane_id_by_point(gt_loc, near_lanes)

    is_same_lane = {}
    for id in lane_ids.keys():
        if lane_ids[id] in successor_ids:
            is_same_lane[id] = True
        else:
            is_same_lane[id] = False

    return(timestamp, is_same_lane, lane_ids)

def get_lane_id(dataManager, map_path, lane_id_file,\
                point_cloud_range = [[-70, 70], [-70, 70], [-5, 3]], pool_num=20):
    global predictionDataManager, roadMap
    predictionDataManager = dataManager
    roadMap = RoadMap(map_path, index_method="rtree")

    pool = multiprocessing.Pool(pool_num)
    res = []
    length = len(predictionDataManager._timestamps)
    for frame_id in tqdm(range(length)):
        #compute_agent_lane_id(frame_id, point_cloud_range)
        res.append(pool.apply_async(compute_agent_lane_id, (frame_id, point_cloud_range)))

    is_same_lane_tmp = {}
    lane_ids = {}
    for i in tqdm(range(len(res))):
        timestamp = res[i].get()[0]
        is_same_lane_tmp[timestamp] = res[i].get()[1]
        lane_ids[timestamp] = res[i].get()[2]

    pool.close()
    pool.join()

    is_same_lane = {}
    for timestamp in is_same_lane_tmp.keys():
        for id in is_same_lane_tmp[timestamp].keys():
            if id not in is_same_lane:
                is_same_lane[id] = []
            is_same_lane[id].append([timestamp, is_same_lane_tmp[timestamp][id]])

    predictionDataManager.savePickleFile([lane_ids, is_same_lane], lane_id_file)

    return lane_ids, is_same_lane

