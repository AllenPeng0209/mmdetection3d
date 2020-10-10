import os, sys
sys.path.append("..")
import json
import math
import numpy as np
from core.read_info import readLidarConfigFile

def getRotationMatrixFromRollPitchYaw(roll_pitch_yaw):
    sr = math.sin(roll_pitch_yaw[0])
    sp = math.sin(roll_pitch_yaw[1])
    sy = math.sin(roll_pitch_yaw[2])
    cr = math.cos(roll_pitch_yaw[0])
    cp = math.cos(roll_pitch_yaw[1])
    cy = math.cos(roll_pitch_yaw[2])
    rot = np.array([cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
                    sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
                    -sp, cp * sr, cp * cr]).reshape((3, 3))
    return rot

def pointsSensingToMap(pose, points):
    pts = np.array(points)
    if pts.shape[1] == 2:
        pts = np.concatenate((pts, np.zeros((pts.shape[0], 1))), axis=1)

    translation_m2v, roll_pitch_yaw = pose
    rotation_m2v = getRotationMatrixFromRollPitchYaw(roll_pitch_yaw)
    pts = (rotation_m2v.dot((pts).T).T + translation_m2v)
    return pts

def pointsMapToSensing(pose, points):
    pts = np.array(points)
    if pts.shape[1] == 2:
        pts = np.concatenate((pts, np.zeros((pts.shape[0], 1))), axis=1)

    translation_m2v, roll_pitch_yaw = pose
    rotation_m2v = getRotationMatrixFromRollPitchYaw(roll_pitch_yaw)
    rotation_m2v_inv = np.linalg.inv(rotation_m2v)
    pts = rotation_m2v_inv.dot((pts - translation_m2v).T).T
    return pts

def pointsSensingToMap_v2(pose, points, config_lidars):
    pts = np.array(points)
    if pts.shape[1] == 2:
        pts = np.concatenate((pts, np.zeros((pts.shape[0], 1))), axis=1)

    vehicle_to_sensing = config_lidars.vehicle_to_sensing
    translation_v2s = np.array([vehicle_to_sensing.position.x, vehicle_to_sensing.position.y, vehicle_to_sensing.position.z])

    translation_m2v, roll_pitch_yaw = pose
    rotation_m2v = getRotationMatrixFromRollPitchYaw(roll_pitch_yaw)
    pts = (rotation_m2v.dot((pts + translation_v2s).T).T + translation_m2v)
    return pts

def pointsMapToSensing_v2(pose, points, config_lidars):
    pts = np.array(points)
    if pts.shape[1] == 2:
        pts = np.concatenate((pts, np.zeros((pts.shape[0], 1))), axis=1)

    vehicle_to_sensing = config_lidars.vehicle_to_sensing
    translation_v2s = np.array([vehicle_to_sensing.position.x, vehicle_to_sensing.position.y, vehicle_to_sensing.position.z])
    translation_m2v, roll_pitch_yaw = pose
    rotation_m2v = getRotationMatrixFromRollPitchYaw(roll_pitch_yaw)
    rotation_m2v_inv = np.linalg.inv(rotation_m2v)
    # translation_m2v = np.array(list(translation_m2v)).reshape(1, -1)
    # print(pts)
    # print(translation_m2v)
    # print(pts - translation_m2v)
    pts = rotation_m2v_inv.dot((pts - translation_m2v).T).T - translation_v2s
    return pts


def local2global2local(data, timeStamp, timeStamps, pose_infos, evaluate_interval):
    idx = timeStamps.index(timeStamp)
    pos = []
    pose_info = pose_infos[str(int(timeStamp.split(".")[0]))]
    pos.append(data[0][:,0:2])
    pos.append(data[1][:,0:2])
    for i in range(2, len(data)):
        pose_info1 = pose_infos[str(int(timeStamps[idx + (i-1) * evaluate_interval].split(".")[0]))]
        pts = pointsSensingToMap(pose_info1, data[i])
        pts = pointsMapToSensing(pose_info, pts)
        pos.append(pts[:,0:2])
    return pos

def velocityglobal2local(velocities, locs, ids, timeStamp, pose_infos, lidarConfigFile, tracker = True):
    pose_info = pose_infos[str(int(timeStamp))]
    lidar_config = readLidarConfigFile(lidarConfigFile)

    # a = list(str(int(timeStamp.split(".")[0]) - 100000))
    # a.insert(-6, "_")
    # b = "".join(a)
    # pose_info_pre = pose_infos[str(int(b))]

    local_velocity = []
    for idx, id in enumerate(ids):
        loc = locs[idx]
        if tracker:
            velocity_box = velocities[idx]
            velocity_icp = velocities[idx]
        else:
            if id in velocities:
                velocity = velocities[id]
                velocity_box = velocity[1]
                velocity_icp = velocity[2]
            else:
                velocity_box = [0.0, 0.0, 0.0]
                velocity_icp = [0.0, 0.0, 0.0]

        velocity_box = [velocity_box[0] + 0.0001, velocity_box[1], 0]
        velocity_icp = [velocity_icp[0] + 0.0001, velocity_icp[1], 0]
        base = pointsMapToSensing_v2(pose_info, np.array([0.0, 0.0, 0.0]).reshape(1,-1), lidar_config)[0].tolist()

        velocity_box_local = pointsMapToSensing_v2(pose_info, np.array(velocity_box).reshape(1,-1), lidar_config)[0].tolist()
        velocity_box_local = [velocity_box_local[0]-base[0], velocity_box_local[1]-base[1], 0]
        angle = math.tanh(velocity_box_local[1]/velocity_box_local[0])
        length = math.sqrt(velocity_box[0]**2 + velocity_box[1]**2)
        velocity_box_local1 = [abs(length*math.cos(angle))*velocity_box_local[0]/abs(velocity_box_local[0]), abs(length*math.sin(angle))*velocity_box_local[1]/abs(velocity_box_local[1]), 0]
        velocity_box_local = [velocity_box_local1[0]+loc[0], velocity_box_local1[1]+loc[1], velocity_box_local1[2]+loc[2]]

        velocity_icp_local = pointsMapToSensing_v2(pose_info, np.array(velocity_icp).reshape(1, -1), lidar_config)[0].tolist()
        velocity_icp_local = [velocity_icp_local[0] - base[0], velocity_icp_local[1] - base[1], 0]
        angle = math.tanh(velocity_icp_local[1] / velocity_icp_local[0])
        length = math.sqrt(velocity_icp[0] ** 2 + velocity_icp[1] ** 2)
        velocity_icp_local1 = [abs(length*math.cos(angle))*velocity_icp_local[0]/abs(velocity_icp_local[0]), abs(length*math.sin(angle))*velocity_icp_local[1]/abs(velocity_icp_local[1]), 0]
        velocity_icp_local = [velocity_icp_local1[0]+loc[0], velocity_icp_local1[1]+loc[1], velocity_icp_local1[2]+loc[2]]

        if int(id) == 2948:
            print(velocity_box)
            print(velocity_box_local1)
        local_velocity.append([loc, velocity_box_local, velocity_icp_local])

    return local_velocity

def get_lidar_box_per_frame(gt_path, det_path, timeStamp):
    timeStamp = timeStamp + ".txt"

    try:
        with open(os.path.join(gt_path, timeStamp), "rb") as f:
            objects = json.load(f)
    except:
        objects = {}
        objects["objects"] = []

    gt_names = []
    gt_loc = []
    gt_dims = []
    gt_yaws = []
    for obj in objects["objects"]:
        if obj['type'].lower() == "cone" and obj["position"]["x"] < 0:
            continue

        gt_names.append(obj['type'].lower())
        gt_loc.append([obj["position"]["x"], obj["position"]["y"], obj["position"]["z"]])
        gt_dims.append([obj["bounding_box"]["length"], obj["bounding_box"]["width"], obj["bounding_box"]["height"]])
        gt_yaws.append([0, 0, obj["heading"]])

    assert os.path.exists(os.path.join(det_path, timeStamp))
    try:
        with open(os.path.join(det_path, timeStamp), "rb") as f:
            objects = json.load(f)
    except:
        objects = {}
        objects["objects"] = []

    dt_names = []
    dt_loc = []
    dt_dims = []
    dt_yaws = []
    for obj in objects["objects"]:
        if obj['type'].lower() == "cone" and obj["position"]["x"] < 0:
            continue

        dt_names.append(obj['type'].lower())
        dt_loc.append([obj["position"]["x"], obj["position"]["y"], obj["position"]["z"]])
        dt_dims.append([obj["bounding_box"]["length"], obj["bounding_box"]["width"], obj["bounding_box"]["height"]])
        dt_yaws.append([0, 0, obj["heading"]])

    return gt_loc, gt_dims, gt_yaws, gt_names, dt_loc, dt_dims, dt_yaws, dt_names
