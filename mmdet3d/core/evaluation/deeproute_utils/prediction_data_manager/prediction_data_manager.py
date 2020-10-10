import sys
sys.path.append("..")
import os
import pickle
import json
import glob
from tqdm import tqdm
import pcl
import numpy as np
import core.box_ops as box_ops
from core.read_info import readLidarConfigFile
import core.calib_trans as calib_trans
from core.geometry_math import point2LineDistance
from prediction_data_manager.lane_id import get_lane_id

class PredictionDataManager(object):
    def __init__(self, gt_path, pcd_path, map_path, config_path, trajectory_file,
                 velocity_file, accelerate_file, turn_file, cut_in_file, on_lane_file, predict_config):
        self._gt_path = gt_path
        self._pcd_path = pcd_path
        self._map_path = map_path
        self._config_path = config_path
        config_files = glob.glob(os.path.join(self._config_path, "lidars_*.cfg"))
        assert len(config_files) == 1
        self._lidar_config_file = os.path.join(self._config_path, config_files[0])
        self._pose_file = os.path.join(self._config_path, "pose.csv")
        self._data_index = os.path.join(self._config_path, "data_index.csv")
        self._trajectory_file = trajectory_file
        self._velocity_file = velocity_file
        self._accelerate_file = accelerate_file
        self._turn_file = turn_file
        self._cut_in_file = cut_in_file
        self._on_lane_file = on_lane_file

        self._config_path = config_path
        self._frequency = int(predict_config.deeproute_eval_input_reader.frequency)
        self._acclerate_thresh = float(predict_config.deeproute_eval_input_reader.acclerate_thresh)
        self._turn_thresh = float(predict_config.deeproute_eval_input_reader.turn_thresh)
        self._predict_time = int(predict_config.deeproute_eval_input_reader.predict_time)
        self._name_to_label = {}
        for sample_group in predict_config.deeproute_eval_input_reader.name_to_label.sample_groups:
            self._name_to_label.update(dict(sample_group.name_to_num))


        self._pose_buffer, self._timestamps = self.loadPose()
        self._config_lidars = readLidarConfigFile(self._lidar_config_file)
        self._gt_trajectories = self.loadTrajectories()
        self._gt_velocity = self.getVelocity()
        self._gt_turn = self.getTurn()
        self._gt_accelerate = self.getAccelerate()
        self._gt_cut_in = self.getCutIn()
        self._gt_on_lane = self.getOnLane()
        #self._gt_avg_velocity = self.getAvgVelocity()

    def loadPose(self):
        with open(self._pose_file, 'r') as f:
            lines = f.readlines()
        timestamps = []
        pose_buffer = {}
        for i, line in enumerate(lines):
            if i == 0:
                continue
            fields = line.strip().split(", ")
            if len(fields) != 7:
                print("Load pose error: {}, in line: {}".format(self._pose_file, line))
                continue
            timestamp = fields[0]
            translation = (float(fields[1]), float(fields[2]), float(fields[3]))
            roll_pitch_yaw = (float(fields[4]), float(fields[5]), float(fields[6]))
            pose_buffer[int(timestamp)] = (translation, roll_pitch_yaw)
            timestamps.append(int(timestamp))
        return pose_buffer, timestamps

    def getPose(self, timestamp):
        return self._pose_buffer[timestamp]

    def getPcd(self, timestamp):
        timestamp = ''.join(list(str(timestamp)))[0:-6] + "_" + ''.join(list(str(timestamp)))[-6:] + ".pcd"
        try:
            points = pcl.load(os.path.join(self._pcd_path, timestamp))
            points = np.array(points)
        except:
            points = np.fromfile(os.path.join(self._pcd_path, timestamp), dtype=np.float32)
            points = points.reshape([-1, 4])
        return points

    def savePickleFile(self, infos, data_file):
        f = open(data_file, "wb")
        pickle.dump(infos, f, protocol=2)
        f.close()

    def loadPickleFile(self, data_file):
        with open(data_file, 'rb') as f:
            infos = pickle.load(f)
        return infos

    # object trajectory
    def loadObjectTrajectories(self):
        trajectories = {}
        for frame_id in tqdm(range(len(self._timestamps))):
            timestamp = self._timestamps[frame_id]
            pose_info = self._pose_buffer[timestamp]
            objects = self.getObject(timestamp)
            gt_ids = []
            for obj in objects["objects"]:
                if obj["id"] in gt_ids:
                    continue
                gt_ids.append(obj["id"])

                if obj["id"] not in trajectories:
                    trajectories[obj["id"]] = []

                x, y, z = obj["position"]["x"], obj["position"]["y"], obj["position"]["z"]
                l, w, h = obj["bounding_box"]["length"], obj["bounding_box"]["width"], obj["bounding_box"]["height"]
                heading = obj["heading"]
                pts = calib_trans.pointsSensingToMap_v2(pose_info, np.array([x,y,z]).reshape(-1, 3), self._config_lidars)[0]

                need_fields = [timestamp, pts[0], pts[1], pts[2], l, w, h, heading]
                trajectories[obj["id"]].append(need_fields)
        return trajectories

    def loadTrajectories(self):
        if os.path.exists(self._trajectory_file):
            print("Load trajectories from existed trajectory file: {}".format(self._trajectory_file))
            trajectories = self.loadPickleFile(self._trajectory_file)
        else:
            print("Load trajectories online...")
            trajectory_dir = "/".join(self._trajectory_file.split("/")[:-1])
            if not os.path.exists(trajectory_dir):
                os.makedirs(trajectory_dir)
            trajectories = self.loadObjectTrajectories()
            self.savePickleFile(trajectories, self._trajectory_file)
        return trajectories

    def getFutureTrajectory(self, obj_id, timestamp, future_length):
        trajectory = self._gt_trajectories[obj_id]
        for idx in range(len(trajectory)):
            if trajectory[idx][0] == timestamp:
                return trajectory[idx:min(idx + future_length, len(trajectory))]
        return []

    def getHistroyTrajectory(self, obj_id, timestamp, history_length):
        trajectory = self._gt_trajectories[obj_id]
        for idx in range(len(trajectory)):
            if trajectory[idx][0] == timestamp:
                return trajectory[max(idx-history_length, 0):idx]
        return []

    # object information
    def getObject(self, timestamp):
        timestamp = ''.join(list(str(timestamp)))[0:-6] + "_" + ''.join(list(str(timestamp)))[-6:] + ".txt"
        filepath = os.path.join(self._gt_path, timestamp)
        try:
            with open(filepath) as f:
                objects = json.load(f)
        except:
            objects = {}
            objects["objects"] = []
        return  objects

    def getObjectInfos(self, timestamp):
        objects = self.getObject(timestamp)

        gt_names = []
        gt_loc = []
        gt_dims = []
        gt_yaws = []
        gt_ids = []
        for obj in objects["objects"]:
            if obj["id"] in gt_ids:
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
        #box_ops.change_box3d_center_(gt_boxes, src=[0.5, 0.5, 0], dst=[0.5, 0.5, 0.5])
        gt_corners = box_ops.get_corners(gt_boxes[:, :2], gt_boxes[:, 3:5], gt_boxes[:, 6])
        return gt_corners, gt_boxes, gt_names, gt_ids, gt_yaws

    # velocity

    def getVelocity(self):
        if os.path.exists(self._velocity_file):
            print("Load velocity from existed velocity file: {}".format(self._velocity_file))
            velocity = self.loadPickleFile(self._velocity_file)
        else:
            velocity_dir = "/".join(self._velocity_file.split("/")[:-1])
            if not os.path.exists(velocity_dir):
                os.makedirs(velocity_dir)
            print("run velocity.py first to generate velocity.pkl file!")
            print("suggest using mpirun -np 3 python velocity.py...")
            raise
        #     print("Load velocity online...")
        #     compute_velocity(self._gt_path, self._pcd_path, self._config_path,\
        #                      self._pose_file, self._config_lidars, self._velocity_file)
        #     velocity = self.loadPickleFile(self._velocity_file)
        return velocity

    # accelerate attribute
    def loadObjectAccelerate(self):
        accelerate = {}
        frame_length = 5
        total_time = frame_length/self._frequency
        for frame_id in tqdm(range(len(self._timestamps))):
            timestamp = self._timestamps[frame_id]
            objects = self.getObject(timestamp)
            accelerate_per_frame = {}
            for obj in objects["objects"]:
                if self._name_to_label[obj['type'].lower()] != 0:
                    continue

                history_trajectory = self.getHistroyTrajectory(obj["id"], timestamp, frame_length)
                future_trajectory = self.getFutureTrajectory(obj["id"], timestamp, frame_length)
                if len(history_trajectory) < frame_length or len(future_trajectory) < frame_length:
                    accelerate_per_frame[obj["id"]] = 0
                    continue

                history_avg_velocity = [(history_trajectory[-1][1]-history_trajectory[0][1])/total_time,
                                        (history_trajectory[-1][2]-history_trajectory[0][2])/total_time]
                future_avg_velocity = [(future_trajectory[-1][1] - future_trajectory[0][1]) / total_time,
                                     (future_trajectory[-1][2] - future_trajectory[0][2]) / total_time]

                if (history_avg_velocity[0]-future_avg_velocity[0])/total_time > self._acclerate_thresh \
                    or (history_avg_velocity[1]-future_avg_velocity[1])/total_time > self._acclerate_thresh:
                    accelerate_per_frame[obj["id"]] = -1
                elif(future_avg_velocity[0]-history_avg_velocity[0])/total_time > self._acclerate_thresh \
                    or (future_avg_velocity[1]-history_avg_velocity[1])/total_time > self._acclerate_thresh:
                    accelerate_per_frame[obj["id"]] = 1
                else:
                    accelerate_per_frame[obj["id"]] = 0
            accelerate[timestamp] = accelerate_per_frame

        return accelerate

    def getAccelerate(self):
        if os.path.exists(self._accelerate_file):
            print("Load accelerate from existed accelerate file: {}".format(self._accelerate_file))
            accelerate = self.loadPickleFile(self._accelerate_file)
        else:
            print("Load accelerate online...")
            accelerate_dir = "/".join(self._accelerate_file.split("/")[:-1])
            if not os.path.exists(accelerate_dir):
                os.makedirs(accelerate_dir)
            accelerate = self.loadObjectAccelerate()
            self.savePickleFile(accelerate, self._accelerate_file)
        return accelerate

    # average velocity
    def loadObjectAvgVelocity(self):
        avg_velocity = {}
        frame_length = 5
        total_time = frame_length/self._frequency
        for frame_id in tqdm(range(len(self._timestamps))):
            timestamp = self._timestamps[frame_id]
            objects = self.getObject(timestamp)
            avg_velocity_per_frame = {}
            for obj in objects["objects"]:
                if self._name_to_label[obj['type'].lower()] != 0:
                    continue
                history_trajectory = self.getHistroyTrajectory(obj["id"], timestamp, frame_length)
                if len(history_trajectory) < frame_length:
                    # avg_velocity_per_frame[obj["id"]] = self._gt_velocity[timestamp][obj["id"]][2]
                    continue

                avg_velocity_per_frame[obj["id"]] = [(history_trajectory[-1][1]-history_trajectory[0][1])/total_time,
                                        (history_trajectory[-1][2]-history_trajectory[0][2])/total_time, 0]

            avg_velocity[timestamp] = avg_velocity_per_frame

        return avg_velocity

    def getAvgVelocity(self):
        velocity_dir = "/".join(self._velocity_file.split("/")[:-1])
        avg_velocity_file = os.path.join(velocity_dir, "avg_velocity.pkl")
        if os.path.exists(avg_velocity_file):
            print("Load average velocity from existed velocity file: {}".format(avg_velocity_file))
            avg_velocity = self.loadPickleFile(avg_velocity_file)
        else:
            print("Load average velocity online...")
            if not os.path.exists(velocity_dir):
                os.makedirs(velocity_dir)
            avg_velocity = self.loadObjectAvgVelocity()
            self.savePickleFile(avg_velocity, avg_velocity_file)
        return avg_velocity

    # turn attribute
    def getTrajcetoryTurn(self, trajectory):
        line = [[trajectory[0][1], trajectory[0][2]], [trajectory[-1][1], trajectory[-1][2]]]
        max_dist = -1e9
        for i in range(1, len(trajectory)-1):
            point = [trajectory[i][1], trajectory[i][2]]
            dist = point2LineDistance(point, line)
            if dist > max_dist:
                max_dist = dist

        turn_val = max_dist
        return turn_val

    def loadObjectTurn(self):
        turn = {}
        frame_length = 30
        for frame_id in tqdm(range(len(self._timestamps))):
            timestamp = self._timestamps[frame_id]
            objects = self.getObject(timestamp)
            turn_per_frame = {}
            for obj in objects["objects"]:
                if self._name_to_label[obj['type'].lower()] != 0:
                    continue

                future_trajectory = self.getFutureTrajectory(obj["id"], timestamp, frame_length)
                if len(future_trajectory) < frame_length:
                    turn_per_frame[obj["id"]] = 0            # no turn
                    continue

                turn_val = self.getTrajcetoryTurn(future_trajectory)
                if turn_val > self._turn_thresh:
                    turn_per_frame[obj["id"]] = 1
                else:
                    turn_per_frame[obj["id"]] = 0

            turn[timestamp] = turn_per_frame
        return turn

    def getTurn(self):
        if os.path.exists(self._turn_file):
            print("Load turn from existed turn file: {}".format(self._turn_file))
            turn = self.loadPickleFile(self._turn_file)
        else:
            print("Load turn online...")
            turn_dir = "/".join(self._turn_file.split("/")[:-1])
            if not os.path.exists(turn_dir):
                os.makedirs(turn_dir)
            turn = self.loadObjectTurn()
            self.savePickleFile(turn, self._turn_file)
        return turn

    # cut in attribute
    def getFutureLaneTrajectory(self, is_same_lane, obj_id, timestamp, future_length):
        lane_trajectory = is_same_lane[obj_id]
        for idx in range(len(lane_trajectory)):
            if lane_trajectory[idx][0] == timestamp:
                return lane_trajectory[idx:min(idx + future_length, len(lane_trajectory))]
        return []

    def loadObjectCutIn(self, is_same_lane):
        cut_in_length = self._predict_time * self._frequency + 1

        cut_in = {}
        for frame_id in tqdm(range(len(self._timestamps))):
            timestamp = self._timestamps[frame_id]
            objects = self.getObject(timestamp)
            cut_in_per_frame = {}
            for obj in objects["objects"]:
                if self._name_to_label[obj['type'].lower()] != 0:
                    continue

                lane_trajectory = self.getFutureLaneTrajectory(is_same_lane, obj["id"], timestamp, cut_in_length)
                trajectory = self.getFutureTrajectory(obj["id"], timestamp, cut_in_length)
                global_trajectory = np.array(trajectory)[:, 1:4]
                local_trajectory = []
                for idx in range(len(global_trajectory)):
                    local_trajectory.append(calib_trans.pointsMapToSensing_v2(\
                        self._pose_buffer[self._timestamps[frame_id+idx]], global_trajectory[idx, :].reshape(-1, 3), self._config_lidars)[0])

                cut_in_flag = False
                prev_flag = False
                for idx, object_lane in enumerate(lane_trajectory):
                    if prev_flag and object_lane[1] and local_trajectory[idx][0] > 0 and local_trajectory[idx][0] < 50:
                        cut_in_flag = True
                    if object_lane[1] == False:
                        prev_flag = True
                if cut_in_flag:
                    cut_in_per_frame[obj["id"]] = 1   # cut in
                else:
                    cut_in_per_frame[obj["id"]] = 0

            cut_in[timestamp] = cut_in_per_frame

        return cut_in

    def getCutIn(self):
        if os.path.exists(self._cut_in_file):
            print("Load cut in from existed cut in file: {}".format(self._cut_in_file))
            cut_in = self.loadPickleFile(self._cut_in_file)
        else:
            print("Load cut in online...")
            cut_in_dir = "/".join(self._cut_in_file.split("/")[:-1])
            if not os.path.exists(cut_in_dir):
                os.makedirs(cut_in_dir)

            lane_id_file = os.path.join(cut_in_dir, "gt_lane_id.pkl")
            if not os.path.exists(lane_id_file):
                print("Load lane id in online...")
                lane_ids, is_same_lane = get_lane_id(self, self._map_path, lane_id_file)
            else:
                print("Load lane id from existed lane id file: {}".format(lane_id_file))
                info = self.loadPickleFile(lane_id_file)
                is_same_lane = info[1]

            cut_in = self.loadObjectCutIn(is_same_lane)
            self.savePickleFile(cut_in, self._cut_in_file)
        return cut_in

    # on lane or off lane
    def loadObjectOnLane(self, lane_ids):
        on_lane = {}
        for frame_id in tqdm(range(len(self._timestamps))):
            timestamp = self._timestamps[frame_id]
            objects = self.getObject(timestamp)
            on_lane_per_frame = {}
            for obj in objects["objects"]:
                if self._name_to_label[obj['type'].lower()] != 0:
                    continue

                if lane_ids[timestamp][obj["id"]] != -1:
                    on_lane_per_frame[obj["id"]] = 1
                else:
                    on_lane_per_frame[obj["id"]] = 0
            on_lane[timestamp] = on_lane_per_frame
        return on_lane

    def getOnLane(self):
        if os.path.exists(self._on_lane_file):
            print("Load on lane from existed on lane file: {}".format(self._on_lane_file))
            on_lane = self.loadPickleFile(self._on_lane_file)
        else:
            print("Load on lane online...")
            on_lane_dir = "/".join(self._on_lane_file.split("/")[:-1])
            if not os.path.exists(on_lane_dir):
                os.makedirs(on_lane_dir)

            lane_id_file = os.path.join(on_lane_dir, "gt_lane_id.pkl")
            if not os.path.exists(lane_id_file):
                print("Load lane id in online...")
                lane_ids, is_same_lane = get_lane_id(self, self._map_path, lane_id_file)
            else:
                print("Load lane id from existed lane id file: {}".format(lane_id_file))
                info = self.loadPickleFile(lane_id_file)
                lane_ids = info[0]

            on_lane = self.loadObjectOnLane(lane_ids)
            self.savePickleFile(on_lane, self._on_lane_file)
        return on_lane



