import sys
sys.path.append("..")
import os

import math
import time

import glob
import numpy as np
import cv2
import pcl
from functools import partial
from tqdm import tqdm

from prediction_visualization import visualize_utils
from core.calib_trans import pointsSensingToMap_v2
from core.hdmap import RoadMap


class SceneRender(object):
    def __init__(self, map_path, predict_config, show_time, show_error):
        # focus areas
        self.xlim = [-10, 70]
        self.ylim = [-40, 40]
        self.lims = [self.xlim, self.ylim, [-4, 3]]
        self.sizes = [800, 800, 3]
        self.resolution = (self.xlim[1] - self.xlim[0]) / self.sizes[0]
        self.resolution_inv = 1.0 / self.resolution
        self.render_obj_types = [5, 6]
        self.need_types = ['JUNCTION', 'CROSS_WALK', 'STOP_LINE', 'LANE_BOUNDARY', 'LANE']
        self.roadmap = RoadMap(map_path, index_method="rtree")
        self.name_to_label = {}
        for sample_group in predict_config.deeproute_eval_input_reader.name_to_label.sample_groups:
            self.name_to_label.update(dict(sample_group.name_to_num))
        frequency = int(predict_config.deeproute_eval_input_reader.frequency)
        self.predict_time = int(predict_config.deeproute_eval_input_reader.predict_time)
        self.show_time = show_time
        self.show_error = show_error
        self.show_length = self.show_time * frequency + 1
        self.stop_thresh = float(predict_config.deeproute_eval_input_reader.stop_thresh)

        eval_frequency = int(predict_config.deeproute_eval_input_reader.eval_frequency)
        self.internal_gap = int(frequency / eval_frequency)
        self.show_selected = list(range(self.show_length - 1, 0, -self.internal_gap))
        self.show_selected.sort()

    def lidarToImg(self, points):
        points = (points - np.array([self.xlim[0], self.ylim[0]])) / (self.xlim[1] - self.xlim[0]) * self.sizes[0]
        return points

    def drawTrjactory(self, points, position, transformer, obj_layer):
        points = np.concatenate((points, position[2] * np.ones((points.shape[0], 1))), axis=1)
        points = transformer(points)[:, 0:2]
        points = self.lidarToImg(points)
        points = np.array(points).astype(int)
        points = points.reshape((-1, 1, 2))
        color = 1
        cv2.polylines(obj_layer, [points], False, color, 2)
        return obj_layer

    def drawObject(self, x, y, z, l, w, heading, obj_layer, transformer, size = 2):
        corners = visualize_utils.xywlYawToCorner([x, y, l, w, heading])
        corners = np.concatenate((corners, z * np.ones((corners.shape[0], 1))), axis=1)
        corners = transformer(corners)
        corners = self.lidarToImg(corners[:, 0:2])
        color = 1
        cv2.polylines(obj_layer, [corners.astype(int)], True, color, size)
        #cv2.fillConvexPoly(obj_layer, corners.astype(int), color)
        return obj_layer

    def drawMarks(self, trajectory, heading, obj_layer, transformer):
        trajectory = trajectory.tolist()
        for i in range(len(trajectory)):
            point = trajectory[i]
            x, y, z = point[0], point[1], point[2]
            obj_layer = self.drawObject(x, y, z, 0.5, 0.5, heading, obj_layer, transformer)

        return obj_layer

    def render(self, pre_objects, gt_ids, gt_pre_dict, predictionDataManager, timestamp, \
               position_center, heading_center, transformer, has_priority):

        map_layers, layer_names = self.roadmap.drawNearMapSlice(position_center, \
                                                                self.lims, heading_center, self.sizes,
                                                                 self.need_types, transformer)

        ego_obj_layer = np.zeros((self.sizes[0], self.sizes[1]), dtype=np.float32)
        gt_obj_layer = np.zeros((self.sizes[0], self.sizes[1]), dtype=np.float32)
        dt_obj_layer = np.zeros((self.sizes[0], self.sizes[1]), dtype=np.float32)
        hp_dt_obj_layer = np.zeros((self.sizes[0], self.sizes[1]), dtype=np.float32)
        ego_obj_layer = self.drawObject(position_center[0], position_center[1], position_center[2],
                                        4.5, 2, -heading_center, ego_obj_layer, transformer)

        gt_obj_trajectory = np.zeros((self.sizes[0], self.sizes[1]), dtype=np.float32)
        dt_obj_trajectory = np.zeros((self.sizes[0], self.sizes[1]), dtype=np.float32)
        hp_dt_obj_trajectory = np.zeros((self.sizes[0], self.sizes[1]), dtype=np.float32)

        gt_marks = np.zeros((self.sizes[0], self.sizes[1]), dtype=np.float32)
        dt_marks = np.zeros((self.sizes[0], self.sizes[1]), dtype=np.float32)
        hp_dt_marks = np.zeros((self.sizes[0], self.sizes[1]), dtype=np.float32)

        objects = predictionDataManager.getObject(timestamp)
        pose_info = predictionDataManager._pose_buffer[timestamp]
        config_lidars = predictionDataManager._config_lidars

        flag = False
        for obj in objects["objects"]:
            if self.name_to_label[obj['type'].lower()] != 0:
                continue
            if obj["id"] not in gt_ids:
                continue

            # gt infos
            # object info
            point = predictionDataManager.getFutureTrajectory(obj["id"], timestamp, 1)[0]
            gt_x, gt_y, gt_z = point[1], point[2], point[3]
            gt_l, gt_w = obj["bounding_box"]["length"], obj["bounding_box"]["width"]
            gt_heading = obj["heading"]
            gt_heading = gt_heading - heading_center

            # trajectory info
            ori_trajectory = predictionDataManager.getFutureTrajectory(obj["id"], timestamp, self.show_length)
            gt_trajectory = np.array(ori_trajectory)[:, 1:3]

            # markers
            points = np.array(ori_trajectory)[:, 1:4]
            if points.shape[0] == self.show_length:
                points = points[self.show_selected, :]
            elif points.shape[0] > 5:
                part_selected = list(range(5, points.shape[0], self.internal_gap))
                points = points[part_selected, :]
            else:
                points = np.array([]).reshape(-1, 3)

            gt_points = np.concatenate((np.array([gt_x, gt_y, gt_z]).reshape(1,3), points), axis = 0)

            # dt infos
            # object info
            dt_id = gt_pre_dict[obj["id"]]
            loc, dims, yaws, names, ids, trajectories, priorities = pre_objects
            pre_idx = ids.index(dt_id)
            dt_l, dt_w = dims[pre_idx][0], dims[pre_idx][1]
            dt_heading = -yaws[pre_idx][2] - heading_center
            new_loc = pointsSensingToMap_v2(pose_info, np.array(loc[pre_idx]).reshape(1,3), config_lidars)[0].tolist()
            dt_x, dt_y, dt_z = new_loc[0], new_loc[1], new_loc[2]

            # trajectory and markers info
            trajectory = trajectories[pre_idx].tolist()
            trajectory = trajectory[0:int((self.show_time/self.predict_time)*len(trajectory))]
            trajectory.insert(0, [loc[pre_idx][0], loc[pre_idx][1]])
            trajectory = np.array(trajectory).reshape(-1, 2)
            trajectory = np.concatenate((trajectory, np.array([loc[pre_idx][2]]*trajectory.shape[0]).reshape(-1, 1)), axis=1)
            new_trajectory_3 = pointsSensingToMap_v2(pose_info, trajectory, config_lidars)
            new_trajectory_2 = new_trajectory_3[:, 0: 2]

            # compare L1 error:
            gt_last_points = gt_points[-1, 0:2]
            dt_last_points = new_trajectory_2[-1, 0:2]
            error = abs(gt_last_points[0]-dt_last_points[0]) + abs(gt_last_points[1]-dt_last_points[1])
            #print(error)
            if error < self.show_error:
                continue

            flag = True

            # draw
            gt_obj_layer = self.drawObject(gt_x, gt_y, gt_z, gt_l, gt_w, gt_heading, gt_obj_layer, transformer)
            gt_obj_trajectory = self.drawTrjactory(gt_trajectory, position_center, transformer, gt_obj_trajectory)
            gt_marks = self.drawMarks(gt_points, gt_heading, gt_marks, transformer)

            if has_priority and priorities[pre_idx] == 1:
                hp_dt_obj_layer = self.drawObject(dt_x, dt_y, dt_z, dt_l, dt_w, dt_heading, hp_dt_obj_layer, transformer)
                hp_dt_obj_trajectory = self.drawTrjactory(new_trajectory_2, position_center, transformer, hp_dt_obj_trajectory)
                hp_dt_marks = self.drawMarks(new_trajectory_3, dt_heading, hp_dt_marks, transformer)
            else:
                dt_obj_layer = self.drawObject(dt_x, dt_y, dt_z, dt_l, dt_w, dt_heading, dt_obj_layer, transformer)
                dt_obj_trajectory = self.drawTrjactory(new_trajectory_2, position_center, transformer, dt_obj_trajectory)
                dt_marks = self.drawMarks(new_trajectory_3, dt_heading, dt_marks, transformer)


        map_layers.append(gt_marks)
        map_layers.append(dt_marks)
        map_layers.append(hp_dt_marks)
        layer_names.append("GT_MARK")
        layer_names.append("DT_MARK")
        layer_names.append("HP_DT_MARK")

        map_layers.append(gt_obj_trajectory)
        map_layers.append(dt_obj_trajectory)
        map_layers.append(hp_dt_obj_trajectory)
        layer_names.append("GT_OBJ_TRAJ")
        layer_names.append("DT_OBJ_TRAJ")
        layer_names.append("HP_DT_OBJ_TRAJ")

        map_layers.append(ego_obj_layer)
        map_layers.append(gt_obj_layer)
        map_layers.append(dt_obj_layer)
        map_layers.append(hp_dt_obj_layer)
        layer_names.append("EGO_OBJECT")
        layer_names.append("GT_OBJECTS")
        layer_names.append("DT_OBJECTS")
        layer_names.append("HP_DT_OBJECTS")

        self.layer_names = layer_names
        return map_layers, flag

    def MapLayersToImage(self, map_layers):
        # hdmap
        rgb_img = self.roadmap.MergeMapLayersToRgb(map_layers[:-10], self.layer_names[:-10])

        # objects
        obj_img = np.zeros(rgb_img.shape, dtype=np.uint8)
        obj_img[:, :, 0] = map_layers[-1] * 255
        rgb_img = cv2.addWeighted(rgb_img, 1, obj_img, 1, 0)
        obj_img = np.zeros(rgb_img.shape, dtype=np.uint8)
        obj_img[:, :, 2] = map_layers[-2] * 255
        rgb_img = cv2.addWeighted(rgb_img, 1, obj_img, 1, 0)
        obj_img = np.zeros(rgb_img.shape, dtype=np.uint8)
        obj_img[:, :, 1] = map_layers[-3] * 255
        rgb_img = cv2.addWeighted(rgb_img, 1, obj_img, 1, 0)
        obj_img = np.zeros(rgb_img.shape, dtype=np.uint8)
        obj_img[:, :, 0] = map_layers[-4] * 255
        rgb_img = cv2.addWeighted(rgb_img, 1, obj_img, 1, 0)

        # trajectory
        obj_img = np.zeros(rgb_img.shape, dtype=np.uint8)
        obj_img[:, :, 0] = map_layers[-5] * 255
        rgb_img = cv2.addWeighted(rgb_img, 1, obj_img, 1, 0)
        obj_img = np.zeros(rgb_img.shape, dtype=np.uint8)
        obj_img[:, :, 2] = map_layers[-6] * 255
        rgb_img = cv2.addWeighted(rgb_img, 1, obj_img, 1, 0)
        obj_img = np.zeros(rgb_img.shape, dtype=np.uint8)
        obj_img[:, :, 1] = map_layers[-7] * 255
        rgb_img = cv2.addWeighted(rgb_img, 1, obj_img, 1, 0)

        # markers
        obj_img = np.zeros(rgb_img.shape, dtype=np.uint8)
        obj_img[:, :, 0] = map_layers[-8] * 255
        rgb_img = cv2.addWeighted(rgb_img, 1, obj_img, 1, 0)
        obj_img = np.zeros(rgb_img.shape, dtype=np.uint8)
        obj_img[:, :, 2] = map_layers[-9] * 255
        rgb_img = cv2.addWeighted(rgb_img, 1, obj_img, 1, 0)
        obj_img = np.zeros(rgb_img.shape, dtype=np.uint8)
        obj_img[:, :, 1] = map_layers[-10] * 255
        rgb_img = cv2.addWeighted(rgb_img, 1, obj_img, 1, 0)
        return rgb_img
