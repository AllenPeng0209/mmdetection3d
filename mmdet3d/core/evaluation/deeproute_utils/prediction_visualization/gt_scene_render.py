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
from core.hdmap import RoadMap


class SceneRender(object):
    def __init__(self, map_path, predict_config):
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
        predict_time = int(predict_config.deeproute_eval_input_reader.predict_time)
        self.future_length = predict_time * frequency + 1
        self.stop_thresh = float(predict_config.deeproute_eval_input_reader.stop_thresh)

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

    def drawObject(self, x, y, z, l, w, heading, obj_layer, transformer):
        corners = visualize_utils.xywlYawToCorner([x, y, l, w, heading])
        corners = np.concatenate((corners, z * np.ones((corners.shape[0], 1))), axis=1)
        corners = transformer(corners)
        corners = self.lidarToImg(corners[:, 0:2])
        color = 1
        cv2.polylines(obj_layer, [corners.astype(int)], True, color, 3)
        #cv2.fillConvexPoly(obj_layer, corners.astype(int), color)
        return obj_layer

    def render(self, predictionDataManager, timestamp, position_center, heading_center, transformer, show_mode = 1):
        map_layers, layer_names = self.roadmap.drawNearMapSlice(position_center, \
                                                                self.lims, heading_center, self.sizes,
                                                                 self.need_types, transformer)

        ego_obj_layer = np.zeros((self.sizes[0], self.sizes[1]), dtype=np.float32)
        obj_layer = np.zeros((self.sizes[0], self.sizes[1]), dtype=np.float32)
        special_obj_layer = np.zeros((self.sizes[0], self.sizes[1]), dtype=np.float32)
        ego_obj_layer = self.drawObject(position_center[0], position_center[1], position_center[2],
                                        4.5, 2, -heading_center, ego_obj_layer, transformer)

        obj_trajectory = np.zeros((self.sizes[0], self.sizes[1]), dtype=np.float32)
        special_obj_trajectory = np.zeros((self.sizes[0], self.sizes[1]), dtype=np.float32)

        objects = predictionDataManager.getObject(timestamp)
        for obj in objects["objects"]:
            if self.name_to_label[obj['type'].lower()] != 0:
                continue

            # object info
            point = predictionDataManager.getFutureTrajectory(obj["id"], timestamp, 1)[0]
            x, y, z = point[1], point[2], point[3]
            l, w = obj["bounding_box"]["length"], obj["bounding_box"]["width"]
            heading = obj["heading"]
            heading = heading - heading_center

            # trajectory info
            trajectory = predictionDataManager.getFutureTrajectory(obj["id"], timestamp, self.future_length)
            trajectory = np.array(trajectory)[:, 1:3]

            if obj["id"] in predictionDataManager._gt_velocity[timestamp]:
                gt_velocity_cur = predictionDataManager._gt_velocity[timestamp][obj["id"]][2]
                abs_gt_velocity_cur = (gt_velocity_cur[0] ** 2 + gt_velocity_cur[1] ** 2) ** 0.5

            if show_mode == 1 and obj["id"] in predictionDataManager._gt_cut_in[timestamp] \
                    and predictionDataManager._gt_cut_in[timestamp][obj["id"]] == 1 \
                    and abs_gt_velocity_cur > self.stop_thresh:   # cut-in
                special_obj_layer = self.drawObject(x, y, z, l, w, heading, special_obj_layer, transformer)
                special_obj_trajectory = self.drawTrjactory(trajectory, position_center, transformer, special_obj_trajectory)

            elif show_mode == 2 and obj["id"] in predictionDataManager._gt_turn[timestamp] \
                    and predictionDataManager._gt_turn[timestamp][obj["id"]] == 1\
                    and abs_gt_velocity_cur > self.stop_thresh:   # turn
                special_obj_layer = self.drawObject(x, y, z, l, w, heading, special_obj_layer, transformer)
                special_obj_trajectory = self.drawTrjactory(trajectory, position_center, transformer, special_obj_trajectory)

            elif show_mode == 3 and obj["id"] in predictionDataManager._gt_on_lane[timestamp] \
                    and predictionDataManager._gt_on_lane[timestamp][obj["id"]] == 0\
                    :   # off_lane
                special_obj_layer = self.drawObject(x, y, z, l, w, heading, special_obj_layer, transformer)
                special_obj_trajectory = self.drawTrjactory(trajectory, position_center, transformer, special_obj_trajectory)

            else:
                obj_layer = self.drawObject(x, y, z, l, w, heading, obj_layer, transformer)
                obj_trajectory = self.drawTrjactory(trajectory, position_center, transformer, obj_trajectory)

        map_layers.append(special_obj_trajectory)
        map_layers.append(obj_trajectory)
        layer_names.append("SPECIAL_TRAJ")
        layer_names.append("OBJ_TRAJ")

        map_layers.append(ego_obj_layer)
        map_layers.append(obj_layer)
        map_layers.append(special_obj_layer)
        layer_names.append("EGO_OBJECT")
        layer_names.append("OBJECTS")
        layer_names.append("SPECIAL_OBJECTS")

        self.layer_names = layer_names
        return map_layers

    def MapLayersToImage(self, map_layers):
        # hdmap
        rgb_img = self.roadmap.MergeMapLayersToRgb(map_layers[:-5], self.layer_names[:-5])

        # objects
        obj_img = np.zeros(rgb_img.shape, dtype=np.uint8)
        obj_img[:, :, 0] = map_layers[-1] * 255
        rgb_img = cv2.addWeighted(rgb_img, 1, obj_img, 1, 0)
        obj_img = np.zeros(rgb_img.shape, dtype=np.uint8)
        obj_img[:, :, 1] = map_layers[-2] * 255
        rgb_img = cv2.addWeighted(rgb_img, 1, obj_img, 1, 0)
        obj_img = np.zeros(rgb_img.shape, dtype=np.uint8)
        obj_img[:, :, 2] = map_layers[-3] * 255
        rgb_img = cv2.addWeighted(rgb_img, 1, obj_img, 1, 0)

        # trajectory
        obj_img = np.zeros(rgb_img.shape, dtype=np.uint8)
        obj_img[:, :, 1] = map_layers[-4] * 255
        rgb_img = cv2.addWeighted(rgb_img, 1, obj_img, 1, 0)
        obj_img = np.zeros(rgb_img.shape, dtype=np.uint8)
        obj_img[:, :, 0] = map_layers[-5] * 255
        rgb_img = cv2.addWeighted(rgb_img, 1, obj_img, 1, 0)
        return rgb_img
