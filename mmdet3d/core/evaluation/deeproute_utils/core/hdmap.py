# Copyright 2019 deeproute.ai. All Rights Reserved.
# Author: Xiaoyi Zou (xiaoyizou@deeproute.ai)
# Author: Jiamiao Xu (jiamiaoxu@deeproute.ai)

import sys
sys.path.append("..")
import math

import numpy as np
import cv2
import copy
import time
import queue

import core.kdtree as myKdtree
import rtree
from protos import hd_map_pb2
from core.geometry_math import compute_distance_point2Seg

LayerNames = ['JUNCTION', 'BG_LANE', 'CROSS_WALK', 'LANE_BOUNDARY', 'LANE', 'STOP_LINE']
MapItemColors = {'LANE': (200, 180, 0),
                 'BG_LANE': (20, 100, 80),
                 'STOP_LINE': (0, 0, 255),
                 'LANE_BOUNDARY': (255, 255, 255),
                 'CROSS_WALK': (180, 80, 80),
                 'JUNCTION': (80, 80, 80)}


def MergeMapLayersToRgb(map_layers):
    colors = [(0, 0, 0)]
    ids = np.zeros((map_layers[0].shape[0], map_layers[0].shape[1]), dtype=np.uint8)
    for i, render_type in enumerate(LayerNames):
        color = MapItemColors[render_type]
        colors.append(color)
        ids[map_layers[i] > 0] = i + 1
    colors = np.array(colors)
    rgb_img = np.zeros((map_layers[0].shape[0], map_layers[0].shape[1], 3), dtype=np.uint8)
    rgb_img[:, :] = colors[ids[:, :]]
    return rgb_img


# road map
class RoadMap:
    def __init__(self, roadmap_file, index_method="rtree"):
        print("loading road map and init kdtree/rtree...")
        self.roadMap = self.loadMapFromFile(roadmap_file)

        self.map_items = {}
        self.map_items['LANE'] = self.roadMap.lanes
        self.map_items['STOP_LINE'] = self.roadMap.stop_lines
        self.map_items['LANE_BOUNDARY'] = self.roadMap.lane_boundaries
        self.map_items['CROSS_WALK'] = self.roadMap.cross_walks
        self.map_items['JUNCTION'] = self.roadMap.junctions

        self.use_rtree = False
        if index_method == "rtree":
            self.rtrees, self.rtrees_objects = self.initRtrees()
            self.use_rtree = True
        else:
            self.kdtrees = self.initKdtrees()

        self.lane_boundaries = {}
        for data in self.map_items['LANE_BOUNDARY']:
            self.lane_boundaries[data.id] = [[pt.x, pt.y] for pt in data.boundary.points]

        self.renderOrder = ['JUNCTION', 'BG_LANE', 'CROSS_WALK', 'LANE_BOUNDARY', 'LANE', 'STOP_LINE']

        self.map_item_colors = {'LANE': (0, 0, 0),
                                'BG_LANE': (0, 0, 0),
                                'STOP_LINE': (0, 0, 255),
                                'LANE_BOUNDARY': (255, 255, 255),
                                'CROSS_WALK': (180, 80, 80),
                                'JUNCTION': (80, 80, 80)}

        self.map_item_thickness = {'LANE': 2,
                                   'STOP_LINE': 4,
                                   'LANE_BOUNDARY': 2}

    def loadMapFromFile(self, map_file):
        roadMap = hd_map_pb2.RoadMap()
        with open(map_file, "rb") as f:
            roadMap.ParseFromString(f.read())
            # text_format.Parse(f.read(), roadMap)
        return roadMap

    # get the width of lane
    def getlaneWidth(self, lane):
        left_boundary_id = lane.left_boundary_id
        right_boundary_id = lane.right_boundary_id
        left_boundary = self.get_lane_boundary_by_id(left_boundary_id)
        right_boundary = self.get_lane_boundary_by_id(right_boundary_id)
        left_points = left_boundary.boundary.points
        right_points = right_boundary.boundary.points

        min_dist = 1e9
        point = [left_points[int(len(left_points)/2)].x, left_points[int(len(left_points)/2)].y]
        for i in range(1, len(right_points)):
            point1 = [right_points[i - 1].x, right_points[i - 1].y]
            point2 = [right_points[i].x, right_points[i].y]
            w = compute_distance_point2Seg(point1, point2, point)
            if w < min_dist:
                min_dist = w

        return min_dist

    # get the lane of any one points(global)
    def get_lane_by_point(self, point):
        min_dist = 1e9
        lane = None
        for data in self.map_items['LANE']:
            points = data.centerline.points
            for i in range(1, len(points)):
                point1 = [points[i-1].x, points[i-1].y]
                point2 = [points[i].x, points[i].y]
                l = compute_distance_point2Seg(point1, point2, point)
                if l < min_dist:
                    min_dist = l
                    lane = data

        if lane != None:
            width = self.getlaneWidth(lane)
            if min_dist > width / 2 + 0.2:  # max_land_width/2
                lane = None
        return lane

    def get_lane_by_point_v2(self, point, lanes):
        min_dist = 1e9
        lane = None
        for data in lanes:
            points = data.centerline.points
            for i in range(1, len(points)):
                point1 = [points[i - 1].x, points[i - 1].y]
                point2 = [points[i].x, points[i].y]
                l = compute_distance_point2Seg(point1, point2, point)
                if l < min_dist:
                    min_dist = l
                    lane = data
        #print("min_dist: ", min_dist)
        if lane != None:
            width = self.getlaneWidth(lane)
            if min_dist > width / 2 + 0.2:  # max_land_width/2
                lane = None
        return lane

    # get near lane around ego car
    def get_lane_around_ego_car(self, position, lims, heading):
        xlim, ylim, zlim = lims
        extend = 1.0
        scope = (xlim[0] - extend, ylim[0] - extend, xlim[1] + extend, ylim[1] + extend)

        rot_mat = np.array([math.cos(heading), -math.sin(heading), \
                            math.sin(heading), math.cos(heading)]).reshape((2, 2))
        left, bottom, right, top = scope
        rect = np.array([[left, bottom], [right, bottom], \
                         [right, top], [left, top]])
        rot_rect = np.dot(rect, np.transpose(rot_mat))
        min_pt = rot_rect.min(axis=0) + np.array(position[:2])
        max_pt = rot_rect.max(axis=0) + np.array(position[:2])
        roi = (min_pt[0], min_pt[1], max_pt[0], max_pt[1])
        near_lanes = self.rtreeSearch(roi, 'LANE')

        return near_lanes

    def get_lane_id_by_point(self, point, lanes=None):
        if lanes is not None:
            lane = self.get_lane_by_point_v2(point, lanes)
        else:
            lane = self.get_lane_by_point(point)
        if lane is not None:
            return lane.id
        return -1


    # get lane boundary by id
    def get_lane_boundary_by_id(self, id):
        for data in self.map_items['LANE_BOUNDARY']:
            if data.id == id:
                return data
        return None

    # get lane by id
    def get_lane_by_id(self, id):
        for data in self.map_items['LANE']:
            if data.id == id:
                return data
        return None

    # get successor_id for each lane in a roi region
    def get_roi_successor_predecessor_id(self, ego_car_lane, ids):
        lane_queue = queue.Queue()

        lane_queue.put(ego_car_lane)
        target_ids = [ego_car_lane.id]
        count = 0
        while lane_queue.empty() == False:
            lane = lane_queue.get()
            count += 1
            successor_ids = list(lane.successor_id)
            for id in successor_ids:
                if (id in ids) and (id not in target_ids):
                    target_ids.append(id)

                lane_next = self.get_lane_by_id(id)
                if lane_next is not None:
                    lane_queue.put(lane_next)

            if count > 20:
                while lane_queue.empty() == False:
                    lane_queue.get()

        lane_queue.put(ego_car_lane)
        count = 0
        while lane_queue.empty() == False:
            lane = lane_queue.get()
            count += 1
            predecessor_ids = list(lane.predecessor_id)
            for id in predecessor_ids:
                if (id in ids) and (id not in target_ids):
                    target_ids.append(id)

                lane_pre = self.get_lane_by_id(id)
                if lane_pre is not None:
                    lane_queue.put(lane_pre)

            if count > 20:
                while lane_queue.empty() == False:
                    lane_queue.get()

        return target_ids

    # get successor_id for each lane
    def get_successor_id(self):
        lane_id = {}
        for data in self.map_items['LANE']:
            lane_id[data.id] = []
            lane_queue = queue.Queue()
            lane_queue.put(data)
            while lane_queue.empty() == False:
                lane = lane_queue.get()
                successor_ids = list(lane.successor_id)
                for id in successor_ids:
                    if id in lane_id[data.id]:
                        continue
                    lane_id[data.id].append(id)
                    lane_next = self.get_lane_by_id(id)
                    if lane_next is not None:
                        lane_queue.put(lane_next)
        return lane_id

    ## kdtree
    def initKdtrees(self):
        kdtrees = {}
        for item_type, items in self.map_items.items():
            all_points = []
            for item in items:
                if item_type == 'LANE':
                    points = item.centerline.points
                elif item_type == 'LANE_BOUNDARY':
                    points = item.boundary.points
                elif item_type == 'STOP_LINE':
                    points = item.stop_line.points
                elif item_type == 'CROSS_WALK':
                    points = item.polygon.points
                elif item_type == 'JUNCTION':
                    points = item.polygon.points
                else:
                    print("Error: unsupport map item type!")
                    sys.exit(-1)
                points = np.array([[pt.x, pt.y] for pt in points])
                mean_pt = points.mean(axis=0)
                all_points.append(myKdtree.Item(mean_pt[0], mean_pt[1], item))
            kdtrees[item_type] = myKdtree.create(all_points)
            # kdtree.visualize(self.kdtree[item_type])
        return kdtrees

    def kdtreeSearch(self, point, k, item_type='LANE'):
        res_tupe = self.kdtrees[item_type].search_knn(point, k)
        return [res[0].data.data for res in res_tupe]

    ## rtree
    def initRtrees(self):
        rtrees = {}
        rtrees_objects = {}
        for item_type, items in self.map_items.items():
            rtrees[item_type] = rtree.index.Index()
            rtrees_objects[item_type] = {}
            for item in items:
                if item_type == 'LANE':
                    points = item.centerline.points
                elif item_type == 'LANE_BOUNDARY':
                    points = item.boundary.points
                elif item_type == 'STOP_LINE':
                    points = item.stop_line.points
                elif item_type == 'CROSS_WALK':
                    points = item.polygon.points
                elif item_type == 'JUNCTION':
                    points = item.polygon.points
                else:
                    print("Error: unsupport map item type!")
                    sys.exit(-1)
                points = np.array([[pt.x, pt.y] for pt in points])
                min_pt = points.min(axis=0)
                max_pt = points.max(axis=0)
                # [xmin, ymin, xmax, ymax] left, bottom, right, top
                bbox = (min_pt[0] - 0.1, min_pt[1] - 0.1, max_pt[0], max_pt[1])
                rtrees[item_type].insert(item.id, bbox)
                rtrees_objects[item_type][item.id] = item
        return rtrees, rtrees_objects

    def rtreeSearch(self, roi, item_type='LANE'):
        return [self.rtrees_objects[item_type][n] for n in self.rtrees[item_type].intersection(roi)]


    ### common
    def getNearMap(self, position, scope, heading, need_types=None):
        need_types = need_types if need_types is not None else self.map_items.keys()
        map_items = {}
        rot_mat = np.array([math.cos(heading), -math.sin(heading), \
                            math.sin(heading), math.cos(heading)]).reshape((2, 2))
        left, bottom, right, top = scope
        rect = np.array([[left, bottom], [right, bottom], \
                         [right, top], [left, top]])
        rot_rect = np.dot(rect, np.transpose(rot_mat))
        for item_type in need_types:
            if self.use_rtree:
                min_pt = rot_rect.min(axis=0) + np.array(position[:2])
                max_pt = rot_rect.max(axis=0) + np.array(position[:2])
                roi = (min_pt[0], min_pt[1], max_pt[0], max_pt[1])
                map_items[item_type] = self.rtreeSearch(roi, item_type)
            else:
                knn = np.linalg.norm(rot_rect, axis=0).max() + 2
                map_items[item_type] = self.kdtreeSearch(position, knn, item_type)
        return map_items

    def drawNearMap(self, position, lims, heading, img_size, \
                    need_types=None, transformer=None, bg_img=None):
        xsize, ysize, zsize = bg_img.shape if bg_img is not None else img_size
        xlim, ylim, zlim = lims
        resolution_inv = np.array([1.0 * xsize / (xlim[1] - xlim[0]), \
                                   1.0 * ysize / (ylim[1] - ylim[0])])
        bg_img = bg_img if bg_img is not None else np.zeros(img_size, dtype=np.uint8)
        extend = 1.0
        scope = (xlim[0] - extend, ylim[0] - extend, xlim[1] + extend, ylim[1] + extend)
        map_items = self.getNearMap(position, scope, heading, need_types)
        for render_type in self.renderOrder:
            item_type = "LANE" if render_type == 'BG_LANE' else render_type
            if item_type not in map_items.keys():
                continue
            is_polygon = False
            for item in map_items[item_type]:
                if render_type == 'CROSS_WALK' or render_type == 'JUNCTION':
                    points = np.array([[pt.x, pt.y] for pt in item.polygon.points])
                    is_polygon = True
                elif render_type == 'BG_LANE':
                    left_boundary = copy.deepcopy(self.lane_boundaries[item.left_boundary_id])
                    right_boundary = self.lane_boundaries[item.right_boundary_id]
                    if item.boundary_direction == 0 or item.boundary_direction == 3:
                        left_boundary.reverse()
                    left_boundary.extend(right_boundary)
                    points = np.array(left_boundary)
                    is_polygon = True
                elif render_type == 'LANE':
                    points = np.array([[pt.x, pt.y] for pt in item.centerline.points])
                elif render_type == 'LANE_BOUNDARY':
                    points = np.array([[pt.x, pt.y] for pt in item.boundary.points])
                elif render_type == 'STOP_LINE':
                    points = np.array([[pt.x, pt.y] for pt in item.stop_line.points])
                else:
                    print("Error: unsupport map item type!")
                    sys.exit(-1)
                if transformer is None:
                    points = points - np.array(position[:2])
                else:
                    points = np.concatenate((points, position[2] * np.ones((points.shape[0], 1))), axis=1)
                    points = transformer(points)[:, 0:2]
                color = self.map_item_colors[render_type]
                # if render_type =='LANE':
                #     angle = math.atan2(points[-1][1] - points[0][1], points[-1][0] - points[0][0])
                #     angle = max(0, min(360, (angle + math.pi) * 180 / math.pi))
                #     color = hsv2rgb(angle, 1, 1)
                points = (points - np.array([xlim[0], ylim[0]])) * resolution_inv
                points = np.array(points).astype(int)
                # self.map_item_colors[item_type] = np.random.randint(20, 255, (1, 3))[0]
                if is_polygon:
                    cv2.fillConvexPoly(bg_img, points, color)
                else:
                    points = points.reshape((-1, 1, 2))
                    cv2.polylines(bg_img, [points], False, color,
                                  self.map_item_thickness[render_type])
        return bg_img

    def drawNearMapSlice(self, position, lims, heading, img_size, \
                         need_types=None, transformer=None):
        xsize, ysize, zsize = img_size
        xlim, ylim, zlim = lims
        resolution_inv = np.array([1.0 * xsize / (xlim[1] - xlim[0]), \
                                   1.0 * ysize / (ylim[1] - ylim[0])])
        extend = 1.0
        scope = (xlim[0] - extend, ylim[0] - extend, xlim[1] + extend, ylim[1] + extend)
        map_items = self.getNearMap(position, scope, heading, need_types)
        map_layers = []
        layer_names = []
        for render_type in self.renderOrder:
            item_type = "LANE" if render_type == 'BG_LANE' else render_type
            if item_type not in map_items.keys():
                continue
            is_polygon = False
            img = np.zeros((xsize, ysize), dtype=np.float32) - 1
            for item in map_items[item_type]:
                if render_type == 'CROSS_WALK' or render_type == 'JUNCTION':
                    points = np.array([[pt.x, pt.y] for pt in item.polygon.points])
                    is_polygon = True
                elif render_type == 'BG_LANE':
                    left_boundary = copy.deepcopy(self.lane_boundaries[item.left_boundary_id])
                    right_boundary = self.lane_boundaries[item.right_boundary_id]
                    if item.boundary_direction == 0 or item.boundary_direction == 3:
                        left_boundary.reverse()
                    left_boundary.extend(right_boundary)
                    points = np.array(left_boundary)
                    is_polygon = True
                elif render_type == 'LANE':
                    points = np.array([[pt.x, pt.y] for pt in item.centerline.points])
                elif render_type == 'LANE_BOUNDARY':
                    points = np.array([[pt.x, pt.y] for pt in item.boundary.points])
                elif render_type == 'STOP_LINE':
                    points = np.array([[pt.x, pt.y] for pt in item.stop_line.points])
                else:
                    print("Error: unsupport map item type!")
                    sys.exit(-1)
                if transformer is None:
                    points = points - np.array(position[:2])
                else:
                    points = np.concatenate((points, position[2] * np.ones((points.shape[0], 1))), axis=1)
                    points = transformer(points)[:, 0:2]
                color = 1
                points = (points - np.array([xlim[0], ylim[0]])) * resolution_inv
                points = np.array(points).astype(int)
                if is_polygon:
                    cv2.fillConvexPoly(img, points, color)
                else:
                    points = points.reshape((-1, 1, 2))
                    cv2.polylines(img, [points], False, color,
                                  self.map_item_thickness[render_type])
            map_layers.append(img)
            layer_names.append(render_type)
        return map_layers, layer_names

    def MergeMapLayersToRgb(self, map_layers, layer_names):
        colors = [(0, 0, 0)]
        ids = np.zeros((map_layers[0].shape[0], map_layers[0].shape[1]), dtype=np.uint8)
        for i, render_type in enumerate(layer_names):
            color = self.map_item_colors[render_type]
            colors.append(color)
            ids[map_layers[i] > 0] = i + 1
        colors = np.array(colors)
        rgb_img = np.zeros((map_layers[0].shape[0], map_layers[0].shape[1], 3), dtype=np.uint8)
        rgb_img[:, :] = colors[ids[:, :]]
        return rgb_img

    def getMapSpaceRange(self):
        min_x, max_x = 1e6, -1e6
        min_y, max_y = 1e6, -1e6
        for lane in self.map_items['LANE']:
            points = lane.centerline.points
            for pt in points:
                min_x = min(pt.x, min_x)
                max_x = max(pt.x, max_x)
                min_y = min(pt.y, min_y)
                max_y = max(pt.y, max_y)
        return min_x, max_x, min_y, max_y


if __name__ == "__main__":
    roadmap_file = "/home/deeproute/dataset/prediction_data/map/longhua_map_0327.bin"
    roadMap = RoadMap(roadmap_file, index_method="rtree")

    # (-665.94566, 220.235488, 78.40504), heading: -2.582084
    # (460.643496, -913.016721, 67.094252), heading: 0.026714311866119292
    position = (460.643496, -913.016721, 67.094252)
    heading = 0.026714311866119292
    xlim = [-64, 64]
    ylim = [-64, 64]
    zlim = [-4, 3]
    lims = [xlim, ylim, zlim]
    sizes = [640, 640, 3]

    # print(roadMap.roadMap)
    # sizes = [1600, 1600, 3]
    # lims = [[-2000, 2000], [-2000, 2000], []]
    # position = [0.5 * (lims[0][0] + lims[0][1]), 0.5 * (lims[1][0] + lims[1][1])]

    # image = roadMap.drawNearMap(position, lims, heading, sizes)
    # cv2.imshow("image", image)
    # cv2.waitKey()

    t = time.time()
    map_layers, layer_names = roadMap.drawNearMapSlice(position, lims, heading, sizes)
    print('draw map ', time.time() - t)
    for i, layer in enumerate(map_layers):
        show_img = (layer + 1) * 100
        cv2.imshow(layer_names[i], show_img.astype(np.uint8))
        cv2.waitKey()

    rgb_img = roadMap.MergeMapLayersToRgb(map_layers, layer_names)
    cv2.imshow("image", rgb_img)
    cv2.waitKey()