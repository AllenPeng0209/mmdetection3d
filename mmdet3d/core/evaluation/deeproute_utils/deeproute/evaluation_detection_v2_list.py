import sys
sys.path.append('..')
import numpy.ma as ma
import itertools
from pathlib import Path
from matplotlib import pyplot as plt
from shapely.geometry.polygon import Polygon
from protos import detection_eval_pb2
import torch
import iou_cuda
from core.occl_dist_assign import get_max_angle_points, occlusion_relation, filter_d, filter_d_v2
from deeproute.compute_occl_ignore import *

def get_occlusion_dt(boxesA, boxesB, occlusion_thresh, fp):
    """
    :param boxesA: array, shape = [M, 4, 2], where M is the number of boxes in the
                     current frame, [4, 2] stands for the coordinates of four corners for each box
    :param boxesB: refer to boxesA
    :param occlusion_thresh: float, the threshold of occlusion ratio, 1/3 is selected here
    :return: occlusion: array, shape = [M, 1], where where M is the number of boxes in the
                        current frame, 0 stands for occlusion
    """
    N = boxesA.shape[0]
    M = boxesB.shape[0]
    occlusion = np.ones(N, dtype=bool)
    for i in range(N):
        if fp[i] == 0:
            continue
        points1, angle1 = get_max_angle_points(boxesA[i])
        out = True
        for j in range(M):
            points2, angle2 = get_max_angle_points(boxesB[j])
            points, union_angle = get_max_angle_points(points1 + points2)
            intersect_angle = angle1 + angle2 - union_angle
            out = out and occlusion_relation(boxesA[i], boxesB[j], angle1, intersect_angle, occlusion_thresh)
            if not out:
                break
        occlusion[i] = out
    return occlusion.astype(int)

class ObjStat():
    def __init__(self,
                 gt_loc,
                 dt_loc,
                 gt_boxes,
                 dt_boxes,
                 gt_names,
                 dt_names,
                 gt_occlusion,
                 gt_ignore,
                 distance_thresh,
                 distance_compute_model,
                 occlusion_thresh,
                 name_to_label,
                 interest_labels,
                 thresh_map,
                 cal_iou_bev):
        self._gt_loc = gt_loc
        self._dt_loc = dt_loc
        self._gt_boxes = gt_boxes
        self._dt_boxes = dt_boxes
        self._gt_names = gt_names
        self._dt_names = dt_names
        self._gt_occlusion = gt_occlusion
        self._cal_iou_bev = cal_iou_bev
        self._gt_dist = []
        distance_thresh = sorted(distance_thresh)
        self._distance_thresh = distance_thresh
        self._distance_compute_model = distance_compute_model
        self._occlusion_thresh = occlusion_thresh
        self._gt_ignore = gt_ignore
        self._name_to_label = name_to_label
        self._interest_labels = interest_labels
        self._thresh_map = thresh_map
        self._tracking = None
        self._gt_num = self._gt_boxes.shape[0]
        self._dt_num = self._dt_boxes.shape[0]
        self._filter_to_use = [(self.name_to_label_filter, self.name_to_label_filter, 8),
                              (self.miss_filter, self.fp_filter, 2),
                              (self.distance_filter, self.distance_filter, 4),
                              (self.occlusion_filter, self.occlusion_filter, 2)]

        self._gt_corners = box_ops.get_corners(gt_boxes[:, :2], gt_boxes[:, 3:5], gt_boxes[:, 6]) if self._gt_num > 0 else np.array([])
        self._dt_corners = box_ops.get_corners(dt_boxes[:, :2], dt_boxes[:, 3:5], dt_boxes[:, 6]) if self._dt_num > 0 else np.array([])
        self._iou = self.get_iou_bev() if cal_iou_bev else self.get_iou_3D()

        self._missing = None
        self._fp = None
        self._tp = None
        self._ignore_num_per_frame = [np.array([0]*((len(distance_thresh)+1)*2)) for i in range(len(interest_labels))]

    def get_iou_bev(self):
        if self._gt_num <= 0 or self._dt_num <= 0:
            return None

        iou = iou_cuda.iou_forward(torch.cuda.FloatTensor(self._gt_corners), torch.cuda.FloatTensor(self._dt_corners))
        iou = iou.cpu().numpy().reshape(self._gt_num, self._dt_num)

        return iou

    def get_iou_3D(self):
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
        if self._gt_num <= 0 or self._dt_num <= 0:
            return None
        iou = np.zeros([self._gt_num, self._dt_num], dtype=np.float32)
        for i in range(self._gt_num):
            for j in range(self._dt_num):
                iw = (min(self._gt_boxes[i, 2] + self._gt_boxes[i, 5],
                          self._dt_boxes[j, 2] + self._dt_boxes[j, 5]) - max(self._gt_boxes[i, 2], self._dt_boxes[j, 2]))
                if iw > 0:
                    p1 = Polygon(self._gt_corners[i])
                    p2 = Polygon(self._dt_corners[j])
                    # first get the intersection of the undersides, then times it with the min height
                    inc = p1.intersection(p2).area * iw
                    if inc > 0:
                        iou[i, j] = inc / (p1.area * self._gt_boxes[i, 5] +
                                           p2.area * self._dt_boxes[j, 5] - inc)
        #print(iou)
        return iou

    def get_match_pair(self):
        if self._gt_num <= 0 and self._dt_num <= 0:
            return []
        if self._gt_num > 0 and self._dt_num <= 0:
            self._missing = np.ones(self._gt_num)
            for i in range(self._gt_num):
                if self._gt_ignore[i]:
                    self._missing[i] = 0
            return []
        if self._gt_num <= 0 and self._dt_num > 0:
            self._fp = np.ones(self._dt_num)
            return []

        missing = np.ones(self._gt_num)
        tp = np.zeros(self._gt_num)
        fp = np.ones(self._dt_num)
        thresh = np.array([self._thresh_map[self._name_to_label[n.lower()]] for n in self._gt_names])
        match_pair = []
        if self._tracking:
            for i in range(self._gt_num):
                if self._gtIds[i] in self._m:
                    prev_dt = self._m[self._gtIds[i]]
                    index, = np.where(self._dtIds == prev_dt)
                    if index.shape[0] != 0:
                        j = index[0]
                        if self._iou[i, j] >= thresh[i]:
                            match_pair.append([i, j])
                            missing[i] = 0
                            tp[i] = 1
                            fp[j] = 0
                            self._iou[index[0]] = ma.masked
                            self._iou[:, index[1]] = ma.masked
        for i in range(self._gt_num):
            maximum = np.max(self._iou)
            if maximum > 0:
                index = np.where(self._iou == maximum)
                if(index[0].size > 1):
                    index = (index[0][0], index[1][0])
                self._iou[index] = ma.masked
                if maximum >= thresh[index[0]] and self._gt_labels[index[0]] == self._dt_labels[index[1]]:
                    match_pair.append(index)
                    missing[index[0]] = 0
                    fp[index[1]] = 0
                    self._iou[index[0]] = ma.masked
                    self._iou[:, index[1]] = ma.masked

        # delete missing samples with ignore attribute
        for i in range(self._gt_num):
            if self._gt_ignore[i]:
                missing[i] = 0
                tp[i] = 0

        self._missing = missing
        self._fp = fp
        self._tp = tp
        return match_pair

    def stat(self, n_split_list, filter_split_list):
        if any(x is None for x in filter_split_list):
            return None, None
        multiplier = np.insert(np.cumprod(n_split_list, dtype=int)[0:-1], 0, 1)
        comb_hash = np.sum(np.stack(filter_split_list, axis=-1) * multiplier, axis=-1)
        unique, inv, count = np.unique(comb_hash, return_inverse=True, return_counts=True)
        idx_list = [np.where(inv == i)[0] for i in range(unique.shape[0])]
        unique_dehash = np.mod((unique.reshape(-1, 1) // multiplier), np.array(n_split_list)).astype(int)
        unique_dehash_list_of_tuple = [tuple(row) for row in unique_dehash]
        comb_stat = dict(zip(unique_dehash_list_of_tuple, count))
        comb_idx = dict(zip(unique_dehash_list_of_tuple, idx_list))
        return comb_stat, comb_idx

    def compute_stat(self):
        gt_filter_split_list = []
        dt_filter_split_list = []
        n_split_list = []
        for f1, f2, n_opt in self._filter_to_use:
            gt_filter_split_list.append(f1(1))
            dt_filter_split_list.append(f2(2))
            n_split_list.append(n_opt)
        self._gt_comb_stat, self._gt_comb_idx = self.stat(n_split_list, gt_filter_split_list)
        self._dt_comb_stat, self._dt_comb_idx = self.stat(n_split_list, dt_filter_split_list)

    def name_to_label_filter(self, i):
        if i == 1 and self._gt_num > 0:
            self._gt_labels = np.array([self._name_to_label[n.lower()] for n in self._gt_names])
            return self._gt_labels
        elif i == 2 and self._dt_num > 0:
            self._dt_labels = np.array([self._name_to_label[n.lower()] for n in self._dt_names])
            return self._dt_labels
        return None

    def miss_filter(self, i):
        """
        filter match,missing
        correspoding 0,1
        """
        self._match_pairs = self.get_match_pair()
        return self._missing

    def fp_filter(self, i):
        """
        filter match,fp
        corresponding 0,1
        """
        return self._fp

    def ignore_filter(self):
        for i in range(self._gt_num):
            if not self._gt_ignore[i]:
                continue
            if self._name_to_label[self._gt_names[i].lower()] >= len(self._interest_labels):
                continue
            for j in range(self._gt_dist[i]*2, (len(self._distance_thresh)+1)*2, 2):
                self._ignore_num_per_frame[self._name_to_label[self._gt_names[i].lower()]][j] += 1
            if self._gt_occlusion[i] == 0:   # 0: occlusion
                continue
            for j in range(self._gt_dist[i] * 2+1, (len(self._distance_thresh) + 1) * 2, 2):
                self._ignore_num_per_frame[self._name_to_label[self._gt_names[i].lower()]][j] += 1

    def distance_filter(self, i):
        """
        e.g., self._distance_thresh = [20, 40, 80]
        partition the bounding box to [0, 20], [20, 40], [40, 80], [80, ]
        corresponding to 0,1,2,3

        Args:
            corners=box_np_ops.center_to_corner_box2d(
                     boxes[:, :2], boxes[:, 3:5], boxes[:, 6])
        """
        dist = []
        if len(self._gt_dist) == 0 and i == 1 and self._gt_num > 0:
            for j in range(self._gt_num):
                dist_tmp = filter_d(self._gt_corners[j], self._distance_thresh) if self._distance_compute_model == 0 \
                    else filter_d_v2(self._gt_loc[j], self._distance_thresh)
                dist.append(dist_tmp)
            self._gt_dist = dist

            return dist

        elif len(self._gt_dist) != 0 and i == 1 and self._gt_num > 0:
            return self._gt_dist

        elif i == 2 and self._dt_num > 0:
            for j in range(self._dt_num):
                dist_tmp = filter_d(self._dt_corners[j], self._distance_thresh) if self._distance_compute_model == 0 \
                    else filter_d_v2(self._dt_loc[j], self._distance_thresh)
                dist.append(dist_tmp)
            if len(self._match_pairs) > 0:
                for p in self._match_pairs:
                    if self._gt_dist[int(p[0])] != dist[int(p[1])]:
                        dist[int(p[1])] = self._gt_dist[int(p[0])]
            return dist
        return None

    def occlusion_filter(self, i):
        if len(self._gt_occlusion) == 0 and i == 1 and self._gt_num > 0:
            self._gt_occlusion = get_occlusion(self._gt_corners, self._gt_corners, self._occlusion_thresh)
            return self._gt_occlusion
        elif len(self._gt_occlusion) != 0 and i == 1 and self._gt_num > 0:
            return self._gt_occlusion
        elif i == 2 and self._dt_num > 0:
            dt_occlusion = get_occlusion_dt(self._dt_corners, self._gt_corners, self._occlusion_thresh, self._fp)
            if len(self._match_pairs) > 0:
                for p in self._match_pairs:
                    if self._gt_occlusion[int(p[0])] != dt_occlusion[int(p[1])]:
                        dt_occlusion[int(p[1])] = self._gt_occlusion[int(p[0])]
            return dt_occlusion
        return None

    def query_idx(self, query_list, i):
        comb_query_list = itertools.product(*query_list)
        temp = []
        if i == 1 and self._gt_comb_idx != None:
            temp = [self._gt_comb_idx.get(x)
                    for x in comb_query_list if x in self._gt_comb_idx]
        elif i == 2 and self._dt_comb_idx != None:
            temp = [self._dt_comb_idx.get(x)
                    for x in comb_query_list if x in self._dt_comb_idx]
        if temp != []:
            temp = np.concatenate(temp)
        return temp

    def query_count(self, query_list, i):
        """
        i=1 get the info of gt boxes, if gt_boxes=none,return 0 because nothing missing
        i=2 get the info of dt boxes, if dt_boxes =none, return 0 because no false positive
        """
        # print("query_list: {}".format(query_list))
        comb_query_list = itertools.product(*query_list)
        # for x in comb_query_list:
        # print("comb_query_list_x: {}".format(x))
        if i == 1 and self._gt_comb_stat != None:
            return sum([self._gt_comb_stat.get(x, 0) for x in comb_query_list])
        elif i == 2 and self._dt_comb_stat != None:
            return sum([self._dt_comb_stat.get(x, 0) for x in comb_query_list])
        return 0

def get_eval(gt_loc, dt_loc, gt_boxes, dt_boxes, gt_names, dt_names, timeStamp, gt_occlusion, gt_ignore):
    objstat = ObjStat(gt_loc, dt_loc, gt_boxes, dt_boxes, gt_names, dt_names, gt_occlusion, gt_ignore, distance_thresh,
                      distance_compute_model, occlusion_thresh, name_to_label, interest_labels, thresh_map, cal_iou_bev)

    objstat.compute_stat()
    objstat.ignore_filter()

    eval_results_per_frame = {label: {key: np.zeros(4) for key in query_lists.keys()} for label in interest_labels}
    eval_info_per_frame = {key: {} for key in query_lists.keys()}
    eval_info_per_class_per_frame = {label: {key: {} for key in query_lists.keys()} for label in interest_labels}

    for key in eval_info_per_frame.keys():
        missing = 0
        fp = 0
        missing_boxes = []
        fp_boxes = []
        for l in interest_labels:
            query_list = [[name_to_label.get(l)]] + query_lists.get(key)
            query_distance_list = [[name_to_label.get(l)]] + query_distance_lists.get(key)
            label_missing = objstat.query_count(query_list, 1)
            label_fp = objstat.query_count(query_list, 2)
            missing_boxes += [x.item() for x in objstat.query_idx(query_list, 1)]
            fp_boxes += [x.item() for x in objstat.query_idx(query_list, 2)]
            label_gt = objstat.query_count(query_distance_list, 1)
            label_dt = objstat.query_count(query_distance_list, 2)
            eval_results_per_frame[l][key] += [label_missing, label_fp, label_gt, label_dt]
            missing += label_missing
            fp += label_fp

            if label_missing >0 or label_fp > 0:
                eval_info_per_class_per_frame[l][key][timeStamp] = {'missing_boxes': [x.item() for x in objstat.query_idx(query_list, 1)],\
                                                             'fp_boxes': [x.item() for x in objstat.query_idx(query_list, 2)]}

        if missing > 0 or fp > 0:
            eval_info_per_frame[key][timeStamp] = {'missing_boxes': missing_boxes, 'fp_boxes': fp_boxes}

    return (timeStamp, objstat._ignore_num_per_frame, eval_results_per_frame, eval_info_per_frame, eval_info_per_class_per_frame)



def get_query_list(distance_thresh):
    distance_thresh = distance_thresh
    namespace = []
    # for thresh in distance_thresh:
    #     namespace += ['within_' + str(thresh)]
    #     namespace += ['within_' + str(thresh) + '_rm_occluded']
    # namespace += ['full_range', 'full_range_rm_occluded']
    # query_lists = {name: [[1]] for name in namespace}
    # query_distance_lists = {name: [[0, 1]] for name in namespace}
    # for i, thresh in enumerate(distance_thresh):
    #     query_lists['within_'+str(thresh)].append(np.arange(i+1).tolist())
    #     query_lists['within_'+str(thresh)].append([0, 1])
    #     query_lists['within_' + str(thresh) + '_rm_occluded'].append(np.arange(i+1).tolist())
    #     query_lists['within_' + str(thresh) + '_rm_occluded'].append([1])
    #     query_distance_lists['within_' + str(thresh)].append(np.arange(i+1).tolist())
    #     query_distance_lists['within_' + str(thresh)].append([0, 1])
    #     query_distance_lists['within_' + str(thresh) + '_rm_occluded'].append(np.arange(i+1).tolist())
    #     query_distance_lists['within_' + str(thresh) + '_rm_occluded'].append([1])
    
    for i in range(len(distance_thresh)):
        prev = distance_thresh[i-1] if i > 0 else "0"
        namespace += [str(prev) + "-" + str(distance_thresh[i])]
        namespace += [str(prev) + "-" + str(distance_thresh[i]) + '_rm_occluded']
    namespace += ['full_range', 'full_range_rm_occluded']
    query_lists = {name: [[1]] for name in namespace}
    query_distance_lists = {name: [[0, 1]] for name in namespace}
    for i, thresh in enumerate(distance_thresh):
        prev = distance_thresh[i-1] if i > 0 else "0"
        query_lists[str(prev) + "-" + str(distance_thresh[i])].append(np.arange(i+1).tolist())
        query_lists[str(prev) + "-" + str(distance_thresh[i])].append([0, 1])
        query_lists[str(prev) + "-" + str(distance_thresh[i]) + '_rm_occluded'].append(np.arange(i+1).tolist())
        query_lists[str(prev) + "-" + str(distance_thresh[i]) + '_rm_occluded'].append([1])
        query_distance_lists[str(prev) + "-" + str(distance_thresh[i])].append(np.arange(i+1).tolist())
        query_distance_lists[str(prev) + "-" + str(distance_thresh[i])].append([0, 1])
        query_distance_lists[str(prev) + "-" + str(distance_thresh[i]) + '_rm_occluded'].append(np.arange(i+1).tolist())
        query_distance_lists[str(prev) + "-" + str(distance_thresh[i]) + '_rm_occluded'].append([1])

    query_lists['full_range'].append(np.arange(len(distance_thresh)+1).tolist())
    query_lists['full_range'].append([0, 1])
    query_lists['full_range_rm_occluded'].append(np.arange(len(distance_thresh)+1).tolist())
    query_lists['full_range_rm_occluded'].append([1])
    query_distance_lists['full_range'].append(np.arange(len(distance_thresh)+1).tolist())
    query_distance_lists['full_range'].append([0, 1])
    query_distance_lists['full_range_rm_occluded'].append(np.arange(len(distance_thresh)+1).tolist())
    query_distance_lists['full_range_rm_occluded'].append([1])

    return query_lists, query_distance_lists

def create_file(eval_path, eval_info, eval_info_per_class, eval_results, distance_thresh, ignore_num, name_to_label):
    for label in eval_info_per_class.keys():
        for key, value in eval_info_per_class[label].items():
            filename = Path(eval_path) / (label+"_"+key + '.pkl')
            with open(filename, 'wb') as fp:
                pickle.dump(value, fp)

    for key, value in eval_info.items():
        filename = Path(eval_path)/("total_"+key+'.pkl')
        with open(filename, 'wb') as fp:
            pickle.dump(value, fp)
    generate_table(eval_path, eval_results, distance_thresh, ignore_num, name_to_label)

def generate_table(eval_path, eval_results, distance_thresh, ignore_num, name_to_label):
    np.seterr(divide='ignore', invalid='ignore')
    total_data = np.zeros((len(distance_thresh)*2+2, 4))
    rows = []
    for l in eval_results.keys():
        rows = [k for k in eval_results[l].keys()]
        data = np.asarray([v for v in eval_results[l].values()]).reshape(len(distance_thresh)*2+2, 4)
        total_data += data
        draw_table(rows, eval_path, l, data, ignore_num[name_to_label[l.lower()]], distance_thresh)

    total_ignore_num = ignore_num[0]
    for i in range(1, len(ignore_num)):
        total_ignore_num += ignore_num[i]
    draw_table(rows, eval_path, "total", total_data, total_ignore_num, distance_thresh)

def draw_table(rows, eval_path, l, rdata, ignore_num, distance_thresh):
    # if l == "cone":
    #     print(data[:, 1])
    #     print(data[:, 2])
    #     print(ignore_num)
    import copy
    data = copy.deepcopy(rdata)
    data[:,2] = data[:, 2] - ignore_num
    tp_num = data[:, 2] - data[:, 0]
    data[:, 3] = tp_num + data[:, 1]

    for i in range(5, 1, -1):
        data[i, :] = data[i, :] - data[i-2, :]

    missing_rate = np.round(1-data[:, 0]/data[:, 2], decimals=4).reshape(len(distance_thresh)*2+2, 1)
    fp_rate = np.round(1-data[:, 1]/data[:, 3], decimals=4).reshape(len(distance_thresh)*2+2, 1)
    data = np.append(data,  missing_rate, axis=1)
    data = np.append(data, fp_rate, axis=1)
    columns = ["missing", "fp", "gt", "dt", "recall", "precision"]
    the_table = plt.table(cellText=data,
                          colWidths=[0.12] * 6,
                          rowLabels=rows,
                          colLabels=columns,
                          loc='center')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(24)
    the_table.scale(4, 4)
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
    for pos in ['right', 'top', 'bottom', 'left']:
        plt.gca().spines[pos].set_visible(False)
    filename = l+"_eval_result.png"
    plt.savefig(Path(eval_path)/filename, bbox_inches='tight', pad_inches=0.05)

def loadConfig(config_path):
    if isinstance(config_path, str):
        config = detection_eval_pb2.DeeprouteDetectionEvalConfig()
        with open(config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, config)
    else:
        config = config_path

    thresh_map = config.deeproute_eval_input_reader.overlap_thresh
    interest_labels = config.deeproute_eval_input_reader.interest_labels
    distance_thresh = config.deeproute_eval_input_reader.distance_thresh
    distance_compute_model = config.deeproute_eval_input_reader.distance_compute_model
    occlusion_thresh = config.deeproute_eval_input_reader.occlusion_thresh
    num_features_for_pc = int(config.deeproute_eval_input_reader.num_features_for_pc)
    filter_points_number = list(config.deeproute_eval_input_reader.filter_points_number)
    cal_iou_bev = config.deeproute_eval_input_reader.cal_iou_bev
    name_to_label = {}
    for sample_group in config.deeproute_eval_input_reader.name_to_label.sample_groups:
        name_to_label.update(dict(sample_group.name_to_num))

    return thresh_map, interest_labels, distance_thresh, distance_compute_model, occlusion_thresh, \
           num_features_for_pc, filter_points_number, cal_iou_bev, name_to_label

def compute_eval(config_path, eval_path, info_path, det_path, pcd_path, save_path, gt_occl_path, ignore_path, pool_num):
    global thresh_map, interest_labels, distance_thresh, distance_compute_model, occlusion_thresh
    global num_features_for_pc, filter_points_number, cal_iou_bev, name_to_label
    global query_lists, query_distance_lists

    thresh_map, interest_labels, distance_thresh, distance_compute_model, occlusion_thresh, \
    num_features_for_pc, filter_points_number, cal_iou_bev, name_to_label = loadConfig(config_path)

    query_lists, query_distance_lists = get_query_list(distance_thresh)

    # loading occlusion and ignore attributes of  groundtruth and detection
    try:
        with open(gt_occl_path, "rb") as f:
            gt_occlusion_all = pickle.load(f)
            if abs(occlusion_thresh-gt_occlusion_all[0]) > 0.01:
                print("rerun compute_occl_ignore.py due to the change of occlusion_thresh!")
                raise
            gt_occlusion_all = gt_occlusion_all[1]
        with open(ignore_path, "rb") as f:
            gt_ignore_all = pickle.load(f)
            for i in range(len(filter_points_number)):
                if filter_points_number[i] != gt_ignore_all[0][i]:
                    print("rerun compute_occl_ignore.py due to the change of filter_points_number!")
                    raise
            gt_ignore_all = gt_ignore_all[1]
    except:
        print("---Loading occlusion attribute of gt---")
        gt_occlusion_all, gt_ignore_all = compute_occl_ignore(info_path, pcd_path, save_path, config_path, pool_num, True, True)

    assert info_path is not None
    assert det_path is not None
    # info_text_names = get_all_txt_names(info_path)

    assert eval_path is not None
    Path(eval_path).mkdir(parents=True, exist_ok=True)

    pool = multiprocessing.Pool(pool_num)
    res = []
    for frame_id in tqdm(range(len(info_text_names))):

        timeStamp = info_text_names[frame_id]
        # loading gt

        gt_loc, gt_dims, gt_yaws, gt_ids, gt_names = readInfo_per_frame(info_path, timeStamp, name_to_label, interest_labels, False)
        gt_boxes = np.concatenate([gt_loc, gt_dims, gt_yaws[..., np.newaxis]], axis=1)
        box_ops.change_box3d_center_(gt_boxes, src=[0.5, 0.5, 0], dst=[0.5, 0.5, 0.5])

        gt_occlusion = np.array(gt_occlusion_all[timeStamp]) if len(gt_occlusion_all) > 0 else np.array([])
        gt_ignore = np.array(gt_ignore_all[timeStamp]) if len(gt_ignore_all) > 0 else np.array([])

        # loading dt
        det_txt_path = os.path.join(det_path, timeStamp)
        assert os.path.exists(det_txt_path)
        dt_loc, dt_dims, dt_yaws, dt_ids, dt_names = readInfo_per_frame(det_path, timeStamp, name_to_label, interest_labels, False)
        dt_boxes = np.concatenate([dt_loc, dt_dims, dt_yaws[..., np.newaxis]], axis=1)
        box_ops.change_box3d_center_(dt_boxes, src=[0.5, 0.5, 0], dst=[0.5, 0.5, 0.5])

        #get_eval(gt_loc, dt_loc, gt_boxes, dt_boxes, gt_names, dt_names, timeStamp)
        res.append(pool.apply_async(get_eval, (gt_loc, dt_loc, gt_boxes, dt_boxes, gt_names, dt_names, timeStamp, gt_occlusion, gt_ignore)))

    eval_results = {label: {key: np.zeros(4) for key in query_lists.keys()} for label in interest_labels}
    eval_info = {key: {} for key in query_lists.keys()}
    eval_info_per_class = {label: {key: {} for key in query_lists.keys()} for label in interest_labels}
    ignore_num = [np.array([0] * ((len(distance_thresh) + 1) * 2)) for i in range(len(interest_labels))]

    print(len(res))
    for i in tqdm(range(len(res))):
        timeStamp = res[i].get()[0]
        for j in range(len(ignore_num)):
            ignore_num[j] = ignore_num[j] + res[i].get()[1][j]
        for key in eval_info.keys():
            for l in interest_labels:
                eval_results[l][key] += res[i].get()[2][l][key]
                if timeStamp in res[i].get()[4][l][key].keys():
                    eval_info_per_class[l][key][timeStamp] = res[i].get()[4][l][key][timeStamp]
            if timeStamp in res[i].get()[3][key].keys():
                eval_info[key][timeStamp] = res[i].get()[3][key][timeStamp]

    pool.close()
    pool.join()

    create_file(eval_path, eval_info, eval_info_per_class, eval_results, distance_thresh, ignore_num, name_to_label)
    print("Evaluation Done!")


def main():
    list_path = "/media/deeproute/HDISK/dataset/lidar/test.list.20191216"
    info_path = "/media/deeproute/HDISK/dataset/lidar/label"
    pcd_path = "/media/deeproute/HDISK/dataset/lidar/pcd"
    det_path = "/home/deeproute/opt/cnnseg_batch/evalution-long/detections"
    eval_path = "/home/deeproute/opt/cnnseg_batch/evalution-long/results-only"
    gt_occl_path = "/home/deeproute/opt/cnnseg_batch/evalution-long/occl/occlusion_detection.pkl"
    ignore_path = "/home/deeproute/opt/cnnseg_batch/evalution-long/occl/ignore_detection.pkl"
    save_path = "/home/deeproute/opt/cnnseg_batch/evalution-long/occl/"
    config_path = "/home/deeproute/workspace/deeproute/detection_evaluation/configs/detection.long.eval.config"

    list_path = "/media/deeproute/HDISK/dataset/lidar/test.list.20191216"
    list_path = "/media/deeproute/HDISK/dataset/lidar/20200420_long_cone_test.list"
    info_path = "/media/deeproute/HDISK/dataset/lidar/label"
    pcd_path = "/media/deeproute/HDISK/dataset/lidar/pcd"
    save_path = "/home/deeproute/opt/cnnseg_batch/evalution-long/occl"
    config_path = "/home/deeproute/workspace/deeproute/detection_evaluation/configs/detection.eval.config"
    config_path = "/home/deeproute/workspace/deeproute/detection_evaluation/configs/detection.long.eval.config"

    if len(sys.argv) > 1:
        pool_num = int(sys.argv[1])
    else:
        pool_num = 10

    global info_text_names
    with open(list_path) as hd:
        info_text_names = [line.strip() + ".txt" for line in hd.readlines()]
    print("Have {} files".format(len(info_text_names)))

    t = time.time()
    compute_eval(config_path, eval_path, info_path, det_path, pcd_path, save_path, gt_occl_path, ignore_path, pool_num)
    print("The total evaluation time: {}".format(time.time()-t))
    print("All evaluation results are placed in the path: {}".format(eval_path))

if __name__ == '__main__':
    main()
