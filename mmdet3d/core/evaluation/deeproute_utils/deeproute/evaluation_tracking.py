import sys
sys.path.append("..")
from pathlib import Path
import copy, math, matplotlib.pyplot as plt
import torch
import json
import iou_cuda
from core.munkres import Munkres
from deeproute.compute_occl_ignore import *
from protos import tracking_eval_pb2
import core.box_ops as box_ops
from core.occl_dist_assign import angle, filter_d, filter_d_v2
from core.read_info import get_all_txt_names

class tData:
    def __init__(self, frame=-1, obj_type="unset", occlusion=False, distance=-1, track_id=-1, priority=0):

        self._frame = frame
        self._track_id = track_id
        self._obj_type = obj_type
        self._occlusion = occlusion
        self._distance = distance
        self._center = np.array([0, 0, 0])
        self._corners = np.zeros((4, 2))
        self._ignored = False
        self._valid = False
        self._tracker = -1
        self._velocity_confidence = -1
        self._velocity_box = [0, 0, 0]
        self._velocity_ICP = [0, 0, 0]
        self._priority = priority

    def __str__(self):
        attrs = vars(self)
        return '\n'.join("%s: %s" % item for item in attrs.items())

class trackingEvaluation(object):
    def __init__(self, gt_path,
                 dt_path,
                 eval_path,
                 gt_occlusion,
                 dt_occlusion,
                 gt_ignore,
                 gt_velocity,
                 cls,
                 interest_labels,
                 distance_compute_model,
                 dist_thresh,
                 overlap_thresh,
                 occl_thresh,
                 velocity_thresh,
                 name_to_label,
                 has_priority=False):

        self._gt_path = gt_path
        self._dt_path = dt_path
        self._eval_path = eval_path
        self._gt_occlusion = gt_occlusion
        self._dt_occlusion = dt_occlusion
        self._gt_ignore = gt_ignore
        self._gt_velocity = gt_velocity
        self._cls = cls
        dist_thresh = sorted(dist_thresh)
        self._dist_thresh = dist_thresh
        self._occl_thresh = occl_thresh
        self._velocity_thresh = velocity_thresh
        self._interest_labels = interest_labels
        self._distance_compute_model = distance_compute_model
        self._name_to_label = name_to_label
        self._has_priority = has_priority
        self._overlap_thresh = overlap_thresh
        self._query_space = self.createQuerySpace(dist_thresh)
        self._n_frames = []

        # statistics and numbers for evaluation
        self._n_gt = [0]*(len(self._dist_thresh)+1)*2 # number of ground truth detections minus ignored false negatives and true positives
        self._n_igt = 0  # number of ignored ground truth detections
        self._n_gts = []  # number of ground truth detections minus ignored false negatives and true positives PER SEQUENCE
        self._n_igts = []  # number of ground ignored truth detections PER SEQUENCE
        self._n_gt_trajectories = 0
        self._n_gt_seq = []
        self._n_tr = [0]*(len(self._dist_thresh)+1)*2# number of tracker detections minus ignored tracker detections
        self._n_trs = []  # number of tracker detections minus ignored tracker detections PER SEQUENCE
        self._n_itr = 0  # number of ignored tracker detections
        self._n_itrs = []  # number of ignored tracker detections PER SEQUENCE
        self._n_igttr = 0  # number of ignored ground truth detections where the corresponding associated tracker detection is also ignored
        self._n_tr_trajectories = 0
        self._n_tr_seq = []
        self._tp = [0]*(len(self._dist_thresh)+1)*2  # number of true positives including ignored true positives!
        self._itp = 0  # number of ignored true positives
        self._tps = []  # number of true positives including ignored true positives PER SEQUENCE
        self._itps = []  # number of ignored true positives PER SEQUENCE
        self._fn = [0]*(len(self._dist_thresh)+1)*2 # number of false negatives WITHOUT ignored false negatives
        self._ifn = 0  # number of ignored false negatives
        self._fns = []  # number of false negatives WITHOUT ignored false negatives PER SEQUENCE
        self._ifns = []  # number of ignored false negatives PER SEQUENCE
        self._fp = [0]*(len(self._dist_thresh)+1)*2  # number of false positives
        self._fps = [] # above PER SEQUENCE
        self._id_switches = [0]*(len(self._dist_thresh)+1)*2

        # this should be enough to hold all groundtruth trajectories
        # is expanded if necessary and reduced in any case
        self._gt_trajectories = None
        self._ign_trajectories = None
        self.timeStamp_lasts = []
        self.timeStamp_currents = []
        self.id_error = []

        self._groundtruth = None
        self._tracker = None

        self._velocity = []
        for dist_idx in range(len(self._dist_thresh)+1):
            velocity_tmp = []
            for i in range(3):
                velocity_tmp.append([[] for i in range(6)])
            self._velocity.append(velocity_tmp)


    def get_iou_bev(self, gt_corners, dt_corners, gt_num, dt_num):
        if gt_num <= 0 or dt_num <= 0:
            return None

        iou = iou_cuda.iou_forward(torch.cuda.FloatTensor(gt_corners),torch.cuda.FloatTensor(dt_corners))
        iou = iou.cpu().numpy().reshape(gt_num, dt_num)

        return iou

    def createEvalDir(self):
        self._eval_dir = os.path.join(self._eval_path, self._cls)
        if not os.path.exists(self._eval_dir):
            os.makedirs(self._eval_dir)
        self.create_file()


    def create_file(self):
        names = ['/missing_', '/fp_', '/ids_']
        for i in range(3):
            items = [self._all_missing_obj[1].items(), self._all_fp_obj[1].items(), self._all_id_switch_obj.items()]
            for j in range(len(items)):
                for key, value in items[i]:
                    filename = self._eval_dir + names[i] + key + '.pkl'
                    with open(filename, 'wb') as fp:
                        pickle.dump(value, fp)

    def get_needed_velocity(self, left_thresh, right_thresh):
        new_velocity = []
        for dist_idx in range(len(self._dist_thresh) + 1):
            velocity_tmp = []
            for i in range(3):
                velocity_tmp.append([[] for i in range(4)])
            new_velocity.append(velocity_tmp)

        for dist_idx in range(len(self._dist_thresh) + 1):
            for i in range(3):
                for j in range(len(self._velocity[dist_idx][i][0])):
                    abs_velocity_box = self._velocity[dist_idx][i][0][j]
                    abs_velocity_icp = self._velocity[dist_idx][i][1][j]
                    if abs_velocity_box > left_thresh and abs_velocity_box < right_thresh:
                        new_velocity[dist_idx][i][0].append(self._velocity[dist_idx][i][2][j])
                        new_velocity[dist_idx][i][1].append(self._velocity[dist_idx][i][3][j])
                    if abs_velocity_icp > left_thresh and abs_velocity_icp < right_thresh:
                        new_velocity[dist_idx][i][2].append(self._velocity[dist_idx][i][4][j])
                        new_velocity[dist_idx][i][3].append(self._velocity[dist_idx][i][5][j])

        return new_velocity


    def generate_velocity_table_category(self):
        for i in range(len(self._velocity_thresh)):
            thresh = self._velocity_thresh[i]
            if i == 0:
                new_velocity = self.get_needed_velocity(-100, thresh)
                filename = "less_"+ str(thresh) + "_eval_velocity_result.png"
            else:
                new_velocity = self.get_needed_velocity(self._velocity_thresh[i-1], thresh)
                filename = str(self._velocity_thresh[i-1]) + "to" + str(thresh) + "_eval_velocity_result.png"

            self.generate_velocity_table(new_velocity, filename)

        thresh = self._velocity_thresh[-1]
        new_velocity = self.get_needed_velocity(thresh, 100)
        filename = "bigger_" + str(thresh) + "_eval_velocity_result.png"
        self.generate_velocity_table(new_velocity, filename)


    def generate_velocity_table(self, new_velocity, filename):
        if self._cls != "car":
            return
        plt.cla()

        velocity_error = []
        for dist_idx in range(len(self._dist_thresh) + 1):
           velocity_error.append(np.zeros((3, 4)))

        numbers = np.zeros(((len(self._dist_thresh)+1), 3))
        for dist_idx in range(len(self._dist_thresh) + 1):
            for i in range(3):
                numbers[dist_idx][i] = len(new_velocity[dist_idx][i][0])
                for j in range(4):
                    velocity_error[dist_idx][i][j] = np.round(np.mean(new_velocity[dist_idx][i][j]), 2)

        total_data = np.concatenate(velocity_error, axis=0)
        total_data = np.concatenate([numbers.reshape(-1, 1), total_data], axis = 1)

        rows_tmp = []
        for i, thresh in enumerate(self._dist_thresh):
            rows_tmp.extend(['high_within_' + str(thresh), 'medium_within_' + str(thresh), 'low_within_' + str(thresh)])
        rows_tmp.extend(['high_full_range', 'medium_full_range', 'low_full_range'])
        rows = []
        for i in [0,3,6,9,1,4,7,10,2,5,8,11]:
            rows.append(rows_tmp[i])

        total_data = total_data[[0,3,6,9,1,4,7,10,2,5,8,11],:]
        columns = ["number of boxes", "velocity_box_along", "velocity_box_across", "velocity_icp_along", "velocity_icp_across"]
        the_table = plt.table(cellText=total_data,
                              colWidths=[0.2] * 5,
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
        save_path = os.path.join(self._eval_dir, "velocity")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, filename), bbox_inches='tight', pad_inches=0.05)


    def generate_table(self, fn, fp, n_gt, n_tr, id_switches, istotal = False):
        np.seterr(divide='ignore', invalid='ignore')
        total_data = np.concatenate([np.array(fn).reshape(-1,1), np.array(fp).reshape(-1,1),\
                                     np.array(n_gt).reshape(-1,1), np.array(n_tr).reshape(-1,1),\
                                     np.array(id_switches).reshape(-1,1)], axis=1)

        rows = self._query_space
        self.draw_table(rows, self._eval_dir, total_data, istotal)

    def draw_table(self, rows, _eval_dir, data, istotal):
        plt.cla()
        missing_rate = np.round(1-data[:, 0]/data[:, 2], decimals=4).reshape((len(self._dist_thresh)+1)*2, 1)
        fp_rate = np.round(1-data[:, 1]/data[:, 3], decimals=4).reshape((len(self._dist_thresh)+1)*2, 1)
        data = np.append(data, missing_rate, axis=1)
        data = np.append(data, fp_rate, axis=1)
        mota = np.round((1-((data[:, 0]+data[:, 1]+data[:, 4])/data[:, 2])), decimals=4).reshape((len(self._dist_thresh)+1)*2, 1)
        data = np.append(data, mota, axis=1)
        columns = ["missing", "fp", "gt", "dt", "id-switch", "recall", "precision", "mota"]
        the_table = plt.table(cellText=data,
                              colWidths=[0.12] * 8,
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
        if istotal:
            filename = "total_eval_result.png"
            plt.savefig(os.path.join(self._eval_path, filename), bbox_inches='tight', pad_inches=0.05)
        else:
            filename = self._cls + "_eval_result.png"
            plt.savefig(Path(_eval_dir)/filename, bbox_inches='tight', pad_inches=0.05)

    def createQuerySpace(self, dist_thresh):
        query_space = []
        for i, thresh in enumerate(dist_thresh):
            query_space.append('within_' + str(thresh))
            query_space.append('within_' + str(thresh) + "_rm_occluded")
        query_space.append("full_range")
        query_space.append("full_range_rm_occluded")
        return query_space

    def computeIdSwitch(self, dist_length):
        # compute id-switches for all groundtruth trajectories

        self._all_id_switch_obj = {key: None for key in self._query_space}

        id_switch_obj= []
        for dist_idx in range(dist_length*2):
            id_switch_obj.append(defaultdict(list))

        for dist_idx, seq_ignored in enumerate(self._ign_trajectories):
            for name in self._gt_trajectories.keys():
                g = self._gt_trajectories[name]
                ign_g = seq_ignored[name]

                # all frames of this gt trajectory are ignored
                if all(ign_g):
                    continue
                # all frames of this gt trajectory are not assigned to any detections
                if all([this == -1 for this in g]):
                    continue

                last_id = g[0][1]
                # print(g)
                # first detection (necessary to be in gt_trajectories) is always tracked
                for f in range(1, len(g)):
                    if ign_g[f]:
                        last_id = -1
                        continue
                    if last_id != g[f][1] and last_id != -1 and g[f][1] != -1 and g[f - 1][1] != -1:
                        self._id_switches[dist_idx] += 1
                        id_switch_obj[dist_idx][g[f][0]].append(name)
                    if g[f][1] != -1:
                        last_id = g[f][1]

        for space_idx, key in enumerate(self._all_id_switch_obj.keys()):
            self._all_id_switch_obj[key] = id_switch_obj[space_idx]

        return True

    def velocity_across_along_error(self, velocity_gt, velocity_tr):

        degree_gt = angle(velocity_gt)
        degree_tr = angle(velocity_tr)
        degree = abs(degree_gt-degree_tr)
        if degree > 180:
            degree = 360 - degree

        velocity_len_gt = math.sqrt(velocity_gt[0]**2+velocity_gt[1]**2)
        velocity_len_tr = math.sqrt(velocity_tr[0]**2+velocity_tr[1]**2)
        along = abs(velocity_len_gt - velocity_len_tr * math.cos(degree/180*math.pi))
        across = abs(velocity_len_tr * math.sin(degree/180*math.pi))

        return along, across

    def compute3rdPartyMetrics(self):

        """
            computes the metrics including fn(missing), fp and id-switch
        """

        # construct Munkres object for Hungarian Method association
        hm = Munkres()
        max_cost = 1e9

        # go through all frames and associate ground truth and tracker results
        dist_length = len(self._dist_thresh) + 1

        self._all_missing_obj = [{key:None for key in self._query_space} for i in range(2)]
        self._all_fp_obj = [{key:None for key in self._query_space} for i in range(2)]


        info_text_names = get_all_txt_names(self._dt_path)
        seqtp = [0] * dist_length * 2
        seqitp = [0] * dist_length * 2
        seqfn = [0] * dist_length * 2
        seqifn = [0] * dist_length * 2
        seqfp = [0] * dist_length * 2
        seqigt = [0] * dist_length * 2
        seqitr = [0] * dist_length * 2
        n_gts, n_trs = 0, 0
        missing_obj_per_seq, fp_obj_per_seq = [], []
        missing_obj_per_seq_all, fp_obj_per_seq_all = [], []
        for dist_idx in range(dist_length*2):
            missing_obj_per_seq.append(defaultdict(list))
            fp_obj_per_seq.append(defaultdict(list))
            missing_obj_per_seq_all.append(defaultdict(list))
            fp_obj_per_seq_all.append(defaultdict(list))

        seq_trajectories, seq_ignored = defaultdict(list), [defaultdict(list) for dist_idx in range(dist_length*2)]
        for frame_idx in tqdm(range(len(info_text_names))):
            timeStamp = info_text_names[frame_idx]

            g = self._groundtruth[frame_idx]
            t = self._tracker[frame_idx]

            for dist_idx in range(dist_length*2):
                self._n_gt[dist_idx] += len(g)
                self._n_tr[dist_idx] += len(t)
                for tt in t:
                    fp_obj_per_seq[dist_idx][timeStamp].append(tt._track_id)
                    fp_obj_per_seq_all[dist_idx][timeStamp].append(tt._track_id)

            n_gts += len(g)
            n_trs += len(t)

            # use hungarian method to associate, using distance error as cost
            cost_matrix = []
            this_ids = [[], []]
            for gg in g:
                # save current ids
                this_ids[0].append(gg._track_id)
                this_ids[1].append(-1)
                gg._tracker = -1
                cost_row = []
                for tt in t:
                    c = self.get_iou_bev(tt._corners.reshape(-1, 4, 2), gg._corners.reshape(-1, 4, 2), 1, 1)[0, 0]
                    if c >= self._overlap_thresh:
                        cost_row.append(1 - c)
                    else:
                        cost_row.append(max_cost)
                cost_matrix.append(cost_row)

                # all ground truth trajectories are initially not associated
                # extend ground truth trajectories lists (merge lists)
                seq_trajectories[gg._track_id].append((timeStamp, -1))

                for dist_idx in range(dist_length*2):
                    seq_ignored[dist_idx][gg._track_id].append(False)


                # check
                if len(seq_trajectories[gg._track_id]) >= 2:
                    timeStamp_last = seq_trajectories[gg._track_id][-2][0].split(".")[0]
                    timeStamp_current = timeStamp.split(".")[0]
                    if int(timeStamp_current)-int(timeStamp_last) > 100000:
                        self.timeStamp_lasts.append(timeStamp_last)
                        self.timeStamp_currents.append(timeStamp_current)
                        self.id_error.append(gg._track_id)
                        print("id: %s ---- timeStamp_last: %s ---- timeStamp_current: %s" % (gg._track_id, timeStamp_last, timeStamp_current))
                        #raise


            if len(g) is 0:
                cost_matrix = [[]]
            # associate

            association_matrix = hm.compute(cost_matrix)
            tmptp, tmpfn, rows = 0, 0, []
            for row, col in association_matrix:
                rows.append(row)
                cost = cost_matrix[row][col]
                if cost < max_cost:
                    g[row]._tracker = t[col]._track_id
                    this_ids[1][row] = t[col]._track_id
                    t[col]._valid = True
                    g[row]._distance_cost = cost
                    seq_trajectories[g[row]._track_id][-1] = (timeStamp, t[col]._track_id)

                    # adjust occlusion relationships of tracking results
                    if t[col]._occlusion != g[row]._occlusion:
                        t[col]._occlusion = g[row]._occlusion
                        t[col].__distance = g[row]._distance

                    tmptp += 1
                    for dist_idx in range(dist_length*2):
                        fp_obj_per_seq[dist_idx][timeStamp].remove(t[col]._track_id)
                        fp_obj_per_seq_all[dist_idx][timeStamp].remove(t[col]._track_id)

                    # compute the error of velocity
                    if self._cls.lower() == "car" and g[row]._velocity_confidence != -1 and \
                            ((not self._has_priority) or (self._has_priority and t[col]._priority==1)):
                        box_along, box_across = self.velocity_across_along_error(g[row]._velocity_box[:2], t[col]._velocity_box[:2])
                        icp_along, icp_across = self.velocity_across_along_error(g[row]._velocity_ICP[:2], t[col]._velocity_box[:2])
                        abs_velocity_box = (g[row]._velocity_box[0]**2+g[row]._velocity_box[1]**2)**0.5
                        abs_velocity_ICP = (g[row]._velocity_ICP[0]**2+g[row]._velocity_ICP[1]**2)**0.5
                        for dist_idx in range(g[row]._distance, dist_length):
                            self._velocity[dist_idx][g[row]._velocity_confidence][0].append(abs_velocity_box)
                            self._velocity[dist_idx][g[row]._velocity_confidence][1].append(abs_velocity_ICP)
                            self._velocity[dist_idx][g[row]._velocity_confidence][2].append(box_along)
                            self._velocity[dist_idx][g[row]._velocity_confidence][3].append(box_across)
                            self._velocity[dist_idx][g[row]._velocity_confidence][4].append(icp_along)
                            self._velocity[dist_idx][g[row]._velocity_confidence][5].append(icp_across)

                else:
                    g[row]._tracker = -1
                    tmpfn += 1
                    for dist_idx in range(dist_length*2):
                        missing_obj_per_seq[dist_idx][timeStamp].append(g[row]._track_id)
                        missing_obj_per_seq_all[dist_idx][timeStamp].append(g[row]._track_id)

            if len(g) > len(t):
                for g_idx in range(len(g)):
                    if g_idx not in rows:
                        g[g_idx]._tracker = -1
                        tmpfn += 1
                        for dist_idx in range(dist_length*2):
                            missing_obj_per_seq[dist_idx][timeStamp].append(g[g_idx]._track_id)
                            missing_obj_per_seq_all[dist_idx][timeStamp].append(g[g_idx]._track_id)

            nignoredtracker_invalid = [0] * dist_length * 2
            nignoredtracker_valid = [0] * dist_length * 2

            # take into consideration attributes including occlusion and distance
            for tt in t:
                # ignore detection if it is occluded by other boxes or it is out of range that we consider
                if tt._occlusion and not tt._valid:
                    for dist_idx in range(1, dist_length*2, 2):
                        nignoredtracker_invalid[dist_idx] += 1
                        if tt._track_id in fp_obj_per_seq[dist_idx][timeStamp]:
                            fp_obj_per_seq[dist_idx][timeStamp].remove(tt._track_id)
                    for dist_idx in range(0, tt._distance*2, 2):
                        nignoredtracker_invalid[dist_idx] += 1
                        if tt._track_id in fp_obj_per_seq[dist_idx][timeStamp]:
                            fp_obj_per_seq[dist_idx][timeStamp].remove(tt._track_id)
                    for dist_idx in range(tt._distance*2):
                        if tt._track_id in fp_obj_per_seq_all[dist_idx][timeStamp]:
                            fp_obj_per_seq_all[dist_idx][timeStamp].remove(tt._track_id)

                elif not tt._occlusion and not tt._valid:
                    for dist_idx in range(tt._distance*2):
                        nignoredtracker_invalid[dist_idx] += 1
                        if tt._track_id in fp_obj_per_seq[dist_idx][timeStamp]:
                            fp_obj_per_seq[dist_idx][timeStamp].remove(tt._track_id)
                            fp_obj_per_seq_all[dist_idx][timeStamp].remove(tt._track_id)

                elif tt._occlusion and tt._valid:
                    for dist_idx in range(1, dist_length*2, 2):
                        nignoredtracker_valid[dist_idx] += 1
                    for dist_idx in range(0, tt._distance*2, 2):
                        nignoredtracker_valid[dist_idx] += 1

                else:
                    for dist_idx in range(tt._distance*2):
                        nignoredtracker_valid[dist_idx] += 1

            ignoredfn = [0] * dist_length * 2  # the number of ignored false negatives
            nignoredtp = [0] * dist_length * 2  # the number of ignored true positives
            # which is ignored but where the associated tracker
            # detection has already been ignored

            for g_idx, gg in enumerate(g):
                if gg._tracker < 0:  # missing
                    if (not gg._occlusion) and (not gg._ignored):
                        for idx in range(gg._distance*2):
                            seq_ignored[idx][gg._track_id][-1] = True
                            missing_obj_per_seq[idx][timeStamp].remove(gg._track_id)
                            missing_obj_per_seq_all[idx][timeStamp].remove(gg._track_id)
                            ignoredfn[idx] += 1

                    elif gg._ignored:
                        for idx in range(0, dist_length*2):
                            seq_ignored[idx][gg._track_id][-1] = True
                            missing_obj_per_seq[idx][timeStamp].remove(gg._track_id)
                            missing_obj_per_seq_all[idx][timeStamp].remove(gg._track_id)
                            ignoredfn[idx] += 1

                    else:
                        for idx in range(1, dist_length*2, 2):
                            seq_ignored[idx][gg._track_id][-1] = True
                            missing_obj_per_seq[idx][timeStamp].remove(gg._track_id)
                            ignoredfn[idx] += 1
                        for idx in range(0, gg._distance*2, 2):
                            seq_ignored[idx][gg._track_id][-1] = True
                            missing_obj_per_seq[idx][timeStamp].remove(gg._track_id)
                            ignoredfn[idx] += 1
                        for idx in range(gg._distance*2):
                            missing_obj_per_seq_all[idx][timeStamp].remove(gg._track_id)


                elif gg._tracker >= 0:  # match
                    if (not gg._occlusion) and (not gg._ignored):
                        for idx in range(0, gg._distance*2):
                            seq_ignored[idx][gg._track_id][-1] = True
                            nignoredtp[idx] += 1

                    elif gg._ignored:
                        for idx in range(0, dist_length*2):
                            seq_ignored[idx][gg._track_id][-1] = True
                            nignoredtp[idx] += 1

                    else:
                        for idx in range(1, dist_length*2, 2):
                            seq_ignored[idx][gg._track_id][-1] = True
                            nignoredtp[idx] += 1
                        for idx in range(0, gg._distance*2, 2):
                            seq_ignored[idx][gg._track_id][-1] = True
                            nignoredtp[idx] += 1

            # update sequence data
            for idx in range(dist_length*2):
                seqtp[idx] += tmptp - nignoredtp[idx]
                self._tp[idx] += tmptp - nignoredtp[idx]
                seqitp[idx] += nignoredtp[idx]
                seqfp[idx] += len(t) - tmptp - nignoredtracker_invalid[idx]
                self._fp[idx] += len(t) - tmptp - nignoredtracker_invalid[idx]
                seqfn[idx] += len(g) - tmptp - ignoredfn[idx]
                self._fn[idx] += len(g) - tmptp - ignoredfn[idx]
                seqifn[idx] += ignoredfn[idx]
                seqigt[idx] += ignoredfn[idx] + nignoredtp[idx]
                seqitr[idx] += nignoredtracker_invalid[idx] + nignoredtracker_valid[idx]
                self._n_gt[idx] -= (ignoredfn[idx] + nignoredtp[idx])
                self._n_tr[idx] -= (nignoredtracker_invalid[idx] + nignoredtp[idx])

        self._gt_trajectories= seq_trajectories
        self._ign_trajectories = seq_ignored

        for space_idx, key in enumerate(self._all_missing_obj[0].keys()):
            self._all_missing_obj[0][key] = missing_obj_per_seq_all[space_idx]
            self._all_fp_obj[0][key] = fp_obj_per_seq_all[space_idx]
            self._all_missing_obj[1][key] = missing_obj_per_seq[space_idx]
            self._all_fp_obj[1][key] = fp_obj_per_seq[space_idx]

        self.computeIdSwitch(dist_length)

        return True

def save_total_info(interest_labels, fp, missing, ids, fp_save_path, missing_save_path, ids_save_path):
    for key in fp[0].keys():
        for idx in range(1, len(interest_labels)):
            fp[0][key].extend(fp[idx][key])
            missing[0][key].extend(missing[idx][key])
        fp[0][key] = list(set(fp[0][key]))
        missing[0][key] = list(set(missing[0][key]))
    with open(fp_save_path, "wb") as f:
        pickle.dump(fp[0], f)
    with open(missing_save_path, "wb") as f:
        pickle.dump(missing[0], f)

    for idx in range(1, len(interest_labels)):
        for key in ids[idx].keys():
            if key in ids[0]:
                ids[0][key].extend(ids[idx][key])
            else:
                ids[0][key] = ids[idx][key]
    for key in ids[0].keys():
        ids[0][key] = list(set(ids[0][key]))
    with open(ids_save_path, "wb") as f:
        pickle.dump(ids[0], f)

def generate_total_info(eval_path, interest_labels, dist_thresh, removeOccluded):
    if removeOccluded:
        endname = "_rm_occluded.pkl"
    else:
        endname = ".pkl"

    eval_paths = []
    for cls in interest_labels:
        eval_paths.append(os.path.join(eval_path, cls))

    fp = []
    missing = []
    ids = []
    for cls_idx in range(len(interest_labels)):
        fp_path = eval_paths[cls_idx] / Path("fp_full_range" + endname)
        missing_path = eval_paths[cls_idx] / Path("missing_full_range" + endname)
        ids_path = eval_paths[cls_idx] / Path("ids_full_range" + endname)

        with open(fp_path, "rb") as f:
            fp.append(pickle.load(f))
        with open(missing_path, "rb") as f:
            missing.append(pickle.load(f))
        with open(ids_path, "rb") as f:
            ids.append(pickle.load(f))

    fp_save_path = eval_path / Path("total_fp_full_range" + endname)
    missing_save_path = eval_path / Path("total_missing_full_range" + endname)
    ids_save_path = eval_path / Path("total_ids_full_range" + endname)
    save_total_info(interest_labels, fp, missing, ids, fp_save_path, missing_save_path, ids_save_path)

    for dist in dist_thresh:
        fp = []
        missing = []
        ids = []

        for cls_idx in range(len(interest_labels)):
            fp_path = eval_paths[cls_idx] / Path("fp_within_" + str(dist) + endname)
            missing_path = eval_paths[cls_idx] / Path("missing_within_" + str(dist) + endname)
            ids_path = eval_paths[cls_idx] / Path("ids_within_" + str(dist) + endname)

            with open(fp_path, "rb") as f:
                fp.append(pickle.load(f))
            with open(missing_path, "rb") as f:
                missing.append(pickle.load(f))
            with open(ids_path, "rb") as f:
                ids.append(pickle.load(f))

        fp_save_path = eval_path / Path("total_fp_within_" + str(dist) + endname)
        missing_save_path = eval_path / Path("total_missing_within_" + str(dist) + endname)
        ids_save_path = eval_path / Path("total_ids_within_" + str(dist) + endname)
        save_total_info(interest_labels, fp, missing, ids, fp_save_path, missing_save_path, ids_save_path)

def get_n_trajectories(data):
    ids = []
    n_trajectories = 0
    for sub_data in data:
        for sub_sub_data in sub_data:
            if sub_sub_data._track_id not in ids:
                ids.append(sub_sub_data._track_id)
                n_trajectories += 1
    return n_trajectories

def load_data(root_dir, cls, frame_id, load_GT, has_priority = False):
    t_data = tData()
    id_frame_cache = []
    f_data = []

    timeStamp = info_text_names[frame_id]
    try:
        with open(os.path.join(root_dir, timeStamp), "rb") as f:
            objects = json.load(f)
    except:
        objects = {}
        objects["objects"] = []

    for obj in objects["objects"]:
        if name_to_label[obj['type'].lower()] >= len(interest_labels):
            continue
        if interest_labels[name_to_label[obj['type'].lower()]].lower() != cls.lower():
            continue
        if obj['type'].lower() == "cone" and obj["position"]["x"] < 0:
            continue

        t_data._track_id = obj['id']
        t_data._obj_type = obj['type'].lower()
        center = [obj["position"]["x"], obj["position"]["y"], obj["position"]["z"]]
        size = [obj["bounding_box"]["length"], obj["bounding_box"]["width"], obj["bounding_box"]["height"]]
        t_data._corners = box_ops.get_corners(np.array(center).reshape(-1, 3), np.array(size).reshape(-1, 3),
                                              np.array(obj["heading"]).reshape(-1)).reshape(4, 2)
        t_data._center = np.array(center)


        if cls == "car":
            if load_GT:
                t_data._velocity_confidence = gt_velocity[timeStamp][obj['id']][0]
                t_data._velocity_box = gt_velocity[timeStamp][obj['id']][1]
                t_data._velocity_ICP = gt_velocity[timeStamp][obj['id']][2]
            else:
                #t_data._velocity_box = [0.0, 0.0, 0.0]
                t_data._velocity_box = [float(obj["velocity"]["x"]), float(obj["velocity"]["y"]), float(obj["velocity"]["z"])]
                if has_priority:
                    t_data._priority = obj["priority"]


        if load_GT:
            t_data._distance = filter_d(t_data._corners, dist_thresh) if distance_compute_model == 0 else filter_d_v2(t_data._center, dist_thresh)
            t_data._occlusion = gt_occlusion[timeStamp][t_data._track_id]
            t_data._ignored = gt_ignore[timeStamp][t_data._track_id]
        else:
            t_data._distance = filter_d(t_data._corners, dist_thresh) if distance_compute_model == 0 else filter_d_v2(t_data._center, dist_thresh)
            t_data._occlusion = dt_occlusion[timeStamp][t_data._track_id]

        id_frame = t_data._track_id
        if id_frame in id_frame_cache and (not load_GT):
            raise Exception("track id occured at least twice for one frame!")
        id_frame_cache.append(id_frame)

        f_data.append(copy.copy(t_data))

    return f_data


def evaluate(gt_path, dt_path, pcd_path, eval_path, config_path, save_path, occlusion_path,
             ignore_path, gt_velocity_path, pool_num, has_priority=False):

    global interest_labels
    global dist_thresh
    global gt_occlusion
    global dt_occlusion
    global gt_ignore
    global gt_velocity
    global info_text_names
    global name_to_label
    global distance_compute_model
    global num_features_for_pc

    # loading configuration information
    if isinstance(config_path, str):
        config = tracking_eval_pb2.DeeprouteTrackingEvalConfig()
        with open(config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, config)
    else:
        config = config_path

    interest_labels = list(config.deeproute_eval_input_reader.interest_labels)
    dist_thresh = list(config.deeproute_eval_input_reader.distance_thresh)
    overlap_thresh = config.deeproute_eval_input_reader.overlap_thresh
    occlusion_thresh = config.deeproute_eval_input_reader.occlusion_thresh
    filter_points_number = list(config.deeproute_eval_input_reader.filter_points_number)
    distance_compute_model = config.deeproute_eval_input_reader.distance_compute_model
    num_features_for_pc = int(config.deeproute_eval_input_reader.num_features_for_pc)
    velocity_thresh = config.deeproute_eval_input_reader.velocity_thresh

    name_to_label = {}
    for sample_group in config.deeproute_eval_input_reader.name_to_label.sample_groups:
        name_to_label.update(dict(sample_group.name_to_num))

    try:
        with open(gt_velocity_path, "rb") as f:
            gt_velocity = pickle.load(f)
    except:
        print("please compute the velocity of gt first!")
        raise

    # loading the occlusion and ignore attribute

    try:
        with open(occlusion_path, "rb") as f:
            gt_occlusion = pickle.load(f)
            if abs(occlusion_thresh - gt_occlusion[0]) > 0.01:
                print("rerun compute_occl_ignore.py due to the change of occlusion_thresh!")
                raise
            gt_occlusion = gt_occlusion[1]
        with open(ignore_path, "rb") as f:
            gt_ignore = pickle.load(f)
            for i in range(len(filter_points_number)):
                if filter_points_number[i] != gt_ignore[0][i]:
                    print("rerun compute_occl_ignore.py due to the change of filter_points_number!")
                    raise
            gt_ignore = gt_ignore[1]
            # print(gt_ignore)
    except:
        print("---Loading occlusion attribute of gt---")
        gt_occlusion, gt_ignore = compute_occl_ignore(gt_path, pcd_path, save_path, config_path, pool_num, False, True)

    # try:
    #     occlusion_path = os.path.join(root_path, "kittiviewer/occlusion_detection.pkl")
    #     ignore_path = os.path.join(root_path, "kittiviewer/ignore_detection.pkl")
    #     with open(occlusion_path, "rb") as f:
    #         dt_occlusion = pickle.load(f)
    #         dt_occlusion = dt_occlusion[1]
    #     with open(ignore_path, "rb") as f:
    #         dt_ignore = pickle.load(f)
    #         dt_ignore = dt_ignore[1]
    #         # print(gt_ignore)
    if True:
        # computing the occlusion and distance attribute of tracking results
        print("---Loading occlusion attribute of tracking results---")
        dt_occlusion, dt_ignore = compute_occl_ignore(gt_path, pcd_path, save_path, config_path, pool_num, False, True, dt_path)

    # compute missing, fp and id-switch for each class including car, pedestrian, cyclist
    total_fn = [0] * (len(dist_thresh) + 1) * 2
    total_fp = [0] * (len(dist_thresh) + 1) * 2
    total_id_switches = [0] * (len(dist_thresh) + 1) * 2
    total_gt = [0] * (len(dist_thresh) + 1) * 2
    total_tr = [0] * (len(dist_thresh) + 1) * 2

    for i, cls in enumerate(interest_labels):
        track_eval = trackingEvaluation(gt_path,
                                        dt_path,
                                        eval_path,
                                        gt_occlusion,
                                        dt_occlusion,
                                        gt_ignore,
                                        gt_velocity,
                                        cls,
                                        interest_labels,
                                        distance_compute_model,
                                        dist_thresh,
                                        overlap_thresh[i],
                                        occlusion_thresh,
                                        velocity_thresh,
                                        name_to_label,
                                        has_priority)

        # loading groundtruth and tracking results
        print("--------------loading gt---------------------")
        #track_eval.loadData(gt_path, load_GT=True)
        info_text_names = get_all_txt_names(gt_path)
        res = []
        groundtruth = []
        pool = multiprocessing.Pool(pool_num)
        print("ccccc")
        for frame_id in range((len(info_text_names))):
            #res.append(pool.apply_async(load_data, (gt_path, cls, frame_id, True)))
            load_data(gt_path, cls, frame_id, True)

        print("BBBBB")
        for frame_id in tqdm(range(len(res))):
            groundtruth.append(res[frame_id].get())

        pool.close()
        pool.join()

        print("aaaaaa")

        track_eval._groundtruth = groundtruth
        track_eval._n_gt_trajectories = get_n_trajectories(groundtruth)

        print("--------------loading dt---------------------")
        #track_eval.loadData(dt_path, load_GT=False)
        info_text_names = get_all_txt_names(dt_path)
        res = []
        tracker = []
        pool = multiprocessing.Pool(pool_num)
        for frame_id in range(len(info_text_names)):
            res.append(pool.apply_async(load_data, (dt_path, cls, frame_id, False, has_priority)))

        for frame_id in tqdm(range(len(res))):
            tracker.append(res[frame_id].get())

        pool.close()
        pool.join()

        track_eval._tracker = tracker
        track_eval._n_tr_trajectories = get_n_trajectories(tracker)

        #assert len(track_eval._groundtruth) == len(track_eval._tracker)

        # evaluate and save evaluation results
        print("--------------Evaluation:{}---------------------".format(cls))
        track_eval.compute3rdPartyMetrics()
        track_eval.createEvalDir()
        track_eval.generate_table(track_eval._fn,
                                  track_eval._fp,
                                  track_eval._n_gt,
                                  track_eval._n_tr,
                                  track_eval._id_switches)

        # accumulate the number of fn, fp, gt, dt id_switch
        for dist_idx in range((len(dist_thresh)+1)*2):
            total_fn[dist_idx] += track_eval._fn[dist_idx]
            total_fp[dist_idx] += track_eval._fp[dist_idx]
            total_gt[dist_idx] += track_eval._n_gt[dist_idx]
            total_tr[dist_idx] += track_eval._n_tr[dist_idx]
            total_id_switches[dist_idx] += track_eval._id_switches[dist_idx]

        if i == len(interest_labels)-1:
            track_eval.generate_table(total_fn, total_fp, total_gt, total_tr, total_id_switches, True)

        track_eval.generate_velocity_table_category()

    generate_total_info(eval_path, interest_labels, dist_thresh, True)
    generate_total_info(eval_path, interest_labels, dist_thresh, False)
    return True

if __name__ == "__main__":
    root_path = '/media/deeproute/SSD/tracking_data'
    gt_path = os.path.join(root_path, "groundtruth")
    dt_path = os.path.join(root_path, "trackings")
    eval_path = os.path.join(root_path, "evaluations/eval_detection")
    pcd_path = os.path.join(root_path, "pcd")
    save_path = os.path.join(root_path, "kittiviewer")
    occlusion_path = os.path.join(root_path, "kittiviewer/occlusion_tracking.pkl")
    ignore_path = os.path.join(root_path, "kittiviewer/ignore_tracking.pkl")
    gt_velocity_path = os.path.join(root_path, "kittiviewer/gt_velocity_confidence.pkl")

    poseConfigFile = os.path.join(root_path, "pose.csv")
    lidarConfigFile = os.path.join(root_path, "lidars_mkz.cfg")

    config_path = "/home/deeproute/workspace/deeproute/detection_evaluation/configs/tracking.eval.config"


    if len(sys.argv) > 1:
        pool_num = int(sys.argv[1])
    else:
        pool_num = 10

    has_priority = False

    success = evaluate(gt_path, dt_path, pcd_path, eval_path, config_path, save_path, occlusion_path,
                       ignore_path, gt_velocity_path, pool_num, has_priority)
    print("Evaluation Finished!")

