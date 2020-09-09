""
import copy
import mmcv
import numpy as np
import os
import tempfile
import torch
from mmcv.utils import print_log
from os import path as osp
from IPython import embed
from mmdet.datasets import DATASETS
from ..core import show_result
from ..core.bbox import Box3DMode, LiDARInstance3DBoxes, points_cam2img
from .custom_3d import Custom3DDataset
import json
@DATASETS.register_module()
class DeeprouteDataset(Custom3DDataset):
    r"""KITTI Dataset.

    This class serves as the API for experiments on the `KITTI Dataset
    <http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d>`_.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pts_prefix (str, optional): Prefix of points files.
            Defaults to 'velodyne'.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
    """
    #CLASSES = ('smallMot', 'bigMot', 'pedestrian', 'nonMot','TrafficCone')
    CLASSES = ('CAR','CAR_HARD','VAN','VAN_HARD','TRUCK','TRUCK_HARD','BIG_TRUCK','BUS','BUS_HARD','PEDESTRIAN', 'PEDESTRIAN_HARD', 'CYCLIST','CYCLIST_HARD','TRICYCLE','TRICYCLE_HARD','CONE')
    #CLASSES_EVAL = ("car", "car", "pedestrian", "cyclist", "cone")
    CLASSES_EVAL = ('car','car','car','car','car','car','car','car','car','pedestrian', 'pedestrian', 'cyclist','cyclist','cyclist','cyclist','cone')


    class_map = {
        "CAR":"smallMot",
        "VAN":"smallMot",
        "VAN_HARD":"smallMot",
        "TRUCK":"bigMot",
        "BIG_TRUCK":"bigMot",
        "TRUCK_HARD":"bigMot",
        "BUS":"bigMot",
        "BUS_HARD":"bigMot",
        "PEDESTRIAN":"pedestrian",
        "CYCLIST":"nonMot",
        "TRICYCLE":"nonMot",
        "TRICYCLE_HARD":"nonMot",
        "CAR_HARD":"smallMot",
        "PEDESTRIAN_HARD":"pedestrian",
        "CYCLIST_HARD":"nonMot",
        #"DONT_CARE":"",
        #"BACKGROUND":"",
        "CONE":"TrafficCone",
    #    "RAILING":"unknow",
    }


    def __init__(self,
                 data_root,
                 ann_file,
                 pts_prefix='bin',
                 pipeline=None,
                 classes=None,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 valid_mode=True):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            )

        assert self.modality is not None
        self.valid_mode= valid_mode
        self.pcd_limit_range = [-80, -80, -3, 80, 80, 1.0]
        self.pts_prefix = pts_prefix
    def get_cat_ids(self, idx):
        class_sample_idx = {name:[] for name in self.class_map}    
        info = self.data_infos[idx]
        gt_names = set(info['annos']['type'])
        for name in gt_names:
            if name in self.class_map:  
               class_sample_idx[name].append(idx) 
        return class_sample_idx
    def _get_pts_filename(self, idx):
        """Get point cloud filename according to the given index.

        Args:
            index (int): Index of the point cloud file to get.

        Returns:
            str: Name of the point cloud file.
        """
        # TODO : Modify the code to trainging and test split mode here
        
        if self.test_mode:
            if self.valid_mode:
                pts_filename = osp.join('./data/deeproute/validating', self.pts_prefix,idx[0],idx[1]+'.bin')
            else:
                pts_filename = osp.join('./data/deeproute/testing', self.pts_prefix,idx[0],idx[1]+'.bin')
        else:
            pts_filename = osp.join('./data/deeproute/training', self.pts_prefix,idx[0],idx[1]+'.bin')
        return pts_filename

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - img_prefix (str | None): Prefix of image files.
                - img_info (dict): Image info.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        sample_idx = info['image']['image_idx']
        #img_filename = os.path.join(self.data_root,info['image']['image_path'])

        # TODO: consider use torch.Tensor only
        pts_filename = self._get_pts_filename(sample_idx)
        input_dict = dict(
            sample_idx=sample_idx,
            pts_filename=pts_filename,
            img_prefix=None,
            )

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes.
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_bboxes (np.ndarray): 2D ground truth bboxes.
                - gt_labels (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        # Use index to get the annos, thus the evalhook could also use this api
        info = self.data_infos[index]
        #rect = info['calib']['R0_rect'].astype(np.float32)
        #Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)

        annos = info['annos']
        # we need other objects to avoid collision when sample
        annos = self.remove_dontcare(annos)
        gt_bboxes_3d = []
        for i in range(len(annos['location'])):        
            loc = list(annos['location'][i].values())
            dims = list(annos['dimensions'][i].values())
            rots = [annos['rotation_y'][i]]
            gt_bboxes_3d.append(loc+dims+rots)
        gt_bboxes_3d = np.array(gt_bboxes_3d)
        gt_bboxes_3d = LiDARInstance3DBoxes(gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1],
                       origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)
        gt_names = annos['type']
        
        # print(gt_names, len(loc))

        # convert gt_bboxes_3d to velodyne coordinates
        #gt_bboxes_3d = CameraInstance3DBoxes(gt_bboxes_3d).convert_to(
        #    self.box_mode_3d, np.linalg.inv(rect @ Trv2c))
        gt_bboxes = annos['bbox2d']

        selected = self.drop_arrays_by_name(gt_names, ['DONT_CARE','BACKGROUND'])
        # gt_bboxes_3d = gt_bboxes_3d[selected].astype('float32')
        bbox_2d=[]
        for bboxes_dic in gt_bboxes:
            for camera, bbox in bboxes_dic.items():
                bbox_2d.append(list(bbox.values()))
        gt_bboxes = np.array(bbox_2d)
        #gt_bboxes = gt_bboxes[selected].astype('float32')  
        
        gt_names = gt_names[selected]
        gt_labels = []
        for cat in gt_names:
            if cat in self.class_map:
                gt_labels.append(self.CLASSES.index(cat))
            else:
                gt_labels.append(-1)
        gt_labels = np.array(gt_labels)
        gt_labels_3d = copy.deepcopy(gt_labels)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            bboxes=gt_bboxes,
            labels=gt_labels,
            gt_names=gt_names)
        return anns_results

    def drop_arrays_by_name(self, gt_names, used_classes):
        """Drop irrelevant ground truths by name.

        Args:
            gt_names (list[str]): Names of ground truths.
            used_classes (list[str]): Classes of interest.

        Returns:
            np.ndarray: Indices of ground truths that will be dropped.
        """
        inds = [i for i, x in enumerate(gt_names) if x not in used_classes]
        inds = np.array(inds, dtype=np.int64)
        return inds

    def keep_arrays_by_name(self, gt_names, used_classes):
        """Keep useful ground truths by name.

        Args:
            gt_names (list[str]): Names of ground truths.
            used_classes (list[str]): Classes of interest.

        Returns:
            np.ndarray: Indices of ground truths that will be keeped.
        """
        inds = [i for i, x in enumerate(gt_names) if x in used_classes]
        inds = np.array(inds, dtype=np.int64)
        return inds

    def remove_dontcare(self, ann_info):
        """Remove annotations that do not need to be cared.

        Args:
            ann_info (dict): Dict of annotation infos. The ``'DontCare'``
                annotations will be removed according to ann_file['name'].

        Returns:
            dict: Annotations after filtering.
        """
        img_filtered_annotations = {}
        relevant_annotation_indices = [
            i for i, x in enumerate(ann_info['type']) if x != 'DONT_CARE' and x != 'BACKGROUND'
        ]
        for key in ann_info.keys():
            img_filtered_annotations[key] = (
                ann_info[key][relevant_annotation_indices])
        return img_filtered_annotations

    def format_results(self,
                       outputs,
                       save_folder,
                       ):
        """Format the results to pkl file.

        Args:
            outputs (list[dict]): Testing results of the dataset.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """

        result_files = self.bbox2result_deeproute(outputs, self.CLASSES_EVAL,
                                                 save_folder)
        return result_files

    def evaluate(self,
                 results,
                 metric=None,
                 logger=None,
                 pklfile_prefix=None,
                 show=False,
                 out_dir=None):
        """Evaluation in deeproute protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            pklfile_prefix (str | None): The prefix of pkl files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        result_files, tmp_dir = self.format_results(results, pklfile_prefix)
        from mmdet3d.core.evaluation import deeproute_eval
        #gt_annos = [info['annos'] for info in self.data_infos]
        result_dict = deeproute_eval()
        #print_log('\n' + ap_result_str, logger=logger)
        if tmp_dir is not None:
            tmp_dir.cleanup()
        if show:
            self.show(results, out_dir)
        return result_dict


    def bbox2result_deeproute(self,
                          net_outputs,
                          class_eval_names,
                          save_folder=None,
                          ):
        """Convert 3D detection results to deeproute format for evaluation and save to txt 
        

        Args:
            net_outputs (list[np.ndarray]): List of array storing the \
                inferenced bounding boxes and scores.
            class_names (list[String]): A list of class names.
            save_folder : save prediction result as txt for original deeproute evaluation pipeline
        """
        #print("bbox2result")
        #embed()
        assert len(net_outputs) == len(self.data_infos)

       
        print('\nConverting prediction to deeproute format')
        for idx, pred_dicts in enumerate(
                mmcv.track_iter_progress(net_outputs)):
            objects = []
            annos={}
            info = self.data_infos[idx]
            sample_idx = info['image']['image_idx']
            pred_dicts = pred_dicts['pts_bbox'] 
            if len(pred_dicts['boxes_3d']) > 0:
                scores = pred_dicts['scores_3d']
                box_preds_lidar = pred_dicts['boxes_3d']
                label_preds = pred_dicts['labels_3d']
                anno = {
                    'type': None,
                    'bounding_box': {},
                    'position': {},
                    'heading': None,
                    'score': None,
                    'id': None
                }
                ids = 0 
                for box_lidar, score, label in zip(box_preds_lidar, scores, label_preds):
                    anno = {'type': None,  'bounding_box': {},'position': {}, 'heading': None, 'score'
                            : None, 'id': None }
                    anno['type'] = class_eval_names[int(label)]
                    anno['bounding_box']['width'] = box_lidar[3].item()
                    anno['bounding_box']['height'] = box_lidar[5].item()
                    anno['bounding_box']['length'] = box_lidar[4].item()
                    anno['position']['x'] = box_lidar[0].item()
                    anno['position']['y'] = box_lidar[1].item()
                    anno['position']['z'] = box_lidar[2].item()
                    anno['heading'] = box_lidar[6].item()
                    anno['score'] = score.item()
                    anno['id'] = ids
                    ids+=1
                    objects.append(anno)
                   
              
            annos['objects']= objects

            #annos[-1]['sample_idx'] = np.array(
            #    [sample_idx] * len(annos[-1]['score']), dtype=np.int64)

            # TODO write file depend on config   
            #create the detection result directory for current experiment.
            if self.valid_mode:
                if not os.path.exists('./work_dirs/'+save_folder+'/evaluation/detections/valid/'+sample_idx[0]):
                    os.makedirs('./work_dirs/'+save_folder+'/evaluation/detections/valid/'+sample_idx[0])
                with open('./work_dirs/'+save_folder+'/evaluation/detections/valid/'+sample_idx[0]+'/'+sample_idx[1]+'.txt', 'w') as f :
                        js:on.dump(annos, f ,ensure_ascii=False)
                print('valid result is saved to prediction folder')
            elif self.test_mode:
                if not os.path.exists('./work_dirs/'+save_folder+'/evaluation/detections/test/'+sample_idx[0]):
                    os.makedirs('./work_dirs/'+save_folder+'/evaluation/detections/test/'+sample_idx[0])     
                with open('./work_dirs/'+save_folder+'/evaluation/detections/test/'+sample_idx[0]+'/'+sample_idx[1]+'.txt', 'w') as f :
                        json.dump(annos, f ,ensure_ascii=False)
                print('test result is saved to prediction folder')

        

    def convert_valid_bboxes(self, box_dict, info):
        """Convert the predicted boxes into valid ones.

        Args:
            box_dict (dict): Box dictionaries to be converted.

                - boxes_3d (:obj:`LiDARInstance3DBoxes`): 3D bounding boxes.
                - scores_3d (torch.Tensor): Scores of boxes.
                - labels_3d (torch.Tensor): Class labels of boxes.
            info (dict): Data info.

        Returns:
            dict: Valid predicted boxes.

                - bbox (np.ndarray): 2D bounding boxes.
                - box3d_camera (np.ndarray): 3D bounding boxes in \
                    camera coordinate.
                - box3d_lidar (np.ndarray): 3D bounding boxes in \
                    LiDAR coordinate.
                - scores (np.ndarray): Scores of boxes.
                - label_preds (np.ndarray): Class label predictions.
                - sample_idx (int): Sample index.
        """
        # TODO: refactor this function
        #print("convert_valid_bbox")
        #embed()
        box_preds = box_dict['boxes_3d']
        scores = box_dict['scores_3d']
        labels = box_dict['labels_3d']
        sample_idx = info['image']['image_idx']
        # TODO: remove the hack of yaw
        box_preds.tensor[:, -1] = box_preds.tensor[:, -1] - np.pi
        box_preds.limit_yaw(offset=0.5, period=np.pi * 2)

        if len(box_preds) == 0:
            return dict(
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                sample_idx=sample_idx)

        # Post-processing
        # check box_preds_camera
        # check box_preds
        limit_range = box_preds.tensor.new_tensor(self.pcd_limit_range)
        valid_pcd_inds = ((box_preds.center > limit_range[:3]) &
                          (box_preds.center < limit_range[3:]))
        valid_inds = valid_pcd_inds
        if valid_inds.sum() > 0:
            return dict(
                box3d_lidar=box_preds[valid_inds].tensor.numpy(),
                scores=scores[valid_inds].numpy(),
                label_preds=labels[valid_inds].numpy(),
                sample_idx=sample_idx,
            )
        else:
            return dict(
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                sample_idx=sample_idx,
            )

    def show(self, results, out_dir):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        for i, result in enumerate(results):
            example = self.prepare_test_data(i)
            data_info = self.data_infos[i]
            pts_path = data_info['point_cloud']['velodyne_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            # for now we convert points into depth mode
            points = example['points'][0]._data.numpy()
            points = points[..., [1, 0, 2]]
            points[..., 0] *= -1
            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d'].tensor
            gt_bboxes = Box3DMode.convert(gt_bboxes, Box3DMode.LIDAR,
                                          Box3DMode.DEPTH)
            gt_bboxes[..., 2] += gt_bboxes[..., 5] / 2
            pred_bboxes = result['boxes_3d'].tensor.numpy()
            pred_bboxes = Box3DMode.convert(pred_bboxes, Box3DMode.LIDAR,
                                            Box3DMode.DEPTH)
            pred_bboxes[..., 2] += pred_bboxes[..., 5] / 2
            show_result(points, gt_bboxes, pred_bboxes, out_dir, file_name)
