import torch
from mmcv.ops.nms import batched_nms
from mmcv.runner import force_fp32
from torch.nn import functional as F
from IPython import embed
from mmdet3d.core.bbox.structures import (DepthInstance3DBoxes,
                                          LiDARInstance3DBoxes,
                                          rotation_3d_in_axis)
from mmdet3d.models.builder import build_loss
from mmdet.core import multi_apply
from mmdet.models import HEADS
from torch import nn as nn
from .vote_head import VoteHead 




@HEADS.register_module()
class Points3DHead(VoteHead):
    r"""Bbox head of `3DSSD <https://arxiv.org/abs/2002.10187>`_.

    Args:
        num_classes (int): The number of class.
        bbox_coder (:obj:`BaseBBoxCoder`): Bbox coder for encoding and
            decoding boxes.
        in_channels (int): The number of input feature channel.
        train_cfg (dict): Config for training.
        test_cfg (dict): Config for testing.
        vote_module_cfg (dict): Config of VoteModule for point-wise votes.
        vote_aggregation_cfg (dict): Config of vote aggregation layer.
        pred_layer_cfg (dict): Config of classfication and regression
            prediction layers.
        conv_cfg (dict): Config of convolution in prediction layer.
        norm_cfg (dict): Config of BN in prediction layer.
        act_cfg (dict): Config of activation in prediction layer.
        objectness_loss (dict): Config of objectness loss.
        center_loss (dict): Config of center loss.
        dir_class_loss (dict): Config of direction classification loss.
        dir_res_loss (dict): Config of direction residual regression loss.
        size_res_loss (dict): Config of size residual regression loss.
        corner_loss (dict): Config of bbox corners regression loss.
        vote_loss (dict): Config of candidate points regression loss.
    """

    def __init__(self,
                 num_classes,
                 bbox_coder,
                 in_channels=256,
                 train_cfg=None,
                 test_cfg=None,
                 vote_module_cfg=None,
                 vote_aggregation_cfg=None,
                 pred_layer_cfg=None,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 act_cfg=dict(type='ReLU'),
                 objectness_loss=None,
                 center_loss=None,
                 dir_class_loss=None,
                 dir_res_loss=None,
                 size_res_loss=None,
                 corner_loss=None,
                 vote_loss=None,
                 aux_cls_loss=None,
                 aux_reg_loss=None):
        super(Points3DHead, self).__init__(
            num_classes,
            bbox_coder,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            vote_module_cfg=vote_module_cfg,
            vote_aggregation_cfg=vote_aggregation_cfg,
            pred_layer_cfg=pred_layer_cfg,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            objectness_loss=objectness_loss,
            center_loss=center_loss,
            dir_class_loss=dir_class_loss,
            dir_res_loss=dir_res_loss,
            size_class_loss=None,
            size_res_loss=size_res_loss,
            semantic_loss=None)
        
        self.corner_loss = build_loss(corner_loss)
        self.vote_loss = build_loss(vote_loss)
        self.num_candidates = vote_module_cfg['num_points']
        self.aux_cls_loss = build_loss(aux_cls_loss)
        self.aux_reg_loss = build_loss(aux_reg_loss)
        self.point_fc = nn.Linear(96, 64, bias=False)
        self.point_cls = nn.Linear(64, 1, bias=False)
        self.point_reg = nn.Linear(64, 3, bias=False)
    def _get_cls_out_channels(self):
        """Return the channel number of classification outputs."""
        # Class numbers (k) + objectness (1)
        return self.num_classes

    def _get_reg_out_channels(self):
        """Return the channel number of regression outputs."""
        # Bbox classification and regression
        # (center residual (3), size regression (3)
        # heading class+residual (num_dir_bins*2)),
        return 3 + 3 + self.num_dir_bins * 2

    def _extract_input(self, feat_dict):
        """Extract inputs from features dictionary.

        Args:
            feat_dict (dict): Feature dict from backbone.

        Returns:
            torch.Tensor: Coordinates of input points.
            torch.Tensor: Features of input points.
            torch.Tensor: Indices of input points.
        """
        #convert point_cat to batch format B ,Point_num, F
        
            
        seed_points = feat_dict.C
        seed_features = feat_dict.F
        seed_indices = feat_dict.idx_query

        return seed_points, seed_features, seed_indices
    def forward(self, feat_dict):
        """Forward pass  (SASSD)
        Note:
            the forward of Points3dhead is devided into 4 steps:
                1. point_features get throgh shared linear layer 
                2. cls branch to classify point cls
                3. if points from cls branch above threshold, reg head predict its center offset
                 
        """
        
        pointwise = self.point_fc(feat_dict.F)
        point_cls = self.point_cls(pointwise)
        point_reg = self.point_reg(pointwise)
        return (point_cls, point_reg)
    @force_fp32(apply_to=('bbox_preds', ))
    def loss(self,
             #bbox_preds,
             point_xyz,
             gt_bboxes_3d,
             gt_labels_3d,
             point_preds,
             pts_semantic_mask=None,
             pts_instance_mask=None,
             img_metas=None,
             gt_bboxes_ignore=None):
        """Compute loss.
        Args:
            bbox_preds (dict): Predictions from forward of SSD3DHead.
            points (list[torch.Tensor]): Input points.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth \
                bboxes of each sample.
            gt_labels_3d (list[torch.Tensor]): Labels of each sample.
            pts_semantic_mask (None | list[torch.Tensor]): Point-wise
                semantic mask.
            pts_instance_mask (None | list[torch.Tensor]): Point-wise
                instance mask.
            img_metas (list[dict]): Contain pcd and img's meta info.
            gt_bboxes_ignore (None | list[torch.Tensor]): Specify
                which bounding.
        Returns:
            dict: Losses of 3DSSD.
        """
        
        (point_cls, point_reg) = point_preds
        targets = self.get_targets(point_xyz, gt_bboxes_3d, gt_labels_3d,
                                   pts_semantic_mask, pts_instance_mask,
                                   )
        (points_cls_targets, points_center_targets) = targets

        
        aux_loss_cls = self.aux_cls_loss(point_cls, points_cls_targets)

        aux_loss_reg = self.aux_reg_loss(point_reg, points_center_targets)
        

        losses = dict(
            aux_loss_cls=aux_loss_cls,
            aux_loss_reg=aux_loss_reg)
        
        return losses

    def get_targets(self,
                    point_xyz,
                    gt_bboxes_3d,
                    gt_labels_3d,
                    pts_semantic_mask=None,
                    pts_instance_mask=None,
                    bbox_preds=None):
        '''
        Notice that points pass in is not original points,
        its the coor of points that match prediction points format
        (batch, x, y, z)
        '''
        ''' 
        batch_preds_coors = []
        for i in range(len(points)):
            idx = torch.nonzero(point_xyz == i).view(-1)
            batch_preds_coors.append(point_preds_coors[idx, 1:])
        '''
        pts_labels, pts_center_offset =multi_apply( self.get_targets_single, point_xyz,                             gt_bboxes_3d, gt_labels_3d)
         
        pts_center_offset = torch.cat(pts_center_offset)
        pts_labels = torch.cat(pts_labels) 
         


        return pts_labels, pts_center_offset

    def get_targets_single(self,
                           point_xyz,
                           gt_bboxes_3d,
                           gt_labels_3d,
                           pts_semantic_mask=None,
                           pts_instance_mask=None,
                           ):
        gt_bboxes_3d = gt_bboxes_3d.to(point_xyz.device) 
        valid_gt = gt_labels_3d != -1 
        gt_bboxes_3d = gt_bboxes_3d[valid_gt] 
        gt_labels_3d = gt_labels_3d[valid_gt]  
        gt_corner3d = gt_bboxes_3d.corners 
        
        (center_targets, size_targets, dir_class_targets,dir_res_targets) = self.bbox_coder.encode(gt_bboxes_3d, gt_labels_3d)

        points_mask, assignment = self._assign_targets_by_points_inside(  gt_bboxes_3d, point_xyz)
        center_offset = center_targets[assignment]
        #make front point and background point through point mask
        
        points_front = points_mask.max(1)[0]
        #TODO better to predict each points cls, instead of just front 
        return points_front, center_offset
        
    def get_bboxes(self, points, bbox_preds, input_metas, rescale=False):
       
        """Generate bboxes from sdd3d head predictions.

        Args:
            points (torch.Tensor): Input points.
            bbox_preds (dict): Predictions from sdd3d head.
            input_metas (list[dict]): Point cloud and image's meta info.
            rescale (bool): Whether to rescale bboxes.

        Returns:
            list[tuple[torch.Tensor]]: Bounding boxes, scores and labels.
        """
        # decode boxes
        sem_scores = F.sigmoid(bbox_preds['obj_scores']).transpose(1, 2)
        obj_scores = sem_scores.max(-1)[0]
        bbox3d = self.bbox_coder.decode(bbox_preds)

        batch_size = bbox3d.shape[0]
        results = list()

        for b in range(batch_size):
            bbox_selected, score_selected, labels = self.multiclass_nms_single(
                obj_scores[b], sem_scores[b], bbox3d[b], points[b, ..., :3],
                input_metas[b])
            bbox = input_metas[b]['box_type_3d'](
                bbox_selected.clone(),
                box_dim=bbox_selected.shape[-1],
                with_yaw=self.bbox_coder.with_rot)
            results.append((bbox, score_selected, labels))

        return results

    def multiclass_nms_single(self, obj_scores, sem_scores, bbox, points,
                              input_meta):
        """Multi-class nms in single batch.

        Args:
            obj_scores (torch.Tensor): Objectness score of bounding boxes.
            sem_scores (torch.Tensor): semantic class score of bounding boxes.
            bbox (torch.Tensor): Predicted bounding boxes.
            points (torch.Tensor): Input points.
            input_meta (dict): Point cloud and image's meta info.

        Returns:
            tuple[torch.Tensor]: Bounding boxes, scores and labels.
        """
        num_bbox = bbox.shape[0]
        bbox = input_meta['box_type_3d'](
            bbox.clone(),
            box_dim=bbox.shape[-1],
            with_yaw=self.bbox_coder.with_rot,
            origin=(0.5, 0.5, 1.0))

        if isinstance(bbox, LiDARInstance3DBoxes):
            box_idx = bbox.points_in_boxes(points)
            box_indices = box_idx.new_zeros([num_bbox + 1])
            box_idx[box_idx == -1] = num_bbox
            box_indices.scatter_add_(0, box_idx.long(),
                                     box_idx.new_ones(box_idx.shape))
            box_indices = box_indices[:-1]
            nonempty_box_mask = box_indices >= 0
        elif isinstance(bbox, DepthInstance3DBoxes):
            box_indices = bbox.points_in_boxes(points)
            nonempty_box_mask = box_indices.T.sum(1) >= 0
        else:
            raise NotImplementedError('Unsupported bbox type!')

        corner3d = bbox.corners
        minmax_box3d = corner3d.new(torch.Size((corner3d.shape[0], 6)))
        minmax_box3d[:, :3] = torch.min(corner3d, dim=1)[0]
        minmax_box3d[:, 3:] = torch.max(corner3d, dim=1)[0]

        bbox_classes = torch.argmax(sem_scores, -1)
        nms_selected = batched_nms(
            minmax_box3d[nonempty_box_mask][:, [0, 1, 3, 4]],
            obj_scores[nonempty_box_mask], bbox_classes[nonempty_box_mask],
            self.test_cfg.nms_cfg)[1]

        if nms_selected.shape[0] > self.test_cfg.max_output_num:
            nms_selected = nms_selected[:self.test_cfg.max_output_num]

        # filter empty boxes and boxes with low score
        scores_mask = (obj_scores >= self.test_cfg.score_thr)
        nonempty_box_inds = torch.nonzero(nonempty_box_mask).flatten()
        nonempty_mask = torch.zeros_like(bbox_classes).scatter(
            0, nonempty_box_inds[nms_selected], 1)
        selected = (nonempty_mask.bool() & scores_mask.bool())

        if self.test_cfg.per_class_proposal:
            bbox_selected, score_selected, labels = [], [], []
            for k in range(sem_scores.shape[-1]):
                bbox_selected.append(bbox[selected].tensor)
                score_selected.append(obj_scores[selected])
                labels.append(
                    torch.zeros_like(bbox_classes[selected]).fill_(k))
            bbox_selected = torch.cat(bbox_selected, 0)
            score_selected = torch.cat(score_selected, 0)
            labels = torch.cat(labels, 0)
        else:
            bbox_selected = bbox[selected].tensor
            score_selected = obj_scores[selected]
            labels = bbox_classes[selected]

        return bbox_selected, score_selected, labels

    def _assign_targets_by_points_inside(self, bboxes_3d, points):
        """Compute assignment by checking whether point is inside bbox.

        Args:
            bboxes_3d (BaseInstance3DBoxes): Instance of bounding boxes.
            points (torch.Tensor): Points of a batch.

        Returns:
            tuple[torch.Tensor]: Flags indicating whether each point is
                inside bbox and the index of box where each point are in.
        """
        # TODO: align points_in_boxes function in each box_structures
        num_bbox = bboxes_3d.tensor.shape[0]
        if isinstance(bboxes_3d, LiDARInstance3DBoxes):
            assignment = bboxes_3d.points_in_boxes(points).long()
            points_mask = assignment.new_zeros(
                [assignment.shape[0], num_bbox + 1])
            assignment[assignment == -1] = num_bbox
            points_mask.scatter_(1, assignment.unsqueeze(1), 1)
            points_mask = points_mask[:, :-1]
            assignment[assignment == num_bbox] = num_bbox - 1
        elif isinstance(bboxes_3d, DepthInstance3DBoxes):
            points_mask = bboxes_3d.points_in_boxes(points)
            assignment = points_mask.argmax(dim=-1)
        else:
            raise NotImplementedError('Unsupported bbox type!')

        return points_mask, assignment
