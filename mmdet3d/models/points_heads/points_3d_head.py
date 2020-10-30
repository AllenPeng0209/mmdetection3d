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
from mmdet3d.ops.group_points import QueryAndGroup 
from mmdet.core import build_bbox_coder

@HEADS.register_module()
class Points3DHead(nn.Module):
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
                 front_point_loss=None,
                 center_offset_loss=None,
                 gravity_loss=None,
                 corner_loss=None,
                 semantic_loss=None,
                 ):
        super(Points3DHead, self).__init__()
        self.num_classes = num_classes
        self.bbox_coder = build_bbox_coder(bbox_coder)  
        self.front_point_loss = build_loss(front_point_loss)
        self.center_offset_loss = build_loss(center_offset_loss)
        self.point_fc = nn.Linear(96, 64, bias=False)
        self.point_front = nn.Linear(64, 1, bias=False)
        self.point_center_offset = nn.Linear(64, 3, bias=False)
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

    def _extract_input(self,point_xyz, feat_dict):
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
        point_front = self.point_front(pointwise)
        point_center_offset = self.point_center_offset(pointwise)   
        return (point_front, point_center_offset)
        
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
         
        (point_front, point_center_offset) = point_preds
        targets = self.get_targets(point_xyz, gt_bboxes_3d, gt_labels_3d,
                                   pts_semantic_mask, pts_instance_mask,
                                   )
        (points_front_targets, points_center_offset_targets) = targets
        
        front_point_loss = self.front_point_loss(point_front, points_front_targets)

        center_offset_loss = self.center_offset_loss(point_center_offset, points_center_offset_targets)
        
        losses = dict(
            front_point_loss=front_point_loss,
            center_offset_loss=center_offset_loss,
            )
        
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
        
        (center_targets, size_targets, dir_class_targets,dir_res_targets, extra_targets) = self.bbox_coder.encode(gt_bboxes_3d, gt_labels_3d)
         
        points_mask, assignment = self._assign_targets_by_points_inside(gt_bboxes_3d, point_xyz)
        num_bbox = len(gt_bboxes_3d.tensor)
        point_front = points_mask.max(1)[0]
        
        #assign center to each points
        back_ground_target = torch.tensor([[0,0,0]]).to(device=center_targets.device)
        center_targets = torch.cat((center_targets,back_ground_target))
        points_center = center_targets[assignment]
        #calculate center offset to center
        points_center_offset = point_xyz - points_center
        front_point_ = torch.stack((point_front,point_front,point_front),axis=1)
        points_center_offset = front_point_ * points_center_offset
        
        '''
        point_cls = torch.zeros((point_front.shape[0],self.num_classes+1))
        #make instance label 
        for i, label in enumerate(gt_labels_3d):
            point_cls[:,label] += assignment[assignment==i]
        #back ground label
        assignment[assignment==assignment.max()]=self.nums_class+1
        '''
        return point_front, points_center_offset
        


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
            #assignment[assignment == num_bbox] = num_bbox - 1 
        elif isinstance(bboxes_3d, DepthInstance3DBoxes):
            points_mask = bboxes_3d.points_in_boxes(points)
            assignment = points_mask.argmax(dim=-1)
        else:
            raise NotImplementedError('Unsupported bbox type!')

        return points_mask, assignment
