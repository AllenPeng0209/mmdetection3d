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
import numpy as np
from torchsparse.utils import sparse_quantize

from mmdet3d.ops import Voxelization
import torchsparse

@HEADS.register_module()
class PointsSemHead(nn.Module):
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
                 in_channels=96,
                 train_cfg=None,
                 test_cfg=None,
                 semantic_loss=None,
                 seg_voxel=None,
                 ):
        super(PointsSegHead, self).__init__()
        self.num_classes = num_classes
        self.point_seg = nn.Linear(in_channels, num_classes, bias=False)
        self.point_seg_loss = build_loss(semantic_loss)
        #self.seg_voxel = Voxelization(seg_voxel) 
    def forward(self, feat_dict):
        point_seg = self.point_seg(feat_dict.F)
        return point_seg
        
    def loss(self,
             points,
             point_xyz,
             gt_seg_3d,
             point_preds,
             ):
        #TODO only predict original label points , other augmentation points ignore
         
        source_coords = np.round(points[0][:,:3].clone().detach().cpu()/0.05)
        #source_coords -= source_coords.min(0, keepdims=1)
        target_coords = np.round(point_xyz[0][:,:3].clone().detach().cpu()/0.05)
        #target_coords -= target_coords.min(0, keepdims=1)
        source_hash = torchsparse.nn.functional.sphash(torch.floor(source_coords).int())
        target_hash = torchsparse.nn.functional.sphash(torch.floor(target_coords).int())
        idx_query = torchsparse.nn.functional.sphashquery(target_hash, source_hash)
        gt_seg_3d_xyz = gt_seg_3d[0][idx_query]
        #TODO visualize final label and its corr, and original label and label
        self.output_pickle_for_vis(points, point_xyz, gt_seg_3d, gt_seg_3d_xyz)
        #gt_seg = self.get_target(points, gt_seg_3d)
        #point_seg_loss = self.point_seg_loss(gt_seg_3d,point_preds)
        #return point_seg_loss
  
    def get_target(self, points, gt_seg_3d):
        #TODO make it to batch form 
        
        return labels


    #TODO this is dirty function, may refactory to vis tool later.
    def output_pickle_for_vis(self, points, point_xyz, gt_seg_3d, gt_seg_3d_xyz):
        import pickle 
        with open('vis/points.pkl','wb') as f: 
            pickle.dump(points[0][:,:3],f)
        with open('vis/point_xyz.pkl','wb') as f: 
            pickle.dump(point_xyz[0][:,:3],f)
        with open('vis/gt_seg_3d.pkl','wb') as f: 
            pickle.dump(gt_seg_3d[0],f)
        with open('vis/gt_seg_3d_xyz.pkl','wb') as f:  
            pickle.dump(gt_seg_3d_xyz,f)
      
        













