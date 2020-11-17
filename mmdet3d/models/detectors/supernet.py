from mmdet3d.core import bbox3d2result
from mmdet.models import DETECTORS
from .mvx_two_stage import MVXTwoStageDetector
from IPython import embed
from torchsparse.utils import  sparse_quantize
import numpy as np
from torchsparse.sparse_tensor import SparseTensor
import torch
from .. import builder


@DETECTORS.register_module()
class SuperNet(MVXTwoStageDetector):
    """Base class of Multi-modality VoxelNet."""

    def __init__(self,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 pts_points_head=None,
                 pts_points_seg_head=None,
                 pts_points_ins_head=None,
                 pts_roi_head=None, 
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SuperNet,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, pts_points_head,
                             pts_roi_head, 
                             img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        if pts_points_seg_head:
            self.pts_points_seg_head = builder.build_head(pts_points_seg_head) 
        if pts_points_ins_head:
            self.pts_points_ins_head = builder.build_head(pts_points_ins_head)
    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        ''' 
        #TODO make spvcnn sparsetensor
        points  = pts[0].cpu().detach().numpy() 
        pc_ = np.round(points[:, :3] / 0.05)
        pc_ -= pc_.min(0, keepdims=1)
        inds = sparse_quantize(pc_,points, return_index=True)
        voxel_features = pts[0][inds]
        pc =torch.tensor(pc_[inds]).to(device=0)       
        coors=pc
        '''
        batch_size = coors[-1, 0] + 1
        feat_dict = self.pts_middle_encoder(voxel_features, coors, batch_size)
        
        return feat_dict
    def forward_pts_train(self,
                          points,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          gt_seg_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        losses={}
        #point-base branch :
        point_xyz = pts_feats['point_xyz']
        if self.pts_points_head:
            point_preds = self.pts_points_head(pts_feats['point_feature'])
            points_loss_inputs = [point_xyz ,gt_bboxes_3d, gt_labels_3d, point_preds]
            points_losses = self.pts_points_head.loss(*points_loss_inputs)
            
            losses.update(points_losses)
        
        if self.pts_points_ins_head:
            point_ins_preds = self.pts_points_ins_head(pts_feats['point_feature'])
            points_loss_inputs = [point_xyz ,gt_bboxes_3d, gt_labels_3d, point_ins_preds]
            points_ins_losses = self.pts_points_ins_head.loss(*points_loss_inputs)
            losses.update(points_ins_losses) 
        if self.pts_roi_head:
            bboxs_preds = self.pts_roi_head(point_xyz, point_preds, point_ins_preds, pts_feats['point_feature']) 
            if bboxs_preds != None:    
                points_bboxs_loss_inputs = [bboxs_preds, points ,gt_bboxes_3d, gt_labels_3d, point_xyz]
                points_bboxs_losses = self.pts_roi_head.loss(*points_bboxs_loss_inputs) 
                losses.update(points_bboxs_losses)
        
        if self.pts_bbox_head:
            #voxel-base branch
            x = self.pts_backbone(pts_feats['voxel_feature']) 
            x = self.pts_neck(x) 
            voxel_preds = self.pts_bbox_head(x)
            voxel_loss_inputs = [gt_bboxes_3d, gt_labels_3d, voxel_preds]
            voxel_losses = self.pts_bbox_head.loss(*voxel_loss_inputs)
            losses.update(voxel_losses)   
        self.simple_vis(point_xyz, point_preds, point_ins_preds,gt_bboxes_3d)
        
        return losses
    def simple_vis(self, point_xyz, point_preds, point_ins_preds,gt_bbox,bbox_results=None):   
        import pickle                                
        with open('./vis/points.pkl', 'wb') as f:      
            pickle.dump(point_xyz, f)         
        with open('./vis/point_preds.pkl', 'wb') as f:      
            pickle.dump(point_preds, f)   
        with open('./vis/point_ins_preds.pkl', 'wb') as f: 
            pickle.dump(point_ins_preds, f)
        with open('./vis/gt_bbox.pkl', 'wb') as f:   
            pickle.dump(gt_bbox, f)                         
        if bbox_results:
            with open('./vis/bbox_results.pkl', 'wb') as f:       
                pickle.dump(bbox_results, f)                                       
    def simple_test_pts(self, pts_feats, img_metas, rescale=False):
        """Test function of point cloud branch."""
        '''
        point_xyz = pts_feats['point_xyz'] 
        point_preds = self.pts_points_head(pts_feats['point_feature'])
        bboxs_preds = self.pts_roi_head(point_xyz, point_preds, pts_feats['point_feature']) 
        bbox_list = self.pts_roi_head.get_bboxes(point_xyz ,bboxs_preds
                                                 , img_metas, rescale=rescale)
        '''
        x = self.pts_backbone(pts_feats['voxel_feature'])

        x = self.pts_neck(x)
        outs = self.pts_bbox_head(x) 
        bbox_list = self.pts_bbox_head.get_bboxes( outs, img_metas, rescale=rescale) 
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results
