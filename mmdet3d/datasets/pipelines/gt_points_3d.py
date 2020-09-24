from IPython import embed
from mmdet.datasets.builder import PIPELINES 
import numpy as np
import time
from mmdet3d.core.bbox import box_np_ops as box_np_ops 



@PIPELINES.register_module()
class GT_Points_3D(object):
    def __init__(self):
        pass
    def __call__(self, input_dict):  
        
        #start = time.time()
        points = input_dict['points']
        gt_bboxes_3d = input_dict['gt_bboxes_3d'].tensor.numpy()
        gt_labels_3d = input_dict['gt_labels_3d']
        points_indices = box_np_ops.points_in_rbbox(points, gt_bboxes_3d)
        points_num_in_the_box =  points_indices.sum(axis=0)
        # filter min points -> can write a function 
        filter_min_points=[]
        for i in range(len(gt_labels_3d)):
            if points_num_in_the_box[i] > 5:
                filter_min_points.append(i)
        gt_bboxes_3d = input_dict['gt_bboxes_3d'][filter_min_points]
        gt_labels_3d = gt_labels_3d[filter_min_points]
        points_indices = points_indices[:,filter_min_points]
        
        #calculate the gt_points_3d
        gt_points_3d = [] 
        for i in range(len(gt_bboxes_3d)):
            gt_points = points[points_indices[:, i]] 
            gt_points_3d_mean = np.mean(gt_points,axis=0)
            gt_points_3d_median = np.median(gt_points, axis=0) 
            gt_points_3d_length = np.array(max(gt_points[:,0])-min(gt_points[:,0])).reshape(1)
            gt_points_3d_width = np.array(max(gt_points[:,1])-min(gt_points[:,1])).reshape(1)
            gt_points_3d_height = np.array(max(gt_points[:,2])-min(gt_points[:,2])).reshape(1)
            gt_points_3d_obj = np.concatenate((gt_points_3d_mean, gt_points_3d_median, 
                                                gt_points_3d_length,gt_points_3d_width
                                                ,gt_points_3d_height), axis=0) 
            gt_points_3d.append(gt_points_3d_obj)
                                               
        
        gt_points_3d = np.stack( gt_points_3d )
        input_dict['gt_points_3d'] = gt_points_3d
        input_dict['gt_labels_3d'] = gt_labels_3d
        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
        #end = time.time()
        #print(end - start)
        return input_dict
    #def filter_min_points(gt_bboxes_3d , points_indices):
        
        
