_base_ = [
    '../_base_/models/deeproute_centerpoint_01_voxel_dcn_circle_nms.py',
    '../_base_/default_runtime.py'
]

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-80, -80, -5.0, 80, 80, 3.0]
# For nuScenes we usually do 10-class detection
class_names = [
'CAR','CAR_HARD','VAN','VAN_HARD','TRUCK','TRUCK_HARD','BIG_TRUCK','BUS','BUS_HARD','PEDESTRIAN',
'PEDESTRIAN_HARD', 'CYCLIST','CYCLIST_HARD','TRICYCLE','TRICYCLE_HARD','CONE']
dataset_type = 'DeeprouteDataset'
data_root = 'data/deeproute_mini/'
# Input modality for nuScenes dataset, this is consistent with the submission
# format which requires the information in input_modality.
input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False)
file_client_args = dict(backend='disk')

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'deeproute_mini_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            CAR=5,
            CAR_HARD=5,
            VAN=5,
            VAN_HARD=5,
            TRUCK=5,
            TRUCK_HARD=5,
            BIG_TRUCK=5,
            BUS =5,
            BUS_HARD=5,
            PEDESTRIAN=5,
            PEDESTRIAN_HARD=5,
            CYCLIST=5,
            CYCLIST_HARD=5,
            TRICYCLE=5,
            TRICYCLE_HARD=5,
            CONE=5,

        )),
    classes=class_names,
    sample_groups=dict( 
            CAR=2,
            CAR_HARD=2,
            VAN=2,
            VAN_HARD=2,
            TRUCK=3,
            TRUCK_HARD=3,
            BIG_TRUCK=3,
            BUS =3,
            BUS_HARD=3,
            PEDESTRIAN=4,
            PEDESTRIAN_HARD=4,
            CYCLIST=5,
            CYCLIST_HARD=5,
            TRICYCLE=5,
            TRICYCLE_HARD=5,
            CONE=6,

    ),
    points_loader=dict(
        type='LoadPointsFromFile',
        load_dim=3,
        use_dim=[0, 1, 2],
        file_client_args=file_client_args))
    

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        load_dim=3,
        use_dim=3,
        file_client_args=file_client_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_points_3d= True),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='GT_Points_3D'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_points_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        load_dim=3,
        use_dim=3,
        file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(type='GT_Points_3D') ,
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
    
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    train=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'deeproute_mini_infos_train.pkl',
            pipeline=train_pipeline,
            classes=class_names,
            modality=input_modality,
            test_mode=False,
            #use_valid_flag=True,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR')),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'deeproute_mini_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'deeproute_mini_infos_test.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'))
# For nuScenes dataset, we usually evaluate the model at the end of training.
# Since the models are trained by 24 epochs by default, we set evaluation
# interval to be 24. Please change the interval accordingly if you do not
# use a default schedule.
# optimizer
# This schedule is mainly used by models on nuScenes dataset
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=0.01)
# max_norm=10 is better for SECOND
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='cyclic',
    target_ratio=(10, 1e-4),
    cyclic_times=1,
    step_ratio_up=0.4,
)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.85 / 0.95, 1),
    cyclic_times=1,
    step_ratio_up=0.4,
)

# runtime settings
total_epochs = 20
evaluation = dict(interval=20)
