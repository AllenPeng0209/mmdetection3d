_base_ = [
    '../_base_/datasets/deeproute-3d.py',
    '../_base_/models/centerpoint_02pillar_second_secfpn_deeproute.py',
    '../_base_/schedules/cyclic_20e.py', '../_base_/default_runtime.py'
]

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-80, -80, -5.0, 80, 80, 3.0]
# For nuScenes we usually do 10-class detection
class_names = [
'CAR','CAR_HARD','VAN','VAN_HARD','TRUCK','TRUCK_HARD','BIG_TRUCK','BUS','BUS_HARD','PEDESTRIAN',
'PEDESTRIAN_HARD', 'CYCLIST','CYCLIST_HARD','TRICYCLE','TRICYCLE_HARD','CONE']

input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False)



model = dict(
    pts_voxel_layer=dict(point_cloud_range=point_cloud_range),
    pts_voxel_encoder=dict(point_cloud_range=point_cloud_range),
    pts_bbox_head=dict(bbox_coder=dict(pc_range=point_cloud_range[:2])))
# model training and testing settings
train_cfg = dict(pts=dict(point_cloud_range=point_cloud_range))
test_cfg = dict(pts=dict(pc_range=point_cloud_range[:2]))

dataset_type = 'DeeprouteDataset'
data_root = 'data/deeproute/'
file_client_args = dict(backend='disk')

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'deeproute_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            CAR=0,
            CAR_HARD=0,
            VAN=0,
            VAN_HARD=0,
            TRUCK=0,
            TRUCK_HARD=0,
            BIG_TRUCK=0,
            BUS =0,
            BUS_HARD=0,
            PEDESTRIAN=0,
            PEDESTRIAN_HARD=0,
            CYCLIST=0,
            CYCLIST_HARD=0,
            TRICYCLE=0,
            TRICYCLE_HARD=0,
            CONE=0,
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
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
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
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
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
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]

data = dict(
    train=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'deeproute_infos_train.pkl',
            pipeline=train_pipeline,
            classes=class_names,
            modality=input_modality,
            test_mode=False,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR')),
    val=dict(pipeline=test_pipeline, classes=class_names),
    test=dict(pipeline=test_pipeline, classes=class_names))

evaluation = dict(interval=1)
