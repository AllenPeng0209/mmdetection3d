voxel_size = [0.25, 0.25, 8]
model = dict(
    type='CenterPointPP',
    pts_voxel_layer=dict(
        max_num_points=64,
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        voxel_size=voxel_size,
        max_voxels=(30000, 40000)),
    pts_voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=5,
        feat_channels=[64],
        with_distance=False,
        voxel_size=(0.2, 0.2, 8),
        point_cloud_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
    ),
    pts_middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=(512, 512)),
    pts_backbone=dict(
        type='SECOND',
        in_channels=64,
        out_channels=[64, 128, 256],
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        out_channels=[128, 128, 128],
        upsample_strides=[0.5, 1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    pts_bbox_head=dict(
        type='CenterHead',
        mode='3d',
        in_channels=sum([128, 128, 128]),
        tasks=[
            dict(num_class=1, class_names=['car']),
            dict(num_class=2, class_names=['truck', 'construction_vehicle']),
            dict(num_class=2, class_names=['bus', 'trailer']),
            dict(num_class=1, class_names=['barrier']),
            dict(num_class=2, class_names=['motorcycle', 'bicycle']),
            dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
        ],
        dataset='nuscenes',
        weight=0.25,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 1.0, 1.0],
        common_heads={
            'reg': (2, 2),
            'height': (1, 2),
            'dim': (3, 2),
            'rot': (2, 2),
            'vel': (2, 2)
        },
        share_conv_channel=64,
        dcn_head=True))
# model training and testing settings
train_cfg = dict(
    pts=dict(
        grid_size=[1024, 1024, 40],
        point_cloud_range=[-51.2, -51.2, -5., 51.2, 51.2, 3.],
        voxel_size=[0.1, 0.1, 0.2],
        out_size_factor=8,
        dense_reg=1,
        gaussian_overlap=0.1,
        max_objs=500,
        min_radius=2,
        no_log=False))
test_cfg = dict(
    pts=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_pre=1000,
        nms_thr=0.2,
        score_thr=0.05,
        min_bbox_size=0,
        max_num=500))