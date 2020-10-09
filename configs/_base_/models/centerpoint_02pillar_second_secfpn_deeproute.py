voxel_size = [0.2, 0.2, 8]
model = dict(
    type='CenterPoint',
    pts_voxel_layer=dict(
        max_num_points=20, voxel_size=voxel_size, max_voxels=(30000, 40000)),
    pts_voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=3,
        feat_channels=[64],
        with_distance=False,
        voxel_size=(0.2, 0.2, 8),
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        legacy=False),
    pts_middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=(800, 800)),
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
        in_channels=sum([128, 128, 128]),
        tasks=[
            dict(num_class=4, class_names=['CAR','CAR_HARD','VAN','VAN_HARD']),
            dict(num_class=5, class_names=['TRUCK','TRUCK_HARD','BIG_TRUCK','BUS','BUS_HARD']),
            dict(num_class=3, class_names=['PEDESTRIAN', 'PEDESTRIAN_HARD','CONE']),
            dict(num_class=4, class_names=['CYCLIST','CYCLIST_HARD','TRICYCLE','TRICYCLE_HARD']),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            post_center_range=[-80, -80, -10.0, 80, 80, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=4,
            voxel_size=voxel_size[:2],
            code_size=7),
        seperate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True))
# model training and testing settings
train_cfg = dict(
    pts=dict(
        grid_size=[800, 800, 1],
        voxel_size=voxel_size,
        out_size_factor=4,
        dense_reg=1,
        gaussian_overlap=0.1,
        max_objs=500,
        min_radius=2,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
test_cfg = dict(
    pts=dict(
        post_center_limit_range=[-80, -80, -10.0, 80, 80, 10.0],
        max_per_img=500,
        max_pool_nms=False,
        min_radius=[4, 12, 0.175, 0.85],
        post_max_size=83,
        score_threshold=0.1,
        pc_range=[-80, -80],
        out_size_factor=4,
        voxel_size=voxel_size[:2],
        nms_type='rotate',
        nms_pre_max_size=1000,
        nms_post_max_size=83,
        nms_iou_threshold=0.2))
