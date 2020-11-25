_base_ = [
    '../_base_/models/hv_spvcnn_secfpn_kitti.py',
    '../_base_/datasets/kitti-3d-3class.py',
    '../_base_/schedules/cyclic_40e.py', '../_base_/default_runtime.py'
]



model = dict(
    middle_encoder=dict(
        type='SPVCNNV2',
        in_channels=4,
        tobev_shape = [200,176],
        sparse_shape=[41, 1600, 1408],
        order=('conv', 'norm', 'act')), 
        )

