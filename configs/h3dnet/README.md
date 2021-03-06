# H3DNet: 3D Object Detection Using Hybrid Geometric Primitives

## Introduction
We implement H3DNet and provide the result and checkpoints on ScanNet datasets.
```
@inproceedings{zhang2020h3dnet,
    author = {Zhang, Zaiwei and Sun, Bo and Yang, Haitao and Huang, Qixing},
    title = {H3DNet: 3D Object Detection Using Hybrid Geometric Primitives},
    booktitle = {Proceedings of the European Conference on Computer Vision},
    year = {2020}
}
```

## Results

### ScanNet
|  Backbone   | Lr schd | Mem (GB) | Inf time (fps) | AP@0.25 |AP@0.5| Download |
| :---------: | :-----: | :------: | :------------: | :----: |:----: | :------: |
|    [MultiBackbone](./h3dnet_scannet-3d-18class.py)     |  3x    |7.9||66.43|48.01|[model](https://openmmlab.oss-accelerate.aliyuncs.com/mmdetection3d/v0.1.0_models/votenet/votenet_8x8_scannet-3d-18class/votenet_8x8_scannet-3d-18class_20200620_230238-2cea9c3a.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmdetection3d/v0.1.0_models/votenet/votenet_8x8_scannet-3d-18class/votenet_8x8_scannet-3d-18class_20200620_230238.log.json)|
