import time
from collections import OrderedDict
from ..registry import MIDDLE_ENCODERS
import torch
import torch.nn as nn
from mmdet3d.ops import spconv as spconv
import torchsparse
import torchsparse.nn as spnn
import torchsparse.nn.functional as spf
from torchsparse.sparse_tensor import SparseTensor
from torchsparse.point_tensor import PointTensor
from torchsparse.utils.kernel_region import *
from torchsparse.utils.helpers import *
from IPython import embed
from mmdet3d.models.utils import *


__all__ = ['SPVCNN_DET']


class BasicConvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                                 outc,
                                 kernel_size=ks,
                                 dilation=dilation,
                                 stride=stride),
            spnn.BatchNorm(outc, eps=1e-3, momentum=0.01),
            spnn.ReLU())
    def forward(self, x):
        out = self.net(x)
        return out

class StrideConvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=2, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc, outc, kernel_size=ks, dilation=dilation,  stride=stride),
            spnn.BatchNorm(outc, eps=1e-3, momentum=0.01),
            spnn.ReLU())

    def forward(self, x):
        out = self.net(x)
        return out



class BasicDeconvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                                 outc,
                                 kernel_size=ks,
                                 stride=stride,
                                 transpose=True), spnn.BatchNorm(outc),
            spnn.ReLU())

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                                 outc,
                                 kernel_size=ks,
                                 dilation=dilation,
                                 stride=stride), spnn.BatchNorm(outc),
            spnn.ReLU(),
            spnn.Conv3d(outc,
                                 outc,
                                 kernel_size=ks,
                                 dilation=dilation,
                                 stride=1), spnn.BatchNorm(outc))

        self.downsample = nn.Sequential() if (inc == outc and stride == 1) else \
            nn.Sequential(
                spnn.Conv3d(inc, outc, kernel_size=1, dilation=1, stride=stride),
                spnn.BatchNorm(outc)
            )

        self.relu = spnn.ReLU()

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out

@MIDDLE_ENCODERS.register_module()
class SPVCNNDET(nn.Module):
    def __init__(self, in_channels=4,sparse_shape=[41, 1600,1408],tobev_shape=[200,176],output_channels=256, **kwargs):
        super().__init__()
        self.classes =['Car', 'Cyclist', 'Pedestrian']
        self.VOXEL_SIZE=0.2
        cr = kwargs.get('cr', 1.0)
        cs = [16, 32, 64, 128, 128, 256, 128, 96, 96]
        cs = [int(cr * x) for x in cs]
        self.bev_shape = tobev_shape
        self.pres = 0.05
        self.vres = 0.05
        '''
        self.PC_AREA_SCOPE = np.array([[-40, 40], [-1, 3], [0, 70.4]], 'float32')
        self.GT_AREA_SCOPE = {
            'Car': np.array([[-40, 40], [-1, 3], [0, 70.4]], 'float32'),
            'Cyclist': np.array([[-20, 20], [-1, 3], [0, 48]], 'float32'),
            'Pedestrian': np.array([[-20, 20], [-1, 3], [0, 48]], 'float32')
             }
        self.LOC_MIN = dict([(c, np.array(np.round((self.GT_AREA_SCOPE[c][:, 0] - self.PC_AREA_SCOPE[:, 0]) / self.VOXEL_SIZE), 'int32')) for c in self.classes])
        self.LOC_MAX = dict([(c, np.array(np.round((self.GT_AREA_SCOPE[c][:, 1] - self.PC_AREA_SCOPE[:, 0]) / self.VOXEL_SIZE), 'int32')) for c in self.classes])
        self.proposal_stride = [8,8,4]
        '''
        self.output_channels= output_channels
        self.stem = nn.Sequential(
            BasicConvolutionBlock(in_channels, cs[0], ks=3, stride=1, dilation=1),
            BasicConvolutionBlock(cs[0], cs[0], ks=3, stride=1, dilation=1),     
            )

        self.stage1 = nn.Sequential(
            StrideConvolutionBlock(cs[0], cs[1], ks=3, stride=2, dilation=1),   
            BasicConvolutionBlock(cs[1], cs[1], ks=3, stride=1, dilation=1),
            BasicConvolutionBlock(cs[1], cs[1], ks=3, stride=1, dilation=1),
        )

        self.stage2 = nn.Sequential(
            StrideConvolutionBlock(cs[1], cs[2], ks=3, stride=2, dilation=1),
            BasicConvolutionBlock(cs[2], cs[2], ks=3, stride=1, dilation=1),
            BasicConvolutionBlock(cs[2], cs[2], ks=3, stride=1, dilation=1),
            BasicConvolutionBlock(cs[2], cs[2], ks=3, stride=1, dilation=1),
        )

        self.stage3 = nn.Sequential(
            StrideConvolutionBlock(cs[2], cs[3], ks=3, stride=2, dilation=1),
            BasicConvolutionBlock(cs[3], cs[3], ks=3, stride=1, dilation=1),
            BasicConvolutionBlock(cs[3], cs[3], ks=3, stride=1, dilation=1),
            BasicConvolutionBlock(cs[3], cs[3], ks=3, stride=1, dilation=1),
        )
        
        self.stage4 = nn.Sequential( 
            BasicConvolutionBlock(cs[3], cs[3], ks=3, stride=1, dilation=1), 
            BasicConvolutionBlock(cs[3], cs[3], ks=3, stride=1, dilation=1),
            BasicConvolutionBlock(cs[3], cs[3], ks=3, stride=1, dilation=1),
        )
        
        self.stage_out = nn.Sequential(
            BasicConvolutionBlock(cs[3], cs[3], ks=1, stride=1, dilation=1),
        )
        self.to_bev = nn.Sequential(
            spnn.ToDenseBEVConvolution(in_channels=cs[3], out_channels=self.output_channels,shape=np.array([self.bev_shape[0] ,self.bev_shape[1],5,1])),
            nn.BatchNorm2d(self.output_channels),
            nn.ReLU(),
        )
        ''' 
        self.point_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cs[1], cs[1]),
                nn.BatchNorm1d(cs[1]),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Linear(cs[1], cs[2]),
                nn.BatchNorm1d(cs[2]),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Linear(cs[2], cs[3]),
                nn.BatchNorm1d(cs[3]),
                nn.ReLU(True),
            )
        ])
        
        self.weight_initialization()
        self.dropout = nn.Dropout(0.3, True)
        '''
    def init_weights(self, pretrained=None):
        pass
    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, voxel_features, coors, batch_size):
        # x: SparseTensor z: PointTensor
        x = SparseTensor(voxel_features, coors[:,[2,3,1,0]].contiguous())

        #z = PointTensor(x.F, x.C.float())
        #x0 = point_to_voxel(x, z)
        


        x0 = self.stem(x)
        #x1 = point_to_voxel(x0, z0)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        out = self.stage_out(x4)
 
        '''
        z1 = voxel_to_point(x1, z0)
        z1.F = z1.F + self.point_transforms[0](z0.F) 
        y1 = point_to_voxel(x1, z1) 
        x2 = self.stage2(y1)
        z2 = voxel_to_point(x2, z0)
        z2.F = z2.F + self.point_transforms[1](z1.F)
        y2 = point_to_voxel(x2, z2) 
        x3 = self.stage3(y2) 
        '''     
        spatial_feature = self.to_bev(out)
        
        return spatial_feature
