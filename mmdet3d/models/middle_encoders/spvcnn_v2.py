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


__all__ = ['SPVCNNV2']


class BasicConvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                                 outc,
                                 kernel_size=ks,
                                 dilation=dilation,
                                 stride=stride), spnn.BatchNorm(outc),
            spnn.ReLU(True))

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
            spnn.ReLU(True))

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
            spnn.ReLU(True),
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

        self.relu = spnn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out

@MIDDLE_ENCODERS.register_module()
class SPVCNNV2(nn.Module):
    def __init__(self, in_channels=4,sparse_shape=[41, 1600,1408],tobev_shape=[200,176],output_channels=256, **kwargs):
        super().__init__()

        cr = kwargs.get('cr', 1.0)
        cs = [32, 64, 64, 128, 256, 256, 128, 96, 96]
        cs = [int(cr * x) for x in cs]
        self.bev_shape = tobev_shape
        self.output_channels= output_channels
        self.stem = nn.Sequential(
            spnn.Conv3d(in_channels, cs[0], kernel_size=1, stride=1),
            spnn.BatchNorm(cs[0]), spnn.ReLU(True),
            )

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(cs[0], cs[1], ks=3, stride=2, dilation=1),
            ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(cs[1], cs[2], ks=3, stride=2, dilation=1),
            ResidualBlock(cs[2], cs[2], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[2], cs[2], ks=3, stride=1, dilation=1),
        )

        self.stage3 = nn.Sequential(
            BasicConvolutionBlock(cs[2], cs[3], ks=3, stride=2, dilation=1),
            ResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1),
        )
       
        self.stage4 = nn.Sequential(
            BasicConvolutionBlock(cs[3], cs[3], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1),
        )
        self.stage_out = nn.Sequential(
            BasicConvolutionBlock(cs[3], cs[3], ks=1, stride=1, dilation=1), 
        )
        self.to_bev = nn.Sequential(
            spnn.ToDenseBEVConvolution(in_channels=cs[3], out_channels=self.output_channels,shape=np.array([self.bev_shape[0],self.bev_shape[1],5,1])),
             nn.BatchNorm2d(self.output_channels),
             nn.ReLU(True),
        )
        ''' 
        self.point_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cs[1], cs[1]),
                nn.BatchNorm1d(cs[1]),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Linear(cs[2], cs[2]),
                nn.BatchNorm1d(cs[2]),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Linear(cs[3], cs[3]),
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
        z = PointTensor(x.F, x.C.float())
        #x0 = point_to_voxel(x, z) 
        x0 = self.stem(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        out = self.stage_out(x4) 
        spatial_feature = self.to_bev(out)
        

        
        #z0 = voxel_to_point(x0, z)
        #z0.F = z0.F + self.point_transforms[0](z.F)
        #z1 = voxel_to_point(x1, z)
        #z1.F = z1.F + self.point_transforms[1](z0.F)
        #z2 = voxel_to_point(x2, z)
        #z2.F = z2.F + self.point_transforms[2](z1.F)
        #z3 = voxel_to_point(x3, z)
        #z3.F = z3.F + self.point_transforms[3](z2.F)
        #z3.F = torch.cat([z0.F,z1.F,z2.F,z3.F], dim=-1)
        


        '''
        z1 = voxel_to_point(x4, z0)
        z1.F = z1.F + self.point_transforms[0](z0.F)

        y1 = point_to_voxel(x4, z1)
        y1.F = self.dropout(y1.F)
        y1 = self.up1[0](y1)
        y1 = torchsparse.cat([y1, x3])
        y1 = self.up1[1](y1)
        
        
        y2 = self.up2[0](y1)
        y2 = torchsparse.cat([y2, x2])
        y2 = self.up2[1](y2)
        z2 = voxel_to_point(y2, z1)
        z2.F = z2.F + self.point_transforms[1](z1.F)

        y3 = point_to_voxel(y2, z2)
        y3.F = self.dropout(y3.F)
        y3 = self.up3[0](y3)
        y3 = torchsparse.cat([y3, x1])
        y3 = self.up3[1](y3)

        y4 = self.up4[0](y3)
        y4 = torchsparse.cat([y4, x0])
        y4 = self.up4[1](y4)
        spatial_feature = self.to_bev(y1)
        
        z3 = voxel_to_point(y4, z2)
        z3.F = z3.F + self.point_transforms[2](z2.F)
        
        #TODO  use collate function built in torchsparse
        batch_size = x.C[:, 3][-1] + 1
        point_xyz=[] 
        for i in range(batch_size):
            inds = (x.C[:, 3]==i)
            point_xyz.append(x.F[:, :3][inds])
        '''    
          
        
        #point_xyz= torch.stack(point_xyz)
        #point_feature = torch.stack(point_feature)
        #feature_dict = {'voxel_feature':tuple(voxel_feature_outs)}
        #point_xyz= x.F[:, :3]
        return spatial_feature
        #feature_dict = {'voxel_feature':spatial_feature, 'point_feature': z3 ,'point_xyz':point_xyz}
        #return feature_dict
