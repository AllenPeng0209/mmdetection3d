import torch
import bev_nms_cuda
import time

x = torch.cuda.FloatTensor([[[0,0], [1, 0], [1,1], [0,1.]],
                            [[0,0], [1, 0], [1,1], [0,1.]],
                            [[ 0.5005, 22.9002],
                                [-0.3002, 22.8996],
                                [-0.3002, 22.1007],
                                [ 0.5000, 22.1007]]
                            ], device=1)

# x = torch.cuda.FloatTensor([[[0,0], [0, 1], [1,1], [1 ,0.]],
#                             [[0,0], [0,1.], [1,1], [1, 0]],
#                             [[0,0], [0,0.5], [0.5,0.5], [0.5, 0]]
#                             ], device=1)

x = x.repeat(2000, 1, 1)
# y = torch.cuda.FloatTensor([[[0,0], [1, 0], [1,1], [0,1.]]])
print(x.size())
scores = torch.cuda.FloatTensor([1.000, 0.83498, 0.9231], device=1).repeat(2000, 1, 1)
cat = torch.cuda.FloatTensor([1., 1., 2.], device=1).repeat(2000, 1, 1)
threshold = torch.cuda.FloatTensor([0.5000, 0.5000, 0.2000], device=1).repeat(2000, 1, 1)

out = bev_nms_cuda.bev_nms(x, scores, cat, threshold)
print(out)
print(out.shape)