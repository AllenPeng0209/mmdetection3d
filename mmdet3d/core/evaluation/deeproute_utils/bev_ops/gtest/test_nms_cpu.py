import torch
import rotate_iou_cpp
import time


x = torch.FloatTensor([[[0,0], [1, 0], [1,1], [0,1.]],
                            [[0,0], [1, 0], [1,1], [0,1.]],
                            [[0,0], [0.5, 0], [0.5,0.5], [0,0.5]]
                            ])
# y = torch.cuda.FloatTensor([[[0,0], [1, 0], [1,1], [0,1.]]])
print(x.size())
scores = torch.FloatTensor([1.0, 0.8, 0.9])
cat = torch.FloatTensor([1, 1, 2])
threshold = torch.FloatTensor([0.5, 0.5, 0.2])

out = rotate_iou_cpp.bev_nms_cpu(x, scores, cat, threshold)
print(out)