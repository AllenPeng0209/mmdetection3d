import torch
import bev_nms_cuda
import time

x = torch.cuda.FloatTensor([[[0,0], [1, 0], [1,1], [0,1.]],
                            [[0,0], [1, 0], [1,1], [0,1.]],
                            [[0,0], [0.5, 0], [0.5,0.5], [0,0.5]]
                            ])
# y = torch.cuda.FloatTensor([[[0,0], [1, 0], [1,1], [0,1.]]])
print(x.size())
scores = torch.cuda.FloatTensor([1.0, 0.8, 0.9])
cat = torch.cuda.FloatTensor([1, 1, 2])
threshold = torch.cuda.FloatTensor([0.5, 0.5, 0.2])

out = bev_nms_cuda.bev_nms(x, scores, cat, threshold)
print(out)

x = torch.cuda.FloatTensor([[[25.83913,   -3.8767796],
 [25.913794 , -4.428568 ],
 [26.611963 , -4.34338  ],
 [26.528769 , -3.7333717]],

[[23.943708 , -2.6684532],
 [23.630007 , -4.2372646],
 [27.581385 , -4.8651733],
 [27.808792 , -3.2285156]],
[[23.943708 , -2.6684532],
 [23.630007 , -4.2372646],
 [27.581385 , -4.8651733],
 [27.808792 , -3.2285156]]
                            ])
# y = torch.cuda.FloatTensor([[[0,0], [1, 0], [1,1], [0,1.]]])
print(x.size())
scores = torch.cuda.FloatTensor([0.18781491, 0.10781491, 0.09])
cat = torch.cuda.FloatTensor([3, 1, 1])
threshold = torch.cuda.FloatTensor([0.02, 0.02, 0.4, 0.4])

out = bev_nms_cuda.bev_nms(x, scores, cat, threshold)
print(out)
print(x[out])