import torch
import rotate_iou_cpp
import time

# rotate_iou_cpp.rotated_iou

# one vs one
# note algo_v4 need nishizhen
x = torch.tensor([[[0,0], [1, 0], [1,1], [0,1.]]])
y = torch.tensor([[[0,0], [1, 0], [1,1], [0,1.]]])
result = rotate_iou_cpp.rotated_iou(x, y)
print("direction ", result)
print(result.size())
print ("-------------------------------------------")

#one vs one
x = torch.tensor([[[0,0], [0,0.], [0,0], [0,0]]])
y = torch.tensor([[[0,0], [0,0.], [0,0], [0,0]]])
result = rotate_iou_cpp.rotated_iou(x, y)
print(result)
print(result.size())
print ("-------------------------------------------")

#one vs one
x = torch.tensor([[[0,0], [0,1.], [1,1], [1,0]]])
y = torch.tensor([[[0,0], [-1,0.], [0,-1], [-1,-1]]])
result = rotate_iou_cpp.rotated_iou(x, y)
print(result)
print(result.size())
print ("-------------------------------------------")

#one vs one
x = torch.tensor([[[1,0], [1,0.], [1,1], [0,1]]])
y = torch.tensor([[[0,0], [1,0],[1,1.],  [0,1.]]])
result = rotate_iou_cpp.rotated_iou(x, y)
print("stop")
print(result)
print(result.size())
print ("-------------------------------------------")

#one vs one
x = torch.tensor([[[0,0], [1,0.], [1,1], [0,1]]])
y = torch.tensor([[[0.5,0], [1,0], [1,1.], [0,1]]])
result = rotate_iou_cpp.rotated_iou(x, y)
print(result)
print(result.size())
print ("-------------------------------------------")

#one vs one
x = torch.tensor([[[0,0], [1,0.], [1,1], [0,1]]])
y = torch.tensor([[[0.,0.], [0.5,0], [0.5,0.5], [0,0.5]]])
result = rotate_iou_cpp.rotated_iou(x, y)
print("here ", result)
print(result.size())
print ("-------------------------------------------")

#one vs one
x = torch.tensor([[[0,0], [1,0.], [1,1], [0,1]]])
y = torch.tensor([[[0.,0.], [2,0], [2,2], [0,2]]])
result = rotate_iou_cpp.rotated_iou(x, y)
print(result)
print(result.size())
print ("-------------------------------------------")

#one vs one
x = torch.tensor([[[0,0], [1,0.], [1,1], [0,1]]])
y = torch.tensor([[[0.,0.5], [0,-0.5], [1,-0.5], [1,0.5]]])
result = rotate_iou_cpp.rotated_iou(x, y)
print(result)
print(result.size())
print ("-------------------------------------------")


# two vs one
x = torch.tensor([[[0,0], [0,1.], [1,1], [1,0]],
                [[0, 0], [0, 0.], [0, 0], [0, 0]]])
# print (x.size())
y = torch.tensor([[[0,0], [0,1.], [1,1], [1,0]]])
result = rotate_iou_cpp.rotated_iou(x, y)
print(result)
print(result.size())
print ("-------------------------------------------")

# two vs two
x = torch.tensor([[[0,0], [1,0], [1,1] , [0,1.]],
                [[0, 0], [0, 0.], [0, 0], [0, 0]]])
# print (x.size())
y = torch.tensor([[[0,0], [1,0], [1,1] , [0,1.]],
                [[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5]]])
print(x.size())
result = rotate_iou_cpp.rotated_iou(x, y)
print(result)
print(result.size())
print ("-------------------------------------------")


# 500 vs 500
x = torch.tensor([[[0,0], [1,0], [1,1] , [0,1.]],
                [[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5]]])
x = x.repeat(25, 1, 1)
# print("x size: ", x.size())
y = torch.tensor([[[0,0], [1,0], [1,1] , [0,1.]],
                [[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5]]])
y = y.repeat(25, 1, 1)
print("y size: ", y.size())
for i in range(10):
    start = time.time()
    result = rotate_iou_cpp.rotated_iou(x, y)
    end = time.time()
    print("duration time: ", end - start)
    # print(result)
    # print(result.size())
# print ("-------------------------------------------")

#one vs one
x = torch.tensor([[[69.4,-4.45], [73.9,-4.45], [73.9,-2.65], [69.4,-2.65]]])
y = torch.tensor([[[69.4,-4.47], [73.9,-4.47], [73.9,-3.65], [69.4,-3.65]]])
x = torch.FloatTensor(
[[[42.18,      -2.58     ],
  [46.480003,  -2.58     ],
  [46.480003 , -4.38     ],
  [42.18 ,     -4.38     ]]]
)
x = x[:, [0, 3, 2, 1], :]
print(x)
y = torch.FloatTensor(
[[[42.18,      -2.58     ],
  [46.480003,  -2.58     ],
  [46.480003 , -4.38     ],
  [42.18 ,     -4.38     ]]]
)
y = y[:, [0, 3, 2, 1], :]
result = rotate_iou_cpp.rotated_iou(x, y)
print(result)
print(result.size())
print ("-------------------------------------------")

x = torch.FloatTensor(
[[[ 49.210003 ,  -2.38     ],
  [ 44.91  ,     -2.38     ],
  [ 44.91  ,     -4.18     ],
  [ 49.210003,   -4.18     ]],

 [[-37.798073,   -2.467461 ],
  [-47.79655,    -2.2929351],
  [-47.841927,   -4.892539 ],
  [-37.84345  ,  -5.0670652]],

 [[-24.99   ,    15.03     ],
  [-33.19 ,      15.03     ],
  [-33.19  ,     12.53     ],
  [-24.99  ,     12.53     ]],

[[ 26.029978, -38.92495 ],
 [ 24.729822, -40.310444],
 [ 28.23002,  -43.595047],
 [ 29.530176, -42.209553]],
 #
 # [[ 27.272432 ,  57.66792  ],
 #  [ 24.672434,   57.6706   ],
 #  [ 24.65994 ,   45.570606 ],
 #  [ 27.25994 ,   45.567924 ]],
 #
 # [[ 13.73 ,     -19.470001 ],
 #  [ 12.73 ,     -19.470001 ],
 #  [ 12.73 ,     -20.27     ],
 #  [ 13.73,      -20.27     ]]
 ]
)

y = torch.FloatTensor(
[[[ 49.58888 ,   -2.361676 ],
  [ 45.115795 ,  -2.165553 ],
  [ 45.034527,   -4.0190725],
  [ 49.50761,    -4.2151957]],

 [[-37.818867,   -2.5541012],
  [-47.51973 ,   -2.2071023],
  [-47.621914,   -5.0638895],
  [-37.92105,    -5.4108887]],

 [[-25.647778,   14.978349 ],
  [-33.65941 ,   15.052167 ],
  [-33.683388 ,  12.4495325],
  [-25.671757,   12.375714 ]],

[[ 26.029976, -38.924942],
 [ 24.72982,  -40.310436],
 [ 28.230019, -43.59504 ],
 [ 29.530174, -42.209545]],


 # [[ 13.732842,  -19.372934 ],
 #  [ 12.768864 , -19.378595 ],
 #  [ 12.774484 , -20.335669 ],
 #  [ 13.738462,  -20.330008 ]],
 #
 # [[-59.58292  ,  44.749214 ],
 #  [-61.69647 ,   45.23392  ],
 #  [-62.956043,   39.741592 ],
 #  [-60.842495 ,  39.256886 ]]
 ]
)

result = rotate_iou_cpp.rotated_iou(x, y)
print(result)
print(result.size())
print ("-------------------------------------------")