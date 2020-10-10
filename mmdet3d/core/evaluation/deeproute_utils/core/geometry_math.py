import numpy as np

def points_distance(point1, point2):
    point1 = np.array(point1)
    point2 = np.array(point2)
    return np.sqrt(np.sum((point1 - point2)**2))

def compute_distance_point2Seg(seg_point1, seg_point2, point):
    seg_point1 = np.array(seg_point1)
    seg_point2 = np.array(seg_point2)
    point = np.array(point)

    distance = points_distance(seg_point1, seg_point2)

    if distance < 1e-6:
        return points_distance(seg_point1, point)
    else:
        cross_val = np.cross((seg_point2-seg_point1), (point-seg_point1))
        lateral = abs(cross_val/distance)
        sign = (seg_point2-seg_point1).dot((point-seg_point1)) * (seg_point1-seg_point2).dot((point-seg_point2))
        if sign < 0 and sign < -0.01:
            lateral = 1e9
        return lateral

def point2LineDistance(point, line):
    point1 = np.array(line[0])
    point2 = np.array(line[1])
    point = np.array(point)

    line_distance = points_distance(point1, point2)
    if line_distance < 1e-6:
        return points_distance(point1, point)

    cross_val = np.cross((point2 - point1), (point - point1))
    lateral = abs(cross_val / line_distance)
    return lateral

