import sys
sys.path.append('..')
import numpy as np
from shapely.geometry.polygon import Polygon
from shapely import geometry
import math

def filter_d(box, distance_thresh):
    """
    :param box: array, shape = [4, 2]
    :param distance_thresh: list, the radius of circle, e.g., [20, 40, 80]
    :return:
        return 0: when box is within the circle with radius 20
        return 1: when box is within the circle with radius 40
        return 2: when box is within the circle with radius 80
        return 3: when box is within the circle with any radius
    """
    center = geometry.Point(0, 0)
    distance_thresh = sorted(distance_thresh)
    circle_radiuses = [center.buffer(radius) for radius in distance_thresh]

    p = Polygon(box)
    for i, circle_radius in enumerate(circle_radiuses):
        if circle_radius.intersects(p):
            return i
    return len(distance_thresh)

def filter_d_v2(loc, distance_thresh):
    """
    :param box: array, shape = [4, 2]
    :param distance_thresh: list, the radius of circle, e.g., [20, 40, 80]
    :return:
        return 0: when box is within the circle with radius 20
        return 1: when box is within the circle with radius 40
        return 2: when box is within the circle with radius 80
        return 3: when box is within the circle with any radius
    """
    distance_thresh = sorted(distance_thresh)
    distance = math.sqrt(loc[0]**2+loc[1]**2)
    for dist_idx, dist in enumerate(distance_thresh):
        if distance < dist:
            return dist_idx

    return len(distance_thresh)

def is_farther(boxA, boxB):
    x1 = np.mean([vertex[0] for vertex in boxA])
    y1 = np.mean([vertex[1] for vertex in boxA])
    x2 = np.mean([vertex[0] for vertex in boxB])
    y2 = np.mean([vertex[1] for vertex in boxB])
    d1 = x1 ** 2 + y1 ** 2
    d2 = x2 ** 2 + y2 ** 2
    if d1 > d2:
        # return true if corner1 is farther
        return True
    else:
        # return false if corner1 is closer
        return False

def occlusion_relation(boxA, boxB, angle1, intersect_angle, occlusion_thresh):
    """
    get the occlusion relationship between two boxes, i.e., box A and box B
    :param boxA: array, shape = (4, 2), the coordinates of four corners of box A
    :param boxB: refer to boxA
    :param angle1: float, the maximum angle of box A
    :param intersect_angle: the overlapping angle of box A and box B
    :param occlusion_thresh:
    :return:
        True, means boxA is not occluded by boxB
        False, means boxA is occluded by boxB
    """

    # no intersection
    if intersect_angle <= 0:
        return True

    else:
        # is corner1 farther than corner2, note that corner1 can be occluded by corner2 in this case
        if is_farther(boxA, boxB):
            if intersect_angle / angle1 > occlusion_thresh:
                return False
            else:
                return True
        else:
            return True

def angle(point):
    """
    :param point: array, shape = [1, 2]
    :return:
        the angle between line OA(i.e., O:(0,0), A:point) and axis x.
    """
    degree = np.degrees(np.arctan(abs(point[1]) / abs(point[0]))) if abs(point[0]) > 1e-9 else 90.0
    if point[0] >= 0 and point[1] >= 0:
        return degree
    elif point[0] >= 0 and point[1] < 0:
        return 360 - degree
    elif point[0] < 0 and point[1] >= 0:
        return 180 - degree
    else:
        return 180 + degree

def get_max_angle_points(box):
    """
    any two points A, B of a box have a angle with center point O:(0,0). Find A and B such that
    angle AOB is the maximum
    :param box: array, shape = [4, 2], the coordinates of four corners of a box
    :return:
        points: list, [point A, point B]
        max_absolute_angle: float, the angle of AOB
    """
    relative_angles = []
    for i in range(len(box)):
        relative_angles.append(angle(box[i]))

    max_absolute_angle = -1
    max_point1 = -1
    max_point2 = -1
    length = len(relative_angles)
    for i in range(length):
        for j in range(i + 1, length):
            absolute_angle = abs(relative_angles[i] - relative_angles[j])
            if absolute_angle > 180:
                absolute_angle = 360 - absolute_angle
            if absolute_angle > max_absolute_angle:
                max_absolute_angle = absolute_angle
                max_point1 = i
                max_point2 = j

    points = [box[max_point1], box[max_point2]]
    return points, max_absolute_angle

def get_occlusion(boxesA, boxesB, occlusion_thresh):
    """
    :param boxesA: array, shape = [M, 4, 2], where M is the number of boxes in the
                     current frame, [4, 2] stands for the coordinates of four corners for each box
    :param boxesB: refer to boxesA
    :param occlusion_thresh: float, the threshold of occlusion ratio, 1/3 is selected here
    :return: occlusion: array, shape = [M, 1], where where M is the number of boxes in the
                        current frame, 0 stands for occlusion
    """
    N = boxesA.shape[0]
    M = boxesB.shape[0]
    occlusion = np.ones(N, dtype=bool)
    for i in range(N):
        points1, angle1 = get_max_angle_points(boxesA[i])
        out = True
        for j in range(M):
            points2, angle2 = get_max_angle_points(boxesB[j])
            points, union_angle = get_max_angle_points(points1 + points2)
            intersect_angle = angle1 + angle2 - union_angle
            out = out and occlusion_relation(boxesA[i], boxesB[j], angle1, intersect_angle, occlusion_thresh)
            if not out:
                break
        occlusion[i] = out
    return occlusion.astype(int)


