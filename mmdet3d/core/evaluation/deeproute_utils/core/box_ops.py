import sys
sys.path.append('..')
import numpy as np
import numba
import os
import pcl

def get_corners(center, size, yaw):
    yaw_cos = np.cos(yaw)
    yaw_sin = np.sin(yaw)

    all_corners = []
    length = center.shape[0]
    for idx in range(length):
        rot = np.asmatrix([[yaw_cos[idx], -yaw_sin[idx]], [yaw_sin[idx], yaw_cos[idx]]])
        plain_pts = np.asmatrix([[0.5 * size[idx][0], 0.5 * size[idx][1]], [0.5 * size[idx][0], -0.5 * size[idx][1]], \
                                 [-0.5 * size[idx][0], -0.5 * size[idx][1]], [-0.5 * size[idx][0], 0.5 * size[idx][1]]])
        tran_pts = np.asarray(rot * plain_pts.transpose())
        tran_pts = tran_pts.transpose()

        tran_pts = tran_pts[[3, 2, 1, 0], :]

        corners = np.arange(8).astype(np.float32).reshape(4, 2)
        for i in range(4):
            corners[i][0] = center[idx][0] + tran_pts[i % 4][0]
            corners[i][1] = center[idx][1] + tran_pts[i % 4][1]
        all_corners.append(corners)

    if len(all_corners) != 0:
        all_corners = np.stack(all_corners, axis = 0)
    else:
        all_corners = np.array(all_corners).reshape(-1, 4, 2)

    return all_corners

def get_corners_v2(center, size, yaw):
    yaw_cos = np.cos(yaw)
    yaw_sin = np.sin(yaw)

    all_corners = []
    length = center.shape[0]
    for idx in range(length):
        rot = np.asmatrix([[yaw_cos[idx], -yaw_sin[idx]], [yaw_sin[idx], yaw_cos[idx]]])
        plain_pts = np.asmatrix([[0.5 * size[idx][0], 0.5 * size[idx][1]], [0.5 * size[idx][0], -0.5 * size[idx][1]], \
                                 [-0.5 * size[idx][0], -0.5 * size[idx][1]], [-0.5 * size[idx][0], 0.5 * size[idx][1]]])
        tran_pts = np.asarray(rot * plain_pts.transpose())
        tran_pts = tran_pts.transpose()
        tran_pts = tran_pts[[3, 2, 1, 0], :]
        corners = np.arange(24).astype(np.float32).reshape(8, 3)
        for i in range(8):
            corners[i][0] = center[idx][0] + tran_pts[i % 4][0]
            corners[i][1] = center[idx][1] + tran_pts[i % 4][1]
            corners[i][2] = center[idx][2] + (float(i >= 4) - 0.5) * size[idx][2]
        all_corners.append(corners)

    all_corners = np.stack(all_corners, axis = 0)
    return all_corners


def change_box3d_center_(box3d, src, dst):
    dst = np.array(dst, dtype=box3d.dtype)
    src = np.array(src, dtype=box3d.dtype)
    box3d[..., :3] += box3d[..., 3:6] * (dst - src)

@numba.njit
def surface_equ_3d_jitv2(surfaces):
    # polygon_surfaces: [num_polygon, num_surfaces, num_points_of_polygon, 3]
    num_polygon = surfaces.shape[0]
    max_num_surfaces = surfaces.shape[1]
    normal_vec = np.zeros((num_polygon, max_num_surfaces, 3), dtype=surfaces.dtype)
    d = np.zeros((num_polygon, max_num_surfaces), dtype=surfaces.dtype)
    sv0 = surfaces[0, 0, 0] - surfaces[0, 0, 1]
    sv1 = surfaces[0, 0, 0] - surfaces[0, 0, 1]
    for i in range(num_polygon):
        for j in range(max_num_surfaces):
            sv0[0] = surfaces[i, j, 0, 0] - surfaces[i, j, 1, 0]
            sv0[1] = surfaces[i, j, 0, 1] - surfaces[i, j, 1, 1]
            sv0[2] = surfaces[i, j, 0, 2] - surfaces[i, j, 1, 2]
            sv1[0] = surfaces[i, j, 1, 0] - surfaces[i, j, 2, 0]
            sv1[1] = surfaces[i, j, 1, 1] - surfaces[i, j, 2, 1]
            sv1[2] = surfaces[i, j, 1, 2] - surfaces[i, j, 2, 2]
            normal_vec[i, j, 0] = (sv0[1] * sv1[2] - sv0[2] * sv1[1])
            normal_vec[i, j, 1] = (sv0[2] * sv1[0] - sv0[0] * sv1[2])
            normal_vec[i, j, 2] = (sv0[0] * sv1[1] - sv0[1] * sv1[0])

            d[i, j] = -surfaces[i, j, 0, 0] * normal_vec[i, j, 0] - \
                      surfaces[i, j, 0, 1] * normal_vec[i, j, 1] - \
                      surfaces[i, j, 0, 2] * normal_vec[i, j, 2]
    return normal_vec, d


def points_in_convex_polygon_3d_jit(points,
                                    polygon_surfaces,
                                    num_surfaces=None):
    """check points is in 3d convex polygons.
    Args:
        points: [num_points, 3] array.
        polygon_surfaces: [num_polygon, max_num_surfaces,
            max_num_points_of_surface, 3]
            array. all surfaces' normal vector must direct to internal.
            max_num_points_of_surface must at least 3.
        num_surfaces: [num_polygon] array. indicate how many surfaces
            a polygon contain
    Returns:
        [num_points, num_polygon] bool array.
    """
    max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[1:3]
    num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]
    if num_surfaces is None:
        num_surfaces = np.full((num_polygons,), 9999999, dtype=np.int64)
    normal_vec, d = surface_equ_3d_jitv2(polygon_surfaces[:, :, :3, :])
    # normal_vec: [num_polygon, max_num_surfaces, 3]
    # d: [num_polygon, max_num_surfaces]
    return _points_in_convex_polygon_3d_jit(points, polygon_surfaces, normal_vec, d, num_surfaces)


@numba.njit
def _points_in_convex_polygon_3d_jit(points,
                                     polygon_surfaces,
                                     normal_vec, d,
                                     num_surfaces=None):
    """check points is in 3d convex polygons.
    Args:
        points: [num_points, 3] array.
        polygon_surfaces: [num_polygon, max_num_surfaces,
            max_num_points_of_surface, 3]
            array. all surfaces' normal vector must direct to internal.
            max_num_points_of_surface must at least 3.
        num_surfaces: [num_polygon] array. indicate how many surfaces
            a polygon contain
    Returns:
        [num_points, num_polygon] bool array.
    """
    max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[1:3]
    num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]
    ret = np.ones((num_points, num_polygons), dtype=np.bool_)
    sign = 0.0
    for i in range(num_points):
        for j in range(num_polygons):
            for k in range(max_num_surfaces):
                if k > num_surfaces[j]:
                    break
                sign = points[i, 0] * normal_vec[j, k, 0] \
                       + points[i, 1] * normal_vec[j, k, 1] \
                       + points[i, 2] * normal_vec[j, k, 2] + d[j, k]
                if sign >= 0:
                    ret[i, j] = False
                    break
    return ret

def get_points_per_box(pcd_path, timeStamp, surfaces):
    timeStamp = timeStamp.split(".")[0] + ".pcd"

    try:
        pc = pcl.load(os.path.join(pcd_path, timeStamp))
        points = np.array(pc)
    except:
        print("check if your pcl file is loaded via numpy.fromfile!")
        points = np.fromfile(os.path.join(pcd_path, timeStamp), dtype=np.float32)
        points = points.reshape([-1, 4])

    ret = points_in_convex_polygon_3d_jit(points[:,0:3], surfaces)
    all_points = []
    for i in range(surfaces.shape[0]):
        all_points.append(points[ret[:, i], :])

    return all_points