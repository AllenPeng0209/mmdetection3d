# Copyright 2019 deeproute.ai. All Rights Reserved.
# Author: Xiaoyi Zou (xiaoyizou@deeproute.ai)
import math
import numpy as np
import cv2


def discretize(vec, lim, size):
    return ((vec - lim[0]) / (lim[1] - lim[0]) * size).astype(int)


def generateTopview(pc, sizes, lims):
    xsize, ysize, zsize = sizes
    xlim, ylim, zlim = lims
    hmap = np.zeros((ysize, xsize, zsize), dtype=np.uint8)
    npoints = len(pc)

    mask = np.ones(npoints, dtype=bool)
    mask = np.logical_and(mask, pc[:, 0] > xlim[0])
    mask = np.logical_and(mask, pc[:, 0] < xlim[1])
    mask = np.logical_and(mask, pc[:, 1] > ylim[0])
    mask = np.logical_and(mask, pc[:, 1] < ylim[1])
    mask = np.logical_and(mask, pc[:, 2] > zlim[0])
    mask = np.logical_and(mask, pc[:, 2] < zlim[1])
    b_reduced = pc[mask, :]

    xidx = discretize(b_reduced[:, 0], xlim, xsize) - 1
    yidx = discretize(b_reduced[:, 1], ylim, ysize) - 1
    zidx = discretize(b_reduced[:, 2], zlim, zsize) - 1
    hmap[yidx, xidx, zidx] = 180
    return hmap


def drawCircles(image, centers, color, radiu=2):
    ori = centers[0]
    for i, center in enumerate(centers):
        if abs(center[0] - ori[0]) > 1000 or abs(center[1] - ori[1]) > 1000:
            # print("Warn: {},{} is predict wrong".format(center[0], center[1]))
            break
        color = color[i] if type(color) == list() else color
        cv2.circle(image, tuple(center), radiu, color, -1)
        # print(image.flags["C_CONTIGUOUS"]) Note: when this is False, cv2 draw
        # function is unvalid !!!  Must use np.ascontiguousarray() to set True
    return image


def cornerToXywlYaw(cor):
    xy = np.sum(cor, axis=0) / 4
    w = (np.linalg.norm((cor[0] - cor[1])) + np.linalg.norm((cor[2] - cor[3]))) / 2
    l = (np.linalg.norm((cor[1] - cor[2])) + np.linalg.norm((cor[3] - cor[0]))) / 2

    long_edge = (cor[0] - cor[1] + cor[3] - cor[2]) / 2
    short_edge = (cor[0] + cor[1] - cor[3] - cor[2]) / 2
    yaw = np.arctan2(long_edge[0], long_edge[1])
    return xy, l, w, yaw


def xywlYawToCorner(xywlYaw):
    x, y, l, w, yaw = xywlYaw
    center = [x, y]
    size = [l, w]
    rot = np.asmatrix([[math.cos(yaw), -math.sin(yaw)], \
                       [math.sin(yaw), math.cos(yaw)]])
    plain_pts = np.asmatrix([[0.5 * size[0], 0.5 * size[1]], \
                             [0.5 * size[0], -0.5 * size[1]], \
                             [-0.5 * size[0], -0.5 * size[1]], \
                             [-0.5 * size[0], 0.5 * size[1]]])
    tran_pts = np.asarray(rot * plain_pts.transpose())
    return tran_pts.transpose() + np.array([x, y], dtype=np.float32)


def fillObjects(image, objs, colors, sizes, lims, transformer=None):
    xsize, ysize, zsize = sizes
    xlim, ylim, zlim = lims
    for idx, obj in enumerate(objs):
        x, y, z = obj.position.x, obj.position.y, obj.position.z
        l, w = obj.bounding_box.x, obj.bounding_box.y
        cor = xywlYawToCorner([x, y, l, w, obj.heading])
        if transformer is not None:
            cor = np.concatenate((cor, z * np.ones((cor.shape[0], 1))), axis=1)
            cor = transformer(cor)
            cor = cor[:, 0:2]
        vrx = (cor - np.array([xlim[0], ylim[0]])) / (xlim[1] - xlim[0]) * xsize
        vrx = vrx.astype(int)
        cv2.fillConvexPoly(image, vrx, colors[idx])
    return image


def fillObjectsStack(image, objs, colors, sizes, lims, transformer=None):
    xsize, ysize, zsize = sizes
    xlim, ylim, zlim = lims
    img = np.zeros_like(image)
    for idx, obj in enumerate(objs):
        x, y, z = obj.position.x, obj.position.y, obj.position.z
        l, w = obj.bounding_box.x, obj.bounding_box.y
        cor = xywlYawToCorner([x, y, l, w, obj.heading])
        if transformer is not None:
            cor = np.concatenate((cor, z * np.ones((cor.shape[0], 1))), axis=1)
            cor = transformer(cor)
            cor = cor[:, 0:2]
        vrx = (cor - np.array([xlim[0], ylim[0]])) / (xlim[1] - xlim[0]) * xsize
        vrx = vrx.astype(int)
        cv2.fillConvexPoly(img, vrx, colors[idx])
    image = cv2.addWeighted(image, 1, img, 1, 0)
    return image


def fillObjectsEach(images, objs, colors, sizes, lims, transformer=None, history_size=20):
    xsize, ysize, zsize = sizes
    xlim, ylim, zlim = lims
    for idx, obj in enumerate(objs):
        x, y, z = obj.position.x, obj.position.y, obj.position.z
        l, w = obj.bounding_box.x, obj.bounding_box.y
        cor = xywlYawToCorner([x, y, l, w, obj.heading])
        if transformer is not None:
            cor = np.concatenate((cor, z * np.ones((cor.shape[0], 1))), axis=1)
            cor = transformer(cor)
            cor = cor[:, 0:2]
        vrx = (cor - np.array([xlim[0], ylim[0]])) / (xlim[1] - xlim[0]) * xsize
        vrx = vrx.astype(int)
        cv2.fillConvexPoly(images[history_size - idx - 1, :, :], vrx, 255)
    return images


def drawPredictObjects(image, objects, class_colors, sizes, lims, transformer=None):
    xsize, ysize, zsize = sizes
    xlim, ylim, zlim = lims
    if isinstance(objects, list):
        for obj in objects:
            x, y, z, l, w, h, heading, cls = obj
            cor = xywlYawToCorner([x, y, l, w, heading])
            if transformer is not None:
                cor = transformer(cor)
                cor = cor[:, 0:2]
            center = np.mean(cor - np.array([0.6, -0.6]), axis=0)
            center = (center - np.array([xlim[0], ylim[0]])) / (xlim[1] - xlim[0]) * xsize
            vrx = (cor - np.array([xlim[0], ylim[0]])) / (xlim[1] - xlim[0]) * xsize
            vrx = vrx.astype(int)
            vrx = vrx.reshape((-1, 1, 2))
            cv2.polylines(image, [vrx], True, class_colors[obj.type], 2)
            cv2.putText(image, str(obj.id), tuple(center.astype(int)), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 2)
    else:
        for obj in objects.objects:
            x, y, z = obj.position.x, obj.position.y, obj.position.z
            l, w = obj.bounding_box.x, obj.bounding_box.y
            cor = xywlYawToCorner([x, y, l, w, obj.heading])
            if transformer is not None:
                cor = np.concatenate((cor, z * np.ones((cor.shape[0], 1))), axis=1)
                cor = transformer(cor)
                cor = cor[:, 0:2]
            center = np.mean(cor - np.array([0.6, -0.6]), axis=0)
            center = (center - np.array([xlim[0], ylim[0]])) / (xlim[1] - xlim[0]) * xsize
            vrx = (cor - np.array([xlim[0], ylim[0]])) / (xlim[1] - xlim[0]) * xsize
            vrx = vrx.astype(int)
            vrx = vrx.reshape((-1, 1, 2))
            cv2.polylines(image, [vrx], True, class_colors[obj.type], 2)
            cv2.putText(image, str(obj.id), tuple(center.astype(int)), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 2)
    return image


def drawPredictTrajectories(image, objects, class_colors, sizes, lims, transformer=None):
    xsize, ysize, zsize = sizes
    xlim, ylim, zlim = lims
    for obj in objects.objects:
        x, y, z = obj.position.x, obj.position.y, obj.position.z
        for pred in obj.prediction:
            trajectory = []
            for traj in pred.trajectory:
                x, y = traj.position.x, traj.position.y
                trajectory.append([x, y])
            if len(trajectory) <= 1:
                continue
            trajectory = np.array(trajectory)
            if transformer is not None:
                trajectory = np.concatenate((trajectory, z * np.ones((trajectory.shape[0], 1))), axis=1)
                trajectory = transformer(trajectory)
                trajectory = trajectory[:, 0:2]
            vrx = (trajectory - np.array([xlim[0], ylim[0]])) / (xlim[1] - xlim[0]) * xsize
            vrx = vrx.astype(int)
            # print(pred.probability)
            color = (0, 0, 255)  # if pred.HasField("probability") else class_colors[obj.type]
            # color = class_colors[obj.type] if pred.probability <= 0.9 else (0, 0, 255)
            image = drawCircles(image, vrx, color)
            # vrx = vrx.reshape((-1, 1, 2))
            # cv2.polylines(image, [vrx], True, class_colors[obj.type], 2)
    return image


def drawTrackTraces(image, tracks, class_colors, sizes, lims):
    xsize, ysize, zsize = sizes
    xlim, ylim, zlim = lims
    for track in tracks:
        for j in range(len(track.trace) - 1):
            loc = (track.trace[j].loc - np.array([xlim[0], ylim[0]])) / (xlim[1] - xlim[0]) * xsize
            x1, y1 = np.sum(loc, axis=0) / 4
            loc = (track.trace[j + 1].loc - np.array([xlim[0], ylim[0]])) / (xlim[1] - xlim[0]) * xsize
            x2, y2 = np.sum(loc, axis=0) / 4
            cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), track.color, 2)
            # class_colors[track.obstacle.class_idx-1], 2)
    return image