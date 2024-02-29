"""Evaluation tools from 4d_occupancy.

https://github.com/tarashakhurana/4d-occ-forecasting
"""

import torch
import numpy as np
from chamferdist import ChamferDistance


PC_RANGE = [-70.0, -70.0, -4.5, 70.0, 70.0, 4.5]
MAX_VALUE = 1e8
chamfer_distance = ChamferDistance()


"""Ray Errors """
def point_at_infinity():
    return np.array([np.inf]*3)


def inside_volume(xyz):
    x, y, z = xyz.T
    xmin, ymin, zmin, xmax, ymax, zmax = PC_RANGE
    xmin, ymin, zmin, xmax, ymax, zmax = xmin - 0.02, ymin - 0.02, zmin - 0.02, xmax + 0.02, ymax + 0.02, zmax + 0.02
    return np.logical_and(
            xmin <= x,
            np.logical_and(
                x <= xmax,
                np.logical_and(
                    ymin <= y,
                    np.logical_and(
                        y <= ymax,
                        np.logical_and(
                            zmin <= z,
                            z <= zmax)))))



def clamp(pcd_, org_, return_invalid_mask=False):
    if torch.is_tensor(pcd_):
        pcd = pcd_.cpu().numpy()
    else:
        pcd = pcd_.copy()
    if torch.is_tensor(org_):
        org = org_.cpu().numpy()
    else:
        org = org_.copy()
    # print("unique values in gt before", np.unique(pcd))
    mask1 = np.logical_and(PC_RANGE[0] <= pcd[:, 0], pcd[:, 0] <= PC_RANGE[3])
    mask2 = np.logical_and(PC_RANGE[1] <= pcd[:, 1], pcd[:, 1] <= PC_RANGE[4])
    mask3 = np.logical_and(PC_RANGE[2] <= pcd[:, 2], pcd[:, 2] <= PC_RANGE[5])
    inner_mask = mask1 & mask2 & mask3
    origin = org.reshape((1, 3))
    origins = np.zeros_like(pcd) + origin
    if np.logical_not(inner_mask).sum() > 0:
        pcd_outer = pcd[np.logical_not(inner_mask)]
        origin_outer, clamped_pcd_outer = _clamp(pcd_outer, origin)
        pcd[np.logical_not(inner_mask)] = clamped_pcd_outer.astype(float)
        origins[np.logical_not(inner_mask)] = origin_outer

    invalid1 = np.logical_and(
            np.isinf(pcd[:, 0]),
            np.logical_and(
                np.isinf(pcd[:, 1]),
                np.isinf(pcd[:, 2])))
    invalid2 = np.logical_and(
            np.isnan(pcd[:, 0]),
            np.logical_and(
                np.isnan(pcd[:, 1]),
                np.isnan(pcd[:, 2])))
    invalid = np.logical_or(invalid1, invalid2)

    if not return_invalid_mask:
        origins = origins[np.logical_not(invalid)]
        pcd = pcd[np.logical_not(invalid)]
        return origins, pcd
    if return_invalid_mask:
        return origins, pcd, invalid


def _clamp(points, origin):
    xmin, ymin, zmin, xmax, ymax, zmax = PC_RANGE
    # points = torch.from_numpy(np.array([[5.99, 2.70, -4.92]]))
    new_origin = np.zeros_like(points) + origin
    # print("ray clamping", origin.tolist(), points.tolist(), end="\t")
    # ray starting point
    xo, yo, zo = origin.T
    # ray end point
    xe, ye, ze = points.T
    # degenerate ray
    mask = np.logical_and(
            (xe - xo) == 0,
            np.logical_and(
                (ye - yo) == 0,
                (ze - zo) == 0))
    if mask.sum() > 0:
        raise RuntimeError("Or`igin and the end point should not be identical at", points[mask])
        # if xo == xe and yo == ye and zo == ze:
        # return (point_at_infinity(), point_at_infinity())
    # ray raw length
    l = np.sqrt((xe-xo)**2 + (ye-yo)**2 + (ze-zo)**2)
    # non-zero
    # offset along x, y, z per unit movement
    dx = (xe - xo) / l
    dy = (ye - yo) / l
    dz = (ze - zo) / l
    # unit direction vector
    d = np.stack([dx, dy, dz])  # shape: 3 x N

    # check if the origin is inside the volume
    # print("bskbvkdr", points[37], new_origin[37])
    ray_intersects_volume = np.ones(d.shape[1]).astype(bool)
    if not inside_volume(origin):
        # print("inside the not inside volume", points[0])
        ray_intersects_volume = np.logical_not(ray_intersects_volume)
        # distance to planes along the ray direction
        origin_to_xmin = np.where(np.isclose(dx, 0.0), MAX_VALUE, (xmin - xo) / dx)
        origin_to_xmax = np.where(np.isclose(dx, 0.0), MAX_VALUE, (xmax - xo) / dx)
        origin_to_ymin = np.where(np.isclose(dy, 0.0), MAX_VALUE, (ymin - yo) / dy)
        origin_to_ymax = np.where(np.isclose(dy, 0.0), MAX_VALUE, (ymax - yo) / dy)
        origin_to_zmin = np.where(np.isclose(dz, 0.0), MAX_VALUE, (zmin - zo) / dz)
        origin_to_zmax = np.where(np.isclose(dz, 0.0), MAX_VALUE, (zmax - zo) / dz)
        # sort distance
        origin_to_planes = np.stack([origin_to_xmin, origin_to_xmax,
                                     origin_to_ymin, origin_to_ymax,
                                     origin_to_zmin, origin_to_zmax])  # shape: 6 x N
        lambda_order = np.argsort(origin_to_planes, axis=0)
        # find each plane in order
        for j in range(lambda_order.shape[1]):
            for i in range(lambda_order.shape[0]):
                # print("hfsgjkahf", points[37], new_origin[37])
                plane = lambda_order[i][j]
                if origin_to_planes[plane][j] + 1e-4 >= 0.0:
                    intersection = origin + origin_to_planes[plane][j] * d[:, j]
                    if inside_volume(intersection):
                        ray_intersects_volume[j] = True
                        new_origin[j] = intersection.copy()
                        if origin_to_planes[plane][j] > l[j]:
                            points[j] = new_origin[j].copy()
                        break


    # distance to planes along the reversed ray direction
    point_to_xmin = np.where(np.isclose(dx, 0.0), MAX_VALUE, (xmin - xe) / (-dx))
    point_to_xmax = np.where(np.isclose(dx, 0.0), MAX_VALUE, (xmax - xe) / (-dx))
    point_to_ymin = np.where(np.isclose(dy, 0.0), MAX_VALUE, (ymin - ye) / (-dy))
    point_to_ymax = np.where(np.isclose(dy, 0.0), MAX_VALUE, (ymax - ye) / (-dy))
    point_to_zmin = np.where(np.isclose(dz, 0.0), MAX_VALUE, (zmin - ze) / (-dz))
    point_to_zmax = np.where(np.isclose(dz, 0.0), MAX_VALUE, (zmax - ze) / (-dz))
    # sort distance
    point_to_planes = np.stack([point_to_xmin, point_to_xmax,
        point_to_ymin, point_to_ymax,
        point_to_zmin, point_to_zmax])
    lambda_order = np.argsort(point_to_planes, axis=0)

    for j, point in enumerate(points):
        if not ray_intersects_volume[j]:
            new_origin[j] = point_at_infinity()
            points[j] = point_at_infinity()
        else:
            if not inside_volume(point):
                # find each plane in order
                touches_volume = False
                for i in range(lambda_order.shape[0]):
                    plane = lambda_order[i][j]
                    if point_to_planes[plane][j] + 1e-4 >= 0.0:
                        intersection = points[j] + point_to_planes[plane][j] * (-d[:, j])
                        if inside_volume(intersection):
                            touches_volume = True
                            points[j] = intersection.copy()
                            break
                assert(touches_volume)

    return (new_origin, points)


def spherical_projection(pcd):
    pcd = pcd.T
    d = np.sqrt(pcd[0] * pcd[0] + pcd[1] * pcd[1] + pcd[2] * pcd[2])
    azimuth = np.arctan2(pcd[0], pcd[1])
    elevation = np.arctan2(pcd[2], pcd[1])
    return azimuth, elevation, d


def compute_ray_errors(pred_pcd, gt_pcd, origin, device, return_interpolated_pcd=False, savename=""):
    pred_pcd_norm = pred_pcd - origin[None, :]
    gt_pcd_norm = gt_pcd - origin[None, :]
    theta_hat, phi_hat, d_hat = spherical_projection(pred_pcd_norm)
    theta, phi, d = spherical_projection(gt_pcd_norm)

    mask_hat = d_hat > 1e-2
    mask = d > 1e-2
    theta_hat, phi_hat, d_hat, pred_pcd = theta_hat[mask_hat], phi_hat[mask_hat], d_hat[mask_hat], pred_pcd[mask_hat]
    theta, phi, d, gt_pcd = theta[mask], phi[mask], d[mask], gt_pcd[mask]

    count = theta.shape[0]
    pred_spherical = np.stack([theta_hat, phi_hat, np.ones_like(theta_hat)], axis=1)
    gt_spherical = np.stack([theta, phi, np.ones_like(theta)], axis=1)
    _, info = chamfer_distance(
        torch.from_numpy(pred_spherical[None, ...]).to(device),
        torch.from_numpy(gt_spherical[None, ...]).to(device),
        reverse=True,
        reduction='mean')
    _, pred_idx = info
    pred_idx = pred_idx.cpu().numpy()
    v = gt_pcd - origin[None, :]
    unit_dir = v / np.sqrt((v ** 2).sum(axis=1, keepdims=True))
    pred_pcd_interp = origin[None, :] + d_hat[pred_idx].T * unit_dir
    if return_interpolated_pcd:
        return pred_pcd_interp
    clamped_gt_origin, clamped_gt_pcd, invalid_mask = clamp(gt_pcd, origin, return_invalid_mask=True)
    _, clamped_pred_pcd_interp, _ = clamp(pred_pcd_interp, origin, return_invalid_mask=True)
    clamped_gt_pcd = clamped_gt_pcd[np.logical_not(invalid_mask)]
    clamped_pred_pcd_interp = clamped_pred_pcd_interp[np.logical_not(invalid_mask)]
    clamped_gt_origin = clamped_gt_origin[np.logical_not(invalid_mask)]
    d_clamped = np.sqrt(((clamped_gt_pcd - clamped_gt_origin) ** 2).sum(axis=1))
    valid = d_clamped > 0.01
    d_clamped = d_clamped[valid]
    clamped_gt_pcd = clamped_gt_pcd[valid]
    clamped_pred_pcd_interp = clamped_pred_pcd_interp[valid]
    eucl_dist = np.sqrt(((clamped_gt_pcd - clamped_pred_pcd_interp) ** 2).sum(axis=1))
    l1_error = eucl_dist
    absrel_error = eucl_dist / d_clamped

    return l1_error.sum() / count, absrel_error.sum() / count