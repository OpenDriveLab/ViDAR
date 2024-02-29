"""Helper functions for E2E future prediction."""

import torch
import numpy as np
import copy


def bev_grids_to_coordinates(ref_grids, pc_range):
    ref_coords = copy.deepcopy(ref_grids)
    ref_coords[..., 0:1] = (ref_coords[..., 0:1] *
                            (pc_range[3] - pc_range[0]) + pc_range[0])
    ref_coords[..., 1:2] = (ref_coords[..., 1:2] *
                            (pc_range[4] - pc_range[1]) + pc_range[1])
    return ref_coords

def bev_coords_to_grids(ref_coords, bev_h, bev_w, pc_range):
    ref_grids = copy.deepcopy(ref_coords)

    ref_grids[..., 0] = ((ref_grids[..., 0] - pc_range[0]) /
                         (pc_range[3] - pc_range[0]))
    ref_grids[..., 1] = ((ref_grids[..., 1] - pc_range[1]) /
                         (pc_range[4] - pc_range[1]))
    ref_grids = ref_grids * 2 - 1.  # [-1, 1]

    # Ignore the border part.
    border_x_min = 0.5 / bev_w * 2 - 1
    border_x_max = (bev_w - 0.5) / bev_w * 2 - 1
    border_y_min = 0.5 / bev_h * 2 - 1
    border_y_max = (bev_h - 0.5) / bev_h * 2 - 1
    valid_mask = ((ref_grids[..., 0:1] > border_x_min) &
                  (ref_grids[..., 0:1] < border_x_max) &
                  (ref_grids[..., 1:2] > border_y_min) &
                  (ref_grids[..., 1:2] < border_y_max))
    return ref_grids, valid_mask

def coords_to_voxel_grids(ref_coords, bev_h, bev_w, pillar_num, pc_range):
    ref_grids = copy.deepcopy(ref_coords)

    ref_grids[..., 0] = ((ref_grids[..., 0] - pc_range[0]) /
                         (pc_range[3] - pc_range[0])) * bev_w
    ref_grids[..., 1] = ((ref_grids[..., 1] - pc_range[1]) /
                         (pc_range[4] - pc_range[1])) * bev_h
    ref_grids[..., 2] = ((ref_grids[..., 2] - pc_range[2]) /
                         (pc_range[5] - pc_range[2])) * pillar_num
    return ref_grids


def get_bev_grids(H, W, bs=1, device='cuda', dtype=torch.float, offset=0.5):
    """Get the reference points used in SCA and TSA.
    Args:
        H, W: spatial shape of bev.
        device (obj:`device`): The device where
            reference_points should be.
    Returns:
        Tensor: reference points used in decoder, has \
            shape (bs, H * W, 2).
    """
    ref_y, ref_x = torch.meshgrid(
        torch.linspace(
            offset, H - (1 - offset), H, dtype=dtype, device=device),
        torch.linspace(
            offset, W - (1 - offset), W, dtype=dtype, device=device)
    )
    ref_y = ref_y.reshape(-1)[None] / H
    ref_x = ref_x.reshape(-1)[None] / W
    ref_bev = torch.stack((ref_x, ref_y), -1)
    ref_bev = ref_bev.repeat(bs, 1, 1)
    return ref_bev


def get_bev_grids_3d(H, W, Z, bs=1, device='cuda', dtype=torch.float):
    # reference points in 3D space, used in spatial cross-attention (SCA)
    zs = torch.linspace(0.5, Z - 0.5, Z, dtype=dtype,
                        device=device).view(-1, 1, 1).expand(Z, H, W) / Z
    xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                        device=device).view(1, 1, W).expand(Z, H, W) / W
    ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                        device=device).view(1, H, 1).expand(Z, H, W) / H
    ref_3d = torch.stack((xs, ys, zs), -1)
    # Z B H W ==> Z B HW ==> Z HW 3
    ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
    ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
    return ref_3d


# JIT
from torch.utils.cpp_extension import load
dvxlr = load("dvxlr", sources=[
    "third_lib/dvxlr/dvxlr.cpp",
    "third_lib/dvxlr/dvxlr.cu"], verbose=True)
class DifferentiableVoxelRenderingLayer(torch.autograd.Function):

    @staticmethod
    def forward(ctx, sigma, origin, points, tindex):
        pred_dist, gt_dist, dd_dsigma, indices = dvxlr.render(sigma,
                                                              origin,
                                                              points,
                                                              tindex)
        ctx.save_for_backward(dd_dsigma, indices, tindex, sigma)
        return pred_dist, gt_dist

    @staticmethod
    def backward(ctx, gradpred, gradgt):
        dd_dsigma, indices, tindex, sigma_shape = ctx.saved_tensors
        elementwise_mult = gradpred[..., None] * dd_dsigma

        invalid_grad = torch.isnan(elementwise_mult)
        elementwise_mult[invalid_grad] = 0.0

        grad_sigma = dvxlr.get_grad_sigma(elementwise_mult, indices, tindex, sigma_shape)[0]

        return grad_sigma, None, None, None


DifferentiableVoxelRendering = DifferentiableVoxelRenderingLayer.apply


# differentiable volume rendering v2.
dvxlr_v2 = load("dvxlr_v2", sources=[
    "third_lib/dvxlr/dvxlr_v2.cpp",
    "third_lib/dvxlr/dvxlr_v2.cu"], verbose=True)
class DifferentiableVoxelRenderingLayerV2(torch.autograd.Function):

    @staticmethod
    def forward(ctx, sigma, origin, points, tindex, sigma_regul):
        (pred_dist, gt_dist, dd_dsigma, indices,
         ray_pred, indicator) = dvxlr_v2.render_v2(
            sigma, origin, points, tindex, sigma_regul)
        ctx.save_for_backward(dd_dsigma, indices, tindex, sigma, indicator)
        return pred_dist, gt_dist, ray_pred, indicator

    @staticmethod
    def backward(ctx, gradpred, gradgt, grad_ray_pred, grad_indicator):
        dd_dsigma, indices, tindex, sigma_shape, indicator = ctx.saved_tensors
        elementwise_mult = gradpred[..., None] * dd_dsigma

        grad_sigma, grad_sigma_regul = dvxlr_v2.get_grad_sigma_v2(
            elementwise_mult, indices, tindex, sigma_shape, indicator, grad_ray_pred)

        return grad_sigma, None, None, None, grad_sigma_regul


DifferentiableVoxelRenderingV2 = DifferentiableVoxelRenderingLayerV2.apply


def get_inside_mask(points, point_cloud_range):
    """Get mask of points who are within the point cloud range.

    Args:
        points: A tensor with shape of [num_points, 3]
        pc_range: A list with content as
            [x_min, y_min, z_min, x_max, y_max, z_max]
    """
    mask = ((point_cloud_range[0] <= points[..., 0]) &
            (points[..., 0] <= point_cloud_range[3]) &
            (point_cloud_range[1] <= points[..., 1]) &
            (points[..., 1] <= point_cloud_range[4]) &
            (point_cloud_range[2] <= points[..., 2]) &
            (points[..., 2] <= point_cloud_range[5]))
    return mask


from chamferdist import ChamferDistance
chamfer_distance = ChamferDistance()
def compute_chamfer_distance(pred_pcd, gt_pcd):
    loss_src, loss_dst, _ = chamfer_distance(
        pred_pcd[None, ...], gt_pcd[None, ...], bidirectional=True, reduction='sum')

    chamfer_dist_value = (loss_src / pred_pcd.shape[0]) + (loss_dst / gt_pcd.shape[0])
    return chamfer_dist_value / 2.0


def compute_chamfer_distance_inner(pred_pcd, gt_pcd, pc_range):
    pred_mask = get_inside_mask(pred_pcd, pc_range)
    inner_pred_pcd = pred_pcd[pred_mask]

    gt_mask = get_inside_mask(gt_pcd, pc_range)
    inner_gt_pcd = gt_pcd[gt_mask]

    if pred_mask.sum() == 0 or gt_mask.sum() == 0:
        return 0.0

    return compute_chamfer_distance(inner_pred_pcd, inner_gt_pcd)


# visualization function for predicted point clouds.
# directly modified from nuscenes toolkit.
def _dbg_draw_pc_function(points, labels, color_map, output_path,
                          ctr=None, ctr_labels=None,):
    """Draw point cloud segmentation mask from BEV

    Args:
        points: A ndarray with shape as [-1, 3]
        labels: the label of each point with shape [-1]
        color_map: color of each label.
    """
    import matplotlib.pyplot as plt

    _, ax = plt.subplots(1, 1, figsize=(9, 9))
    axes_limit = 40
    # points: LiDAR points with shape [-1, 3]
    viz_points = points
    dists = np.sqrt(np.sum(viz_points[:, :2] ** 2, axis=1))
    colors = np.minimum(1, dists / axes_limit / np.sqrt(2))

    # prepare color_map
    points_color = color_map[labels] / 255.  # -1, 3

    point_scale = 0.2
    scatter = ax.scatter(viz_points[:, 0], viz_points[:, 1],
                         c=points_color, s=point_scale)

    if ctr is not None:
        # draw center of the point cloud (Ego position).
        ctr_scale = 100
        ctr_color = color_map[ctr_labels] / 255.
        ax.scatter(ctr[:, 0], ctr[:, 1], c=ctr_color, s=ctr_scale, marker='x')

    ax.plot(0, 0, 'x', color='red')
    ax.set_xlim(-axes_limit, axes_limit)
    ax.set_ylim(-axes_limit, axes_limit)
    ax.axis('off')
    ax.set_aspect('equal')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=200)


def _get_direction_of_each_query_points(points, origin=0.5):
    """
    Args:
        points: A tensor with shape as [..., 2/3] with a range of [0, 1]
        origin: The origin point position of start points.
    """
    r = points - origin
    r_norm = r / torch.sqrt((r ** 2).sum(-1, keepdims=True))
    return r_norm


