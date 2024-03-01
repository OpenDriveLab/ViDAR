#---------------------------------------------------------------------------------#
# Visual Point Cloud Forecasting enables Scalable Autonomous Driving              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

"""
<V1.multiframe> of ViDAR future prediction head:
    * Predict future & history frames simultaneously.
"""

import copy
import torch
import torch.nn as nn
import numpy as np

from mmdet.models import HEADS, build_loss

from mmcv.runner import force_fp32, auto_fp16
from .vidar_head_base import ViDARHeadBase


@HEADS.register_module()
class ViDARHeadV1(ViDARHeadBase):
    def __init__(self,
                 history_queue_length,
                 pred_history_frame_num=0,
                 pred_future_frame_num=0,
                 per_frame_loss_weight=(1.0,),

                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.history_queue_length = history_queue_length

        self.pred_history_frame_num = pred_history_frame_num
        self.pred_future_frame_num = pred_future_frame_num

        self.pred_frame_num = 1 + self.pred_history_frame_num + self.pred_future_frame_num
        self.per_frame_loss_weight = per_frame_loss_weight
        assert len(self.per_frame_loss_weight) == self.pred_frame_num

        self._init_bev_pred_layers()

    def _init_bev_pred_layers(self):
        """Overwrite the {self.bev_pred_head} of super()._init_layers()
        """
        bev_pred_branch = []
        for _ in range(self.num_pred_fcs):
            bev_pred_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            bev_pred_branch.append(nn.LayerNorm(self.embed_dims))
            bev_pred_branch.append(nn.ReLU(inplace=True))
        bev_pred_branch.append(nn.Linear(
            self.embed_dims, self.pred_frame_num * self.num_pred_height))
        bev_pred_head = nn.Sequential(*bev_pred_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # Auxiliary supervision for all intermediate results.
        num_pred = self.transformer.decoder.num_layers
        self.bev_pred_head = _get_clones(bev_pred_head, num_pred)

    def forward_head(self, next_bev_feats):
        """Get freespace estimation from multi-frame BEV feature maps.

        Args:
            next_bev_feats (torch.Tensor): with shape as
                [pred_frame_num, inter_num, bs, bev_h * bev_w, dims]
                pred_frame_num: history frames + current frame + future frames.
        """
        next_bev_preds = []
        for lvl in range(next_bev_feats.shape[1]):
            # pred_frame_num, bs, bev_h * bev_w, num_height_pred * num_frame
            #  ===> pred_frame_num, bs, bev_h * bev_w, num_height_pred, num_frame
            #  ===> pred_frame_num, num_frame, bs, bev_h * bev_w, num_height_pred.
            next_bev_pred = self.bev_pred_head[lvl](next_bev_feats[:, lvl])
            next_bev_pred = next_bev_pred.view(
                *next_bev_pred.shape[:-1], self.num_pred_height, self.pred_frame_num)

            base_bev_pred = next_bev_pred[..., self.pred_history_frame_num][..., None]
            next_bev_pred = torch.cat([
                next_bev_pred[..., :self.pred_history_frame_num] + base_bev_pred,
                base_bev_pred,
                next_bev_pred[..., self.pred_history_frame_num + 1:] + base_bev_pred
            ], -1)

            next_bev_pred = next_bev_pred.permute(0, 4, 1, 2, 3).contiguous()
            next_bev_preds.append(next_bev_pred)
        # pred_frame_num, inter_num, num_frame, bs, bev_h*bev_w, num_height_pred
        next_bev_preds = torch.stack(next_bev_preds, 1)
        return next_bev_preds

    def _get_reference_gt_points(self,
                                 gt_points,
                                 src_frame_idx_list,
                                 tgt_frame_idx_list,
                                 img_metas):
        """Transform gt_points at src_frame_idx in {src_frame_idx_list} to the coordinate space
        of each tgt_frame_idx in {tgt_frame_idx_list}.
        """
        bs = len(gt_points)
        aligned_gt_points = []
        batched_origin_points = []
        for frame_idx, src_frame_idx, tgt_frame_idx in zip(
                range(len(src_frame_idx_list)), src_frame_idx_list, tgt_frame_idx_list):
            # 1. get gt_points belongs to src_frame_idx.
            src_frame_gt_points = [p[p[:, -1] == src_frame_idx] for p in gt_points]

            # 2. get transformation matrix..
            src_to_ref = [img_meta['total_cur2ref_lidar_transform'][src_frame_idx] for img_meta in img_metas]
            src_to_ref = gt_points[0].new_tensor(np.array(src_to_ref))  # bs, 4, 4
            ref_to_tgt = [img_meta['total_ref2cur_lidar_transform'][tgt_frame_idx] for img_meta in img_metas]
            ref_to_tgt = gt_points[0].new_tensor(np.array(ref_to_tgt))  # bs, 4, 4
            src_to_tgt = torch.matmul(src_to_ref, ref_to_tgt)

            # 3. transfer src_frame_gt_points to src_to_tgt.
            aligned_gt_points_per_frame = []
            for batch_idx, points in enumerate(src_frame_gt_points):
                new_points = points.clone()  # -1, 4
                new_points = torch.cat([
                    new_points[:, :3], new_points.new_ones(new_points.shape[0], 1)
                ], 1)
                new_points = torch.matmul(new_points, src_to_tgt[batch_idx])
                new_points[..., -1] = frame_idx
                aligned_gt_points_per_frame.append(new_points)
            aligned_gt_points.append(aligned_gt_points_per_frame)

            # 4. obtain the aligned origin points.
            aligned_origin_points = torch.from_numpy(
                np.zeros((bs, 1, 3))).to(src_to_tgt.dtype).to(src_to_tgt.device)
            aligned_origin_points = torch.cat([
                aligned_origin_points[..., :3], torch.ones_like(aligned_origin_points)[..., 0:1]
            ], -1)
            aligned_origin_points = torch.matmul(aligned_origin_points, src_to_tgt)
            batched_origin_points.append(aligned_origin_points[..., :3].contiguous())

        # stack points from different timestamps, and transfer to occupancy representation.
        batched_gt_points = []
        for b in range(bs):
            cur_gt_points = [
                aligned_gt_points[frame_idx][b]
                for frame_idx in range(len(src_frame_idx_list))]
            cur_gt_points = torch.cat(cur_gt_points, 0)
            batched_gt_points.append(cur_gt_points)

        batched_origin_points = torch.cat(batched_origin_points, 1)
        return batched_gt_points, batched_origin_points

    @force_fp32(apply_to=('pred_dict'))
    def loss(self,
             pred_dict,
             gt_points,
             start_idx,
             tgt_bev_h,
             tgt_bev_w,
             tgt_pc_range,
             pred_frame_num,
             img_metas=None,
             batched_origin_points=None):
        """"Compute loss for all history according to gt_points.

        gt_points: ground-truth point cloud in each frame.
            list of tensor with shape [-1, 5], indicating ground-truth point cloud in
            each frame.
        """
        bev_preds = pred_dict['next_bev_preds']
        valid_frames = np.array(pred_dict['valid_frames'])
        start_frames = (valid_frames + self.history_queue_length - self.pred_history_frame_num)
        tgt_frames = valid_frames + self.history_queue_length

        full_prev_bev_exists = pred_dict.get('full_prev_bev_exists', True)
        if not full_prev_bev_exists:
            frame_idx_for_loss = [self.pred_history_frame_num] * self.pred_frame_num
        else:
            frame_idx_for_loss = np.arange(0, self.pred_frame_num)

        loss_dict = dict()
        for idx, i in enumerate(frame_idx_for_loss):
            # 1. get the predicted occupancy of frame-i.
            cur_bev_preds = bev_preds[:, :, i, ...].contiguous()

            # 2. get the frame index of current frame.
            src_frames = start_frames + i

            # 3. get gt_points belonging to cur_valid_frames.
            cur_gt_points, cur_origin_points = self._get_reference_gt_points(
                gt_points,
                src_frame_idx_list=src_frames,
                tgt_frame_idx_list=tgt_frames,
                img_metas=img_metas)

            # 4. compute loss.
            if i != self.pred_history_frame_num:
                # For aux history-future supervision:
                #  only compute loss for cur_frame prediction.
                loss_weight = np.array([[1]] + [[0]] * (len(self.loss_weight) - 1))
            else:
                loss_weight = self.loss_weight

            cur_loss_dict = super().loss(
                dict(next_bev_preds=cur_bev_preds,
                     valid_frames=np.arange(0, len(src_frames))),
                cur_gt_points,
                start_idx=start_idx,
                tgt_bev_h=tgt_bev_h,
                tgt_bev_w=tgt_bev_w,
                tgt_pc_range=tgt_pc_range,
                pred_frame_num=len(self.loss_weight)-1,
                img_metas=img_metas,
                batched_origin_points=cur_origin_points,
                loss_weight=loss_weight)

            # 5. merge dict.
            cur_frame_loss_weight = self.per_frame_loss_weight[i]
            cur_frame_loss_weight = cur_frame_loss_weight * (idx == i)
            for k, v in cur_loss_dict.items():
                loss_dict.update({f'frame.{idx}.{k}.loss': v * cur_frame_loss_weight})
        return loss_dict

    @force_fp32(apply_to=('pred_dict'))
    def get_point_cloud_prediction(self,
                                   pred_dict,
                                   gt_points,
                                   start_idx,
                                   tgt_bev_h,
                                   tgt_bev_w,
                                   tgt_pc_range,
                                   img_metas=None,
                                   batched_origin_points=None):
        """"Generate point cloud prediction.
        """
        # pred_frame_num, inter_num, num_frame, bs, bev_h * bev_w, num_height_pred
        pred_dict['next_bev_preds'] = pred_dict['next_bev_preds'][:, :, self.pred_history_frame_num, ...].contiguous()

        valid_frames = np.array(pred_dict['valid_frames'])
        valid_gt_points, cur_origin_points = self._get_reference_gt_points(
            gt_points,
            src_frame_idx_list=valid_frames + self.history_queue_length,
            tgt_frame_idx_list=valid_frames + self.history_queue_length,
            img_metas=img_metas)
        return super().get_point_cloud_prediction(
            pred_dict=pred_dict,
            gt_points=valid_gt_points,
            start_idx=start_idx,
            tgt_bev_h=tgt_bev_h,
            tgt_bev_w=tgt_bev_w,
            tgt_pc_range=tgt_pc_range,
            img_metas=img_metas,
            batched_origin_points=cur_origin_points)
