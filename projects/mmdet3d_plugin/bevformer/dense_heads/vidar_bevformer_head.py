#---------------------------------------------------------------------------------#
# Visual Point Cloud Forecasting enables Scalable Autonomous Driving              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

import torch
import torch.nn as nn

from mmdet.models import HEADS
from mmcv.runner import force_fp32, auto_fp16

from .bevformer_head import BEVFormerHead


@HEADS.register_module()
class ViDARBEVFormerHead(BEVFormerHead):
    """Custom BEV Former Head without cls/reg branch and decoder transformer.
    """
    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        del self.transformer.reference_points

    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self, mlvl_feats, img_metas, prev_bev=None,  only_bev=False,
                return_intermediate=False,):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder.
            return_intermediate: return all intermediate results for supervision?
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        assert only_bev
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        bev_queries = self.bev_embedding.weight.to(dtype)
        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        return self.transformer.get_bev_features(
            mlvl_feats,
            bev_queries,
            self.bev_h,
            self.bev_w,
            grid_length=(self.real_h / self.bev_h,
                         self.real_w / self.bev_w),
            bev_pos=bev_pos,
            img_metas=img_metas,
            prev_bev=prev_bev,
            return_intermediate=return_intermediate,
        )