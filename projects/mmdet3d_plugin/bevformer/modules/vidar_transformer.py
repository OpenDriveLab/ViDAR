#---------------------------------------------------------------------------------#
# Visual Point Cloud Forecasting enables Scalable Autonomous Driving              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

"""
ViDAR future decoder.
"""

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner.base_module import BaseModule

from mmdet.models.utils.builder import TRANSFORMER
from torch.nn.init import normal_
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from mmcv.runner.base_module import BaseModule
from torchvision.transforms.functional import rotate
from .temporal_self_attention import TemporalSelfAttention
from .spatial_cross_attention import MSDeformableAttention3D
from .decoder import CustomMSDeformableAttention
from projects.mmdet3d_plugin.models.utils.bricks import run_time
from mmcv.runner import force_fp32, auto_fp16


@TRANSFORMER.register_module()
class PredictionTransformer(BaseModule):
    """Implementations of End-to-End Future Prediction Transformer.
    """

    def __init__(self,
                 decoder=None,
                 embed_dims=256,
                 **kwargs):
        super().__init__(**kwargs)
        # Decode the next BEV feature from multi-frame BEV features.
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = embed_dims

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) \
                    or isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()

    @auto_fp16(apply_to=('prev_feats', 'bev_queries', 'bev_pos'))
    def get_bev_features(
            self,
            prev_feats,
            bev_queries,
            tgt_points,
            ref_points,
            bev_pos,
            bev_h,
            bev_w,
            **kwargs):
        """
        predict the next BEV features from prev_feats queue.

        Args:
            prev_feats: aligned bev features in previous frames with shape as
                [b, num_frames, bev_h * bev_w, dims]
            bev_queries: BEV queries of the next frame with shape
                [b, bev_h * bev_w, dims]
            bev_pos: positional embedding of bev-queries. Implemented from DETR,
                with a shape as [bs, dims, bev_h, bev_w]
            tgt_points: positions of points in deformable self-attention layers.
                positions of query points in reference frame coordinates.
            ref_points: positions of points in deformable cross-attention layers.
                positions of query points in previous frame coordinates.
        """
        # align bev_queries and bev_pos to the same shape.
        #  b, bev_h * bev_w, dims
        bev_pos = bev_pos.flatten(2).permute(0, 2, 1).contiguous()
        bev_embed = self.decoder(
            bev_queries,
            prev_feats,
            tgt_points=tgt_points,
            ref_points=ref_points,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            **kwargs)  # layer_num, bs, bev_h * bev_w, dims
        return bev_embed

    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'object_query_embed', 'prev_bev', 'bev_pos'))
    def forward(self,
                prev_feats,
                bev_queries,
                tgt_points,
                ref_points,
                bev_h,
                bev_w,
                bev_pos,
                **kwargs):
        bev_embed = self.get_bev_features(
            prev_feats=prev_feats,
            bev_queries=bev_queries,
            tgt_points=tgt_points,
            ref_points=ref_points,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            **kwargs)  # inter_num, bs, bev_h*bev_w, embed_dims
        return bev_embed
