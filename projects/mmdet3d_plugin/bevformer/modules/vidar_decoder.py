#---------------------------------------------------------------------------------#
# Visual Point Cloud Forecasting enables Scalable Autonomous Driving              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

import numpy as np
import math
import torch
import torch.nn as nn
import copy
import warnings
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.utils import ext_loader
from .custom_base_transformer_layer import MyCustomBaseTransformerLayer
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])
from .ray_operations import LatentRendering


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class PredictionDecoder(TransformerLayerSequence):

    """
    Decoder of End-to-End prediction transformer.
    Attention with both self and cross.
    """

    def __init__(self, *args,
                 return_intermediate=False,
                 keep_idx=(2,),
                 **kwargs):

        super().__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

        self.keep_idx = keep_idx
        # remove latent rendering in previous layers.
        for lid, layer in enumerate(self.layers):
            if lid not in self.keep_idx:
                # if this is not the last layer, and remove operations in previous layers.
                if getattr(layer, 'latent_render', None):
                    del layer.latent_render
                    layer.operation_order = ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')

    @auto_fp16()
    def forward(self,
                bev_query,
                prev_feats,
                *args,
                tgt_points=None,
                ref_points=None,
                bev_h=None,
                bev_w=None,
                bev_pos=None,
                **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            bev_query (Tensor): BEV queries with shape as [b, bev_h * bev_w, dims]
            prev_feats (Tensor): previous BEV features with shape as
                [b, num_frames, bev_h * bev_w, dims]
            bev_pos (Tensor): bev positional embedding with shape as
                [bs, bev_h * bev_w, dims]
            tgt_points: positions of points in deformable self-attention layers.
                positions of query points in reference frame coordinates with shape
                as [bs, tgt_bev_h * tgt_bev_w, 2]
            ref_points: positions of points in deformable cross-attention layers.
                positions of query points in previous frame coordinates with shape
                as [bs, ref_bev_h * ref_bev_w, 2]
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = bev_query
        intermediate = []

        for lid, layer in enumerate(self.layers):
            output = layer(
                bev_query,
                prev_feats,
                *args,
                bev_pos=bev_pos,
                tgt_points=tgt_points,
                ref_points=ref_points,
                bev_h=bev_h,
                bev_w=bev_w,
                **kwargs)

            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


@TRANSFORMER_LAYER.register_module()
class PredictionTransformerLayer(MyCustomBaseTransformerLayer):
    """Implements decoder layer in End-to-End future point
    cloud prediction network.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 latent_render=None,
                 **kwargs):
        super().__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        self.fp16_enabled = False
        if latent_render is not None:
            self.latent_render = LatentRendering(**latent_render)

    def forward(self,
                query,
                prev_feats=None,
                bev_pos=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                tgt_points=None,
                ref_points=None,
                bev_h=None,
                bev_w=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [bs, num_queries, dims]
            prev_feats (Tensor): The key / value tensor in cross-attention
                with shape [bs, num_frames, num_keys, dims]
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                f'operation_order {self.num_attn}'

        # Pre-process some parameters:
        #   * change ref_points from [bs, bev_h * bev_w, 2] to
        #     [bs, bev_h * bev_w, 1, 2] where 1 stands for num_level.
        tgt_points = tgt_points.unsqueeze(2)
        if len(ref_points.shape) != 4:
            ref_points = ref_points.unsqueeze(2)
        #   * change prev_feats to from [bs, num_frames, bev_h * bev_w, dims]
        #     to [bs, num_frames * bev_h * bev_w, dims]
        bs, num_frames, prev_token_num, prev_dims = prev_feats.shape
        assert prev_feats.shape[2] == bev_h * bev_w

        self_attn_spatial_shapes = torch.tensor(
            [[bev_h, bev_w]], device=query.device)
        self_attn_level_start_index = torch.tensor([0], device=query.device)

        cross_attn_spatial_shapes = torch.tensor(
            [[bev_h, bev_w] for i in range(num_frames)], device=query.device)
        cross_attn_level_start_index = torch.cat((cross_attn_spatial_shapes.new_zeros(
            (1,)), cross_attn_spatial_shapes.prod(1).cumsum(0)[:-1]))
        prev_feats = prev_feats.view(bs, num_frames * prev_token_num, prev_dims)

        for layer in self.operation_order:
            # temporal self attention
            if layer == 'self_attn':
                query = self.attentions[attn_index](
                    query,
                    None,
                    None,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=bev_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    reference_points=tgt_points,
                    spatial_shapes=self_attn_spatial_shapes,
                    level_start_index=self_attn_level_start_index,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            # Temporal cross-attention for query features from history memory bank.
            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    prev_feats,
                    prev_feats,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    # use ref_points coordinates in previous frames.
                    reference_points=ref_points,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=cross_attn_spatial_shapes,
                    level_start_index=cross_attn_level_start_index,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'latent_render':
                bs, token_num, embed_dim = query.shape
                query = self.latent_render(query.view(bs, bev_h, bev_w, embed_dim))
                query = query.view(bs, token_num, embed_dim)

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query


from mmcv.utils import ConfigDict, build_from_cfg, deprecated_api_warning, to_2tuple
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmcv.cnn import xavier_init, constant_init
from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32, \
    MultiScaleDeformableAttnFunction_fp16
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
@ATTENTION.register_module()
class PredictionMSDeformableAttention(BaseModule):
    """An attention module used in Deformable-Detr.

    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.

    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.fp16_enabled = False

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        assert self.batch_first
        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MultiScaleDeformableAttention')
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                flag='decoder',
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (Tensor): Query of Transformer with shape
                (bs, num_query, embed_dims).
            key (Tensor): The key tensor with shape
                `(bs, num_key, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(bs, num_key, embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes, with shape as
                [bs, num_query, 2]
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if value is None:
            value = query
        if key is None:
            key = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)

        # Predict sampling offsets / attention weight for each query.
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)

        # Compute the deformable location for reference query points.
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)  # 1, 2
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        # Input_shapes:
        #   * value: bs, num_value, num_heads, embed // num_heads.
        #       multi-level features are stacked at the {num_value} dimension
        #       and are split at this dimension during inference.
        #   * spatial_shapes: spatial shapes of each feature map.
        #       [-1, 2], where -1 means num_levels.
        #   * sampling_locations: The location of sampled points, with shape
        #       as [bs ,num_queries, num_heads, num_levels, num_points, 2]
        #       0-1 position at the value map scale.
        #   * attention_weights: The weight of sampled points with shape as
        #       [bs ,num_queries, num_heads, num_levels, num_points].
        if torch.cuda.is_available() and value.is_cuda:

            # using fp16 deformable attention is unstable because it performs many sum operations
            if value.dtype == torch.float16:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            else:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)
        return self.dropout(output) + identity
