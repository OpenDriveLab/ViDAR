import torch
import copy
import warnings
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.runner import force_fp32, auto_fp16
from mmcv.cnn.bricks.transformer import build_attention
from mmdet.models.utils.builder import TRANSFORMER

from .custom_base_transformer_layer import MyCustomBaseTransformerLayer
from .encoder import BEVFormerEncoder
from .transformer import PerceptionTransformer
from .ray_operations import LatentRendering


@TRANSFORMER.register_module()
class CustomPerceptionTransformer(PerceptionTransformer):

    def init_weights(self):
        """Initialize the transformer weights."""
        for m in self.modules():
            if isinstance(m, LatentRendering):
                m.init_weights()


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class CustomBEVFormerEncoder(BEVFormerEncoder):
    def __init__(self,
                 keep_idx=(2,),
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.keep_idx = keep_idx
        # remove latent rendering in previous layers.
        for lid, layer in enumerate(self.layers):
            if lid not in self.keep_idx:
                # if this is not the last layer, and remove operations in previous layers.
                if getattr(layer, 'latent_render', None):
                    del layer.latent_render
                    layer.operation_order = ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')

    @auto_fp16()
    def forward(self, *args, **kwargs):
        default_return_intermediate = self.return_intermediate
        self.return_intermediate = kwargs.get('return_intermediate', self.return_intermediate)
        ret = super().forward(*args, **kwargs)
        self.return_intermediate = default_return_intermediate
        return ret


@TRANSFORMER_LAYER.register_module()
class BEVFormerLayerV2(MyCustomBaseTransformerLayer):
    """BEVFormerLayerV2, enhanced with ray-aware deformable attention.

    The corresponding ray-aware attention layer is responsible for summarizing ray-pretraining results
    from our proposed point cloud pretrain.
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
                key=None,
                value=None,
                bev_pos=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ref_2d=None,
                ref_3d=None,
                bev_h=None,
                bev_w=None,
                reference_points_cam=None,
                mask=None,
                spatial_shapes=None,
                level_start_index=None,
                prev_bev=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
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

        for layer in self.operation_order:
            # temporal self attention
            if layer == 'self_attn':

                query = self.attentions[attn_index](
                    query,
                    prev_bev,
                    prev_bev,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=bev_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    reference_points=ref_2d,
                    spatial_shapes=torch.tensor(
                        [[bev_h, bev_w]], device=query.device),
                    level_start_index=torch.tensor([0], device=query.device),
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            # spaital cross attention
            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    mask=mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    **kwargs)
                attn_index += 1
                identity = query

            # unsupervised ray-wise marching operation.
            elif layer == 'latent_render':
                bs, token_num, embed_dim = query.shape
                query = self.latent_render(query.view(bs, bev_h, bev_w, embed_dim))
                query = query.view(bs, token_num, embed_dim)

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query
