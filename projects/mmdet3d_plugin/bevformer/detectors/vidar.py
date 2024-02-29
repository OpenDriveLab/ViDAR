#---------------------------------------------------------------------------------#
# Visual Point Cloud Forecasting enables Scalable Autonomous Driving              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

import mmcv
import torch
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
import copy
import numpy as np
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask

from .bevformer import BEVFormer
from mmdet3d.models import builder
from ..utils import e2e_predictor_utils, eval_utils


@DETECTORS.register_module()
class ViDAR(BEVFormer):
    def __init__(self,
                 # Future predictions.
                 future_pred_head,
                 future_pred_frame_num,  # number of future prediction frames.
                 test_future_frame_num,  # number of future prediction frames when testing.

                 # BEV configurations.
                 point_cloud_range,
                 bev_h,
                 bev_w,

                 # Augmentations.
                 # A1. randomly drop current image (to enhance temporal feature.)
                 random_drop_image_rate=0.0,
                 # A2. add noise to previous_bev_queue.
                 random_drop_prev_rate=0.0,
                 random_drop_prev_start_idx=1,
                 random_drop_prev_end_idx=None,
                 # A3. grid mask augmentation.
                 grid_mask_image=True,
                 grid_mask_backbone_feat=False,
                 grid_mask_fpn_feat=False,
                 grid_mask_prev=False,
                 grid_mask_cfg=dict(
                     use_h=True,
                     use_w=True,
                     rotate=1,
                     offset=False,
                     ratio=0.5,
                     mode=1,
                     prob=0.7
                 ),

                 # Supervision.
                 supervise_all_future=True,

                 _viz_pcd_flag=False,
                 _viz_pcd_path='dbg/pred_pcd',  # root/{prefix}

                 *args,
                 **kwargs,):

        super().__init__(*args, **kwargs)
        self.future_pred_head = builder.build_head(future_pred_head)
        self.future_pred_frame_num = future_pred_frame_num
        self.test_future_frame_num = test_future_frame_num
        # if not predict any future,
        #  then only predict current frame.
        self.only_train_cur_frame = (future_pred_frame_num == 0)

        self.point_cloud_range = point_cloud_range
        self.bev_h = bev_h
        self.bev_w = bev_w

        # Augmentations.
        self.random_drop_image_rate = random_drop_image_rate
        self.random_drop_prev_rate = random_drop_prev_rate
        self.random_drop_prev_start_idx = random_drop_prev_start_idx
        self.random_drop_prev_end_idx = random_drop_prev_end_idx

        # Grid mask.
        self.grid_mask_image = grid_mask_image
        self.grid_mask_backbone_feat = grid_mask_backbone_feat
        self.grid_mask_fpn_feat = grid_mask_fpn_feat
        self.grid_mask_prev = grid_mask_prev
        self.grid_mask = GridMask(**grid_mask_cfg)

        # Training configurations.
        # randomly sample one future for loss computation?
        self.supervise_all_future = supervise_all_future

        self._viz_pcd_flag = _viz_pcd_flag
        self._viz_pcd_path = _viz_pcd_path

        # remove the useless modules in pts_bbox_head
        #  * box/cls prediction head; decoder transformer.
        del self.pts_bbox_head.cls_branches, self.pts_bbox_head.reg_branches
        del self.pts_bbox_head.query_embedding
        del self.pts_bbox_head.transformer.decoder

        if self.only_train_cur_frame:
            # remove useless parameters.
            del self.future_pred_head.transformer
            del self.future_pred_head.bev_embedding
            del self.future_pred_head.prev_frame_embedding
            del self.future_pred_head.can_bus_mlp
            del self.future_pred_head.positional_encoding

    ####################### Image Feature Extraction. #######################
    @auto_fp16(apply_to=('img'))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        if ('aug_param' in img_metas[0] and
                img_metas[0]['aug_param'] is not None and
                img_metas[0]['aug_param']['CropResizeFlipImage_param'][-1] is True):
            img_feats = [torch.flip(x, dims=[-1, ]) for x in img_feats]

        return img_feats

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask and self.grid_mask_image:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
            if self.use_grid_mask and self.grid_mask_backbone_feat:
                new_img_feats = []
                for img_feat in img_feats:
                    img_feat = self.grid_mask(img_feat)
                    new_img_feats.append(img_feat)
                img_feats = new_img_feats
        else:
            return None

        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
            if self.use_grid_mask and self.grid_mask_fpn_feat:
                new_img_feats = []
                for img_feat in img_feats:
                    img_feat = self.grid_mask(img_feat)
                    new_img_feats.append(img_feat)
                img_feats = new_img_feats

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B / len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    ############# Align coordinates between reference (current frame) to other frames. #############
    def _get_history_ref_to_previous_transform(self, tensor, num_frames, img_metas_list):
        """Get transformation matrix from reference frame to all previous frames.

        Args:
            tensor: to convert {ref_to_prev_transform} to device and dtype.
            num_frames: total num of available history frames.
            img_metas_list: a list of batch_size items.
                In each item, there is {num_prev_frames} img_meta for transformation alignment.

        Return:
            ref_to_history_list (torch.Tensor): with shape as [bs, num_prev_frames, 4, 4]
        """
        ref_to_history_list = []
        for img_metas in img_metas_list:
            cur_ref_to_prev = [img_metas[i]['ref_lidar_to_cur_lidar'] for i in range(num_frames)]
            ref_to_history_list.append(cur_ref_to_prev)
        ref_to_history_list = tensor.new_tensor(np.array(ref_to_history_list))
        return ref_to_history_list

    def _align_bev_coordnates(self, frame_idx, ref_to_history_list, img_metas):
        """Align the bev_coordinates of frame_idx to each of history_frames.

        Args:
            frame_idx: the index of target frame.
            ref_to_history_list (torch.Tensor): a tensor with shape as [bs, num_prev_frames, 4, 4]
                indicating the transformation metric from reference to each history frames.
            img_metas: a list of batch_size items.
                In each item, there is one img_meta (reference frame)
                whose {future2ref_lidar_transform} & {ref2future_lidar_transform} are for
                transformation alignment.
        """
        bs, num_frame = ref_to_history_list.shape[:2]

        # 1. get future2ref and ref2future_matrix of frame_idx.
        future2ref = [img_meta['future2ref_lidar_transform'][frame_idx] for img_meta in img_metas]  # b, 4, 4
        future2ref = ref_to_history_list.new_tensor(np.array(future2ref))  # bs, 4, 4

        ref2future = [img_meta['ref2future_lidar_transform'][frame_idx] for img_meta in img_metas]  # b, 4, 4
        ref2future = ref_to_history_list.new_tensor(np.array(ref2future))  # bs, 4, 4

        # 2. compute the transformation matrix from current frame to all previous frames.
        future2ref = future2ref.unsqueeze(1).repeat(1, num_frame, 1, 1).contiguous()
        future_to_history_list = torch.matmul(future2ref, ref_to_history_list)

        # 3. compute coordinates of future frame.
        bev_grids = e2e_predictor_utils.get_bev_grids(
            self.bev_h, self.bev_w, bs * num_frame)  # bs * num_frame, bev_h, bev_w, 2 (x, y)
        bev_grids = bev_grids.view(bs, num_frame, -1, 2)
        bev_coords = e2e_predictor_utils.bev_grids_to_coordinates(
            bev_grids, self.point_cloud_range)

        # 4. align target coordinates of future frame to each of previous frames.
        aligned_bev_coords = torch.cat([
            bev_coords, torch.ones_like(bev_coords[..., :2])], -1)  # b, num_frame, h*w, 4
        aligned_bev_coords = torch.matmul(aligned_bev_coords, future_to_history_list)
        aligned_bev_coords = aligned_bev_coords[..., :2]  # b, num_frame, h*w, 2
        aligned_bev_grids, _ = e2e_predictor_utils.bev_coords_to_grids(
            aligned_bev_coords, self.bev_h, self.bev_w, self.point_cloud_range)
        aligned_bev_grids = (aligned_bev_grids + 1) / 2.  # range of [0, 1]
        # b, h*w, num_frame, 2
        aligned_bev_grids = aligned_bev_grids.permute(0, 2, 1, 3).contiguous()

        # 5. get target bev_grids at target future frame.
        tgt_grids = bev_grids[:, -1].contiguous()
        return tgt_grids, aligned_bev_grids, ref2future

    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      img=None,
                      gt_points=None,
                      img_depth=None,
                      img_mask=None,
                      ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            gt_points (torch.Tensor optional): groundtruth point clouds for future
                frames with shape (x, x, x). Defaults to None.
                The 0-th frame represents current frame for reference.
        Returns:
            dict: Losses of different branches.
        """
        # Augmentations.
        # A1. Randomly drop cur image input.
        if np.random.rand() < self.random_drop_image_rate:
            img[:, -1:, ...] = torch.zeros_like(img[:, -1:, ...])
        # A2. Randomly drop previous image inputs.
        num_frames = img.size(1)
        if np.random.rand() < self.random_drop_prev_rate:
            # randomly drop any frame.
            # 4 his + 1 cur --> 3 his + 1 cur + 1 future
            # if drop_prev_index == 2:
            #     2 his + 1 cur --> 3 his + 1 cur + 1 future
            # if drop_prev_index == 3:
            #     1 his + 1 cur --> 3 his + 1 cur + 1 future
            random_drop_prev_v2_end_idx = (
                self.random_drop_prev_end_idx if self.random_drop_prev_end_idx is not None
                else num_frames)
            drop_prev_index = np.random.randint(
                self.random_drop_prev_start_idx, random_drop_prev_v2_end_idx)
        else:
            drop_prev_index = -1

        # Extract history BEV features.
        # B1. Forward previous frames.
        prev_img = img[:, :-1, ...]
        prev_img_metas = copy.deepcopy(img_metas)
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas, drop_prev_index=drop_prev_index)
        # B2. Randomly grid-mask prev_bev.
        if self.grid_mask_prev and prev_bev is not None:
            b, n, c = prev_bev.shape
            assert n == self.bev_h * self.bev_w
            prev_bev = prev_bev.view(b, self.bev_h, self.bev_w, c)
            prev_bev = prev_bev.permute(0, 3, 1, 2).contiguous()
            prev_bev = self.grid_mask(prev_bev)
            prev_bev = prev_bev.view(b, c, n).permute(0, 2, 1).contiguous()

        # Extract current BEV features.
        # C1. Forward.
        img = img[:, -1, ...]
        img_metas = [each[num_frames-1] for each in img_metas]
        img_feats = self.extract_feat(img=img, img_metas=img_metas)  # b, 1, c, h, w
        if not img_metas[0]['prev_bev_exists']:
            prev_bev = None
        # C2. Check whether the frame has previous frames.
        prev_bev_exists_list = []
        assert len(prev_img_metas) == 1, 'Only supports bs=1 for now.'
        for prev_img_meta in prev_img_metas:  # Loop batch.
            max_key = len(prev_img_meta) - 1
            prev_bev_exists = True
            for k in range(max_key, -1, -1):
                each = prev_img_meta[k]
                prev_bev_exists_list.append(prev_bev_exists)
                prev_bev_exists = prev_bev_exists and each['prev_bev_exists']
        prev_bev_exists_list = np.array(prev_bev_exists_list)[::-1]
        # C3. BEVFormer Encoder Forward.
        # ref_bev: bs, bev_h * bev_w, c
        ref_bev = self.pts_bbox_head(img_feats, img_metas, prev_bev, only_bev=True)

        # Extract future BEV features.
        # D1. preparations.
        next_bev_feats = [
            # intermediate_num, bs, bev_h * bev_w, c
            ref_bev.unsqueeze(0).repeat(
                len(self.future_pred_head.bev_pred_head), 1, 1, 1).contiguous()
        ]  # auxiliary supervision on current frame.
        valid_frames = [0]
        # D2. Align previous frames to the reference coordinates.
        prev_bev_input = ref_bev.unsqueeze(1)  # bs, prev_frame_num, bev_h * bev_w, c
        prev_img_metas = [[each[num_frames-1]] for each in prev_img_metas]
        ref_to_history_list = self._get_history_ref_to_previous_transform(
            prev_bev_input, prev_bev_input.shape[1], prev_img_metas)
        # D3. future decoder forward.
        if not self.only_train_cur_frame:
            if self.supervise_all_future:
                valid_frames.extend(list(range(1, self.future_pred_frame_num + 1)))
            else:  # randomly select one future frame for computing loss to save memory cost.
                train_frame = np.random.choice(np.arange(1, self.future_pred_frame_num + 1), 1)[0]
                valid_frames.append(train_frame)

            for future_frame_index in range(1, self.future_pred_frame_num + 1):
                # 1. obtain the coordinates of future BEV query to previous frames.
                tgt_grids, aligned_prev_grids, ref2future = self._align_bev_coordnates(
                    future_frame_index, ref_to_history_list, img_metas)

                # 2. transform for generating freespace of future frame.
                # pred_feat: inter_num, bs, bev_h * bev_w, c
                if future_frame_index in valid_frames:  # compute loss if it is a valid frame.
                    pred_feat = self.future_pred_head(
                        prev_bev_input, img_metas, future_frame_index,
                        tgt_points=tgt_grids, bev_h=self.bev_h, bev_w=self.bev_w, ref_points=aligned_prev_grids)
                    next_bev_feats.append(pred_feat)
                else:
                    with torch.no_grad():
                        pred_feat = self.future_pred_head(
                            prev_bev_input, img_metas, future_frame_index,
                            tgt_points=tgt_grids, bev_h=self.bev_h, bev_w=self.bev_w, ref_points=aligned_prev_grids)

                # 3. update pred_feat to prev_bev_input and update ref_to_history_list.
                prev_bev_input = torch.cat([prev_bev_input, pred_feat[-1].unsqueeze(1)], 1)
                prev_bev_input = prev_bev_input[:, 1:, ...].contiguous()
                # update ref2future to ref_to_history_list.
                ref_to_history_list = torch.cat([ref_to_history_list, ref2future.unsqueeze(1)], 1)
                ref_to_history_list = ref_to_history_list[:, 1:].contiguous()
        # D4. compute loss.
        # next_bev_feats: frame_num, intermediate_num, bs, bev_h * bev_w, c
        next_bev_feats = torch.stack(next_bev_feats, 0)
        # next_bev_preds: frame_num, intermediate_num, bs, bev_h * bev_w, num_pred_height.
        next_bev_preds = self.future_pred_head.forward_head(next_bev_feats)
        pred_dict = {
            'next_bev_features': next_bev_feats,
            'next_bev_preds': next_bev_preds,
            'valid_frames': valid_frames,
            'full_prev_bev_exists': prev_bev_exists_list.all(),
            'prev_bev_exists_list': prev_bev_exists_list,
        }

        # 5. Compute loss for point cloud predictions.
        start_idx = 0
        losses = dict()
        loss_dict = self.future_pred_head.loss(
            pred_dict, gt_points, start_idx,
            tgt_bev_h=self.bev_h, tgt_bev_w=self.bev_w,
            tgt_pc_range=self.point_cloud_range,
            pred_frame_num=self.future_pred_frame_num+1,
            img_metas=img_metas)
        losses.update(loss_dict)
        return losses

    def forward_test(self, img_metas, img=None,
                     gt_points=None, **kwargs):
        """has similar implementation with train forward."""
        # 1. Extract history BEV features.
        num_frames = img.size(1)
        prev_bev = self.obtain_history_bev(img, img_metas)
        self.eval()

        # 2. Align previous frames to reference coordinates.
        prev_bev = prev_bev[:, None, ...].contiguous()  # bs, 1, bev_h * bev_w, c
        next_bev_feats, valid_frames = [], []
        ref_bev = prev_bev[:, -1].contiguous()
        next_bev_feats.append(ref_bev.unsqueeze(0).repeat(
            len(self.future_pred_head.bev_pred_head), 1, 1, 1
        ).contiguous())
        valid_frames.append(0)

        # 3. predict future BEV.
        valid_frames.extend(list(range(1, self.test_future_frame_num+1)))
        prev_img_metas = copy.deepcopy(img_metas)
        prev_img_metas = [[each[num_frames - 1]] for each in prev_img_metas]
        ref_to_history_list = self._get_history_ref_to_previous_transform(
            prev_bev, prev_bev.shape[1], prev_img_metas)
        img_metas = [each[num_frames - 1] for each in img_metas]
        for future_frame_index in range(1, self.test_future_frame_num + 1):
            # 1. obtain the coordinates of future BEV query to previous frames.
            tgt_grids, aligned_prev_grids, ref2future = self._align_bev_coordnates(
                future_frame_index, ref_to_history_list, img_metas)

            next_bev = self.future_pred_head(
                prev_bev, img_metas, future_frame_index,
                tgt_points=tgt_grids, bev_h=self.bev_h, bev_w=self.bev_w, ref_points=aligned_prev_grids)
            next_bev_feats.append(next_bev)

            # 3. update pred_feat to prev_bev_input and update ref_to_history_list.
            prev_bev = torch.cat([prev_bev, next_bev[-1].unsqueeze(1)], 1)
            prev_bev = prev_bev[:, 1:, ...].contiguous()
            # update ref2future to ref_to_history_list.
            ref_to_history_list = torch.cat([ref_to_history_list, ref2future.unsqueeze(1)], 1)
            ref_to_history_list = ref_to_history_list[:, 1:].contiguous()

        # pred_frame_num, inter_num, bs, bev_h * bev_w, embed_dims.
        next_bev_feats = torch.stack(next_bev_feats, 0)
        next_bev_preds = self.future_pred_head.forward_head(next_bev_feats)
        pred_dict = {
            'next_bev_features': next_bev_feats,
            'next_bev_preds': next_bev_preds,
            'valid_frames': valid_frames,
        }

        # decode results and compute some statistic results if needed.
        start_idx = 0
        decode_dict = self.future_pred_head.get_point_cloud_prediction(
            pred_dict, gt_points, start_idx,
            tgt_bev_h=self.bev_h, tgt_bev_w=self.bev_w,
            tgt_pc_range=self.point_cloud_range, img_metas=img_metas)

        # convert decode_dict to quantitative statistics.
        pred_pcds = decode_dict['pred_pcds']
        gt_pcds = decode_dict['gt_pcds']
        scene_origin = decode_dict['origin']

        pred_frame_num = len(pred_pcds[0])
        ret_dict = dict()
        for frame_idx in range(pred_frame_num):
            count = 0
            frame_name = frame_idx + start_idx
            ret_dict[f'frame.{frame_name}'] = dict(
                count=0,
                chamfer_distance=0,
                chamfer_distance_inner=0,
                l1_error=0,
                absrel_error=0,
            )
            for bs in range(len(pred_pcds)):
                pred_pcd = pred_pcds[bs][frame_idx]
                gt_pcd = gt_pcds[bs][frame_idx]

                ret_dict[f'frame.{frame_name}']['chamfer_distance'] += (
                    e2e_predictor_utils.compute_chamfer_distance(pred_pcd, gt_pcd).item())
                ret_dict[f'frame.{frame_name}']['chamfer_distance_inner'] += (
                    e2e_predictor_utils.compute_chamfer_distance_inner(
                        pred_pcd, gt_pcd, self.point_cloud_range).item())

                l1_error, absrel_error = eval_utils.compute_ray_errors(
                    pred_pcd.cpu().numpy(), gt_pcd.cpu().numpy(),
                    scene_origin[bs, frame_idx].cpu().numpy(), scene_origin.device)
                ret_dict[f'frame.{frame_name}']['l1_error'] += l1_error
                ret_dict[f'frame.{frame_name}']['absrel_error'] += absrel_error

                if self._viz_pcd_flag:
                    cur_name = img_metas[bs]['sample_idx']
                    out_path = f'{self._viz_pcd_path}_{cur_name}_{frame_name}.png'
                    gt_inside_mask = e2e_predictor_utils.get_inside_mask(gt_pcd, self.point_cloud_range)
                    gt_pcd_inside = gt_pcd[gt_inside_mask]
                    pred_pcd_inside = pred_pcd[gt_inside_mask]
                    root_path = '/'.join(out_path.split('/')[:-1])
                    mmcv.mkdir_or_exist(root_path)
                    self._viz_pcd(
                        pred_pcd_inside.cpu().numpy(),
                        scene_origin[bs, frame_idx].cpu().numpy()[None, :],
                        output_path=out_path,
                        gt_pcd=gt_pcd_inside.cpu().numpy()
                    )

                count += 1
            ret_dict[f'frame.{frame_name}']['count'] = count

        if self._viz_pcd_flag:
            print('==== Visualize predicted point clouds done!! End the program. ====')
            print(f'==== The visualized point clouds are stored at {out_path} ====')
        return [ret_dict]

    def _viz_pcd(self, pred_pcd, pred_ctr,  output_path, gt_pcd=None):
        """Visualize predicted future point cloud."""
        color_map = np.array([
            [0, 0, 230], [219, 112, 147], [255, 0, 0]
        ])
        pred_label = np.ones_like(pred_pcd)[:, 0].astype(np.int) * 0
        if gt_pcd is not None:
            gt_label = np.ones_like(gt_pcd)[:, 0].astype(np.int)

            pred_label = np.concatenate([pred_label, gt_label], 0)
            pred_pcd = np.concatenate([pred_pcd, gt_pcd], 0)

        e2e_predictor_utils._dbg_draw_pc_function(
            pred_pcd, pred_label, color_map, output_path=output_path,
            ctr=pred_ctr, ctr_labels=np.zeros_like(pred_ctr)[:, 0].astype(np.int)
        )
