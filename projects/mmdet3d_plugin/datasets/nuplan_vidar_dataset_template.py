#---------------------------------------------------------------------------------#
# Visual Point Cloud Forecasting enables Scalable Autonomous Driving              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

import os
import copy

from .nuscenes_vidar_dataset_template import NuScenesViDARDatasetTemplate
import mmcv
from mmdet.datasets import DATASETS
import numpy as np
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion


@DATASETS.register_module()
class NuPlanViDARDatasetTemplate(NuScenesViDARDatasetTemplate):
    r"""nuPlan dataset for visual point cloud forecasting.
    """
    def load_annotations(self, ann_file):
        """Rewrite load_annotation files for nuplan.

        ann_file: the root of nuplan pickle files.
        """
        data_infos = []
        for file in os.listdir(ann_file):
            path = os.path.join(ann_file, file)
            if file.endswith('.pkl'):
                data_infos.extend(mmcv.load(path))
        data_infos = data_infos[::self.load_interval]
        return data_infos

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch

        # For BEVFormer alignment.
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = Quaternion(info['lidar2ego_rotation']).rotation_matrix
        lidar2ego[:3, 3] = np.array(info['lidar2ego_translation'])
        ego2global = np.eye(4)
        ego2global[:3, :3] = Quaternion(info['ego2global_rotation']).rotation_matrix
        ego2global[:3, 3] = np.array(info['ego2global_translation'])
        lidar2global_rotation = ego2global[:3, :3] @ lidar2ego[:3, :3]

        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=os.path.join(self.data_root, info['lidar_path']),
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            lidar2global_rotation=lidar2global_rotation,
            prev_idx=info['sample_prev'],
            next_idx=info['sample_next'],
            sweeps=[],
            scene_token=info['scene_token'],
            can_bus=info['can_bus'],
            frame_idx=info['frame_idx'],
            timestamp=info['timestamp'] / 1e6,
        )

        image_paths = []
        lidar2img_rts = []
        lidar2cam_rts = []
        cam_intrinsics = []
        for cam_type, cam_info in info['cams'].items():
            image_paths.append(
                os.path.join(self.data_root, cam_info['data_path']))
            # obtain lidar to image transformation matrix
            lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
            lidar2cam_t = cam_info[
                'sensor2lidar_translation'] @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            intrinsic = cam_info['cam_intrinsic']
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2img_rt = (viewpad @ lidar2cam_rt.T)
            lidar2img_rts.append(lidar2img_rt)

            cam_intrinsics.append(viewpad)
            lidar2cam_rts.append(lidar2cam_rt.T)

        input_dict.update(
            dict(
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
                cam2img=cam_intrinsics,
                lidar2cam=lidar2cam_rts,
                lidar2ego_translation=info['lidar2ego_translation'],
                lidar2ego_rotation=info['lidar2ego_rotation'],
            ))

        rotation = Quaternion(input_dict['ego2global_rotation'])
        translation = input_dict['ego2global_translation']
        can_bus = input_dict['can_bus']
        # Overwrite the canbus pos&rot information.
        can_bus[:3] = translation
        can_bus[3:7] = rotation
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle
        return input_dict