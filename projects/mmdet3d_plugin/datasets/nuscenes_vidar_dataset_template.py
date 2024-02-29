#---------------------------------------------------------------------------------#
# Visual Point Cloud Forecasting enables Scalable Autonomous Driving              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

import copy

from .nuscenes_dataset import CustomNuScenesDataset
import mmcv
from mmdet.datasets import DATASETS
import numpy as np


@DATASETS.register_module()
class NuScenesViDARDatasetTemplate(CustomNuScenesDataset):
    r"""VIDAR dataset for visual point cloud forecasting.
    """

    def __init__(self,
                 future_length,
                 ego_mask=None,
                 load_frame_interval=None,
                 rand_frame_interval=(1,),

                 *args,
                 **kwargs):
        """
        Args:
            future_length: the number of predicted future point clouds.
            ego_mask: mask points belonging to the ego vehicle.
            load_frame_interval: partial of training set.
            rand_frame_interval: augmentation for future prediction.
        """
        # Hack the original {self._set_group_flag} function.
        self.usable_index = []

        super().__init__(*args, **kwargs)
        self.future_length = future_length
        self.ego_mask = ego_mask
        self.load_frame_interval = load_frame_interval
        self.rand_frame_interval = rand_frame_interval

        # Remove data_infos without enough history & future.
        # if test, assert all history frames are available
        #  Align with the setting of 4D-occ: https://github.com/tarashakhurana/4d-occ-forecasting
        last_scene_index = None
        last_scene_frame = -1
        usable_index = []
        valid_prev_length = (self.queue_length if self.test_mode else 0)
        for index, info in enumerate(mmcv.track_iter_progress(self.data_infos)):
            if last_scene_index != info['scene_token']:
                last_scene_index = info['scene_token']
                last_scene_frame = -1
            last_scene_frame += 1
            if last_scene_frame >= valid_prev_length:
                # has enough previous frame.
                # now, let's check whether it has enough future frame.
                tgt_future_index = index + self.future_length
                if tgt_future_index >= len(self.data_infos):
                    break
                if last_scene_index != self.data_infos[tgt_future_index]['scene_token']:
                    # the future scene is not corresponded to the current scene
                    continue
                usable_index.append(index)

        # Remove useless frame index if load_frame_interval is assigned.
        if self.load_frame_interval is not None:
            usable_index = usable_index[::self.load_frame_interval]
        self.usable_index = usable_index

        if not self.test_mode:
            self._set_group_flag()

    def get_data_info(self, index):
        """Also return lidar2ego transformations."""
        input_dict = super().get_data_info(index)

        info = self.data_infos[index]
        input_dict.update(dict(
            lidar2ego_translation=info['lidar2ego_translation'],
            lidar2ego_rotation=info['lidar2ego_rotation'],
            cam2img=input_dict['cam_intrinsic'],
        ))
        return input_dict

    def _prepare_data_info_single(self, index, aug_param=None):
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        if aug_param is not None:
            input_dict['aug_param'] = copy.deepcopy(aug_param)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example
        
    def _prepare_data_info(self, index, rand_interval=None):
        """
        Modified from BEVFormer:CustomNuScenesDataset,
            BEVFormer logits: randomly select (queue_length-1) previous images.
            Modified logits: directly select (queue_length) previous images.
        """
        rand_interval = (
            rand_interval if rand_interval is not None else
            np.random.choice(self.rand_frame_interval, 1)[0]
        )

        # 1. get previous camera information.
        previous_queue = []
        previous_index_list = list(range(
            index - self.queue_length * rand_interval, index, rand_interval))
        previous_index_list = sorted(previous_index_list)
        if rand_interval < 0:  # the inverse chain.
            previous_index_list = previous_index_list[::-1]
        previous_index_list.append(index)
        aug_param = None
        for i in previous_index_list:
            i = min(max(0, i), len(self.data_infos) - 1)
            example = self._prepare_data_info_single(i, aug_param=aug_param)

            aug_param = copy.deepcopy(example['aug_param']) if 'aug_param' in example else None
            if example is None:
                return None
            previous_queue.append(example)

        # 2. get future lidar information.
        future_queue = []
        # Future: from current to future frames.
        # use current frame as the 0-th future.
        future_index_list = list(range(
            index, index + (self.future_length + 1) * rand_interval, rand_interval))
        future_index_list = sorted(future_index_list)
        if rand_interval < 0:  # the inverse chain.
            future_index_list = future_index_list[::-1]
        has_future = False
        for i in future_index_list:
            i = min(max(0, i), len(self.data_infos) - 1)
            example = self._prepare_data_info_single(i)
            if example is None and not has_future:
                return None
            future_queue.append(example)
            has_future = True
        return self.union2one(previous_queue, future_queue)

    def union2one(self, previous_queue, future_queue):
        pass

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluate nuScenes future point cloud prediction result.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        print('Start to convert nuScenes future prediction metric...')
        res_dict_all = None
        for sample_id, result in enumerate(mmcv.track_iter_progress(results)):
            if res_dict_all is None:
                res_dict_all = result
            else:
                for frame_k, frame_res in result.items():
                    for k, v in frame_res.items():
                        res_dict_all[frame_k][k] += v
        print('Summary all metrics together ...')
        for frame_k, frame_res in res_dict_all.items():
            frame_count = res_dict_all[frame_k]['count']
            for k, v in frame_res.items():
                if k == 'count': continue
                frame_res[k] = v / frame_count

        print('Evaluation Done. Printing all metrics ...')
        for frame_k, frame_res in res_dict_all.items():
            print(f'==== {frame_k} results: ====')
            for k, v in frame_res.items():
                print(f'{k}: {v}')
        return res_dict_all

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        rand_interval = None
        while True:
            data = self._prepare_data_info(
                self.usable_index[idx], rand_interval=rand_interval)
            if data is None:
                if self.test_mode:
                    idx += 1
                else:
                    if rand_interval is None:
                        rand_interval = 1  # use rand_interval = 1 for the same sample again.
                    else:  # still None for rand_interval = 1, no enough future.
                        idx = self._rand_another(idx)
                        rand_interval = None
                continue
            assert data is not None
            return data

    def __len__(self):
        return len(self.usable_index)
