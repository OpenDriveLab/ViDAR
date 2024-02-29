import copy
import mmcv
import numpy as np
from mmdet.datasets.builder import PIPELINES

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet3d.datasets.pipelines import VoxelBasedPointSampler


class LoadPointsFromMultiSweeps(object):
    """Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points.
            Defaults to False.
        test_mode (bool): If test_model=True used for testing, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(self,
                 sweeps_num=10,
                 load_dim=5,
                 use_dim=[0, 1, 2, 4],
                 file_client_args=dict(backend='disk'),
                 pad_empty_sweeps=False,
                 remove_close=False,
                 random_select=True,
                 load_future=False,
                 test_mode=False):
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode

        self.random_select = random_select
        self.load_future = load_future

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)
        return points

    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray | :obj:`BasePoints`): Sweep points.
            radius (float): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def _select_index(self, sweeps, ts):
        if len(sweeps) <= self.sweeps_num:
            choices = np.arange(len(sweeps))
        elif self.test_mode:
            choices = np.arange(self.sweeps_num)
        elif self.random_select:
            choices = np.random.choice(
                len(sweeps), self.sweeps_num, replace=False)
        else:
            # sort those sweeps by their timestamps:
            #    from close to cur_frame to distant to cur_frame, and select top.
            sweep_ts = [sweep['timestamp'] for sweep in sweeps]
            sweep_ts = np.array(sweep_ts) / 1e6
            ts_interval = np.abs(sweep_ts - ts)
            choices = np.argsort(ts_interval)
            choices = choices[:self.sweeps_num]
        return choices

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud \
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data. \
                Added key and value are described below.

                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point \
                    cloud arrays.
        """
        points = results['points']
        points.tensor[:, 4] = 0
        sweep_points_list = [points]
        ts = results['timestamp']
        if self.pad_empty_sweeps and len(results['sweeps']) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    sweep_points_list.append(self._remove_close(points))
                else:
                    sweep_points_list.append(points)
        else:
            choices = self._select_index(results['sweeps'], ts)
            if self.load_future:
                future_choices = self._select_index(results['future_sweeps'], ts)
                future_choices = future_choices + len(results['sweeps'])
                results['sweeps'] = (copy.deepcopy(results['sweeps']) +
                                     copy.deepcopy(results['future_sweeps']))
                choices = np.concatenate([choices, future_choices])

            for idx in choices:
                sweep = results['sweeps'][idx]
                points_sweep = self._load_points(sweep['data_path'])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)
                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                sweep_ts = sweep['timestamp'] / 1e6
                points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                    'sensor2lidar_rotation'].T
                points_sweep[:, :3] += sweep['sensor2lidar_translation']
                points_sweep[:, 4] = ts - sweep_ts
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)
        points = points[:, self.use_dim]
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'


@PIPELINES.register_module()
class CustomLoadPointsFromMultiSweeps(LoadPointsFromMultiSweeps):
    def __init__(self,
                 ego_mask=None,
                 hard_sweeps_timestamp=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.ego_mask = ego_mask

        # if hard_sweeps_timestamp:
        #  set timestamps of all points to {hard_sweeps_timestamp}.
        self.hard_sweeps_timestamp = hard_sweeps_timestamp

    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray | :obj:`BasePoints`): Sweep points.
            radius (float): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        points = super()._remove_close(points, radius)

        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        if self.ego_mask is not None:
            # remove points belonging to ego vehicle.
            ego_mask = np.logical_and(
                np.logical_and(self.ego_mask[0] <= points_numpy[:, 0],
                               self.ego_mask[2] >= points_numpy[:, 0]),
                np.logical_and(self.ego_mask[1] <= points_numpy[:, 1],
                               self.ego_mask[3] >= points_numpy[:, 1]),
            )
            not_ego = np.logical_not(ego_mask)
            points = points[not_ego]
        return points

    def __call__(self, results):
        results = super().__call__(results)

        if self.hard_sweeps_timestamp is not None:
            points = results['points']
            points.tensor[:, 4] = self.hard_sweeps_timestamp
            results['points'] = points
        return results


@PIPELINES.register_module()
class CustomVoxelBasedPointSampler(VoxelBasedPointSampler):
    def _sample_points(self, points, sampler, point_dim):
        """Sample points for each points subset.

        Args:
            points (np.ndarray): Points subset to be sampled.
            sampler (VoxelGenerator): Voxel based sampler for
                each points subset.
            point_dim (int): The dimention of each points

        Returns:
            np.ndarray: Sampled points.
        """
        voxels, coors, num_points_per_voxel = sampler.generate(points)
        return voxels
