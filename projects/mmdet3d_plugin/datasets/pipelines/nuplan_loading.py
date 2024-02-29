import copy
import mmcv
import numpy as np
from mmdet.datasets.builder import PIPELINES

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet3d.datasets.pipelines import VoxelBasedPointSampler

from io import BytesIO
from typing import IO, Any, List, NamedTuple
class PointCloudHeader(NamedTuple):
    """Class for Point Cloud header."""

    version: str
    fields: List[str]
    size: List[int]
    type: List[str]
    count: List[int]  # type: ignore
    width: int
    height: int
    viewpoint: List[int]
    points: int
    data: str


class PointCloud:
    """
    Class for raw .pcd file.
    """

    def __init__(self, header, points) -> None:
        """
        PointCloud.
        :param header: Pointcloud header.
        :param points: <np.ndarray, X, N>. X columns, N points.
        """
        self._header = header
        self._points = points

    @property
    def header(self) -> PointCloudHeader:
        """
        Returns pointcloud header.
        :return: A PointCloudHeader instance.
        """
        return self._header

    @property
    def points(self):
        """
        Returns points.
        :return: <np.ndarray, X, N>. X columns, N points.
        """
        return self._points

    def save(self, file_path: str) -> None:
        """
        Saves to .pcd file.
        :param file_path: The path to the .pcd file.
        """
        with open(file_path, 'wb') as fp:
            fp.write('# .PCD v{} - Point Cloud Data file format\n'.format(self._header.version).encode('utf8'))
            for field in self._header._fields:
                value = getattr(self._header, field)
                if isinstance(value, list):
                    text = ' '.join(map(str, value))
                else:
                    text = str(value)
                fp.write('{} {}\n'.format(field.upper(), text).encode('utf8'))
            fp.write(self._points.tobytes())

    @classmethod
    def parse(cls, pcd_content: bytes):
        """
        Parses the pointcloud from byte stream.
        :param pcd_content: The byte stream that holds the pcd content.
        :return: A PointCloud object.
        """
        with BytesIO(pcd_content) as stream:
            header = cls.parse_header(stream)
            points = cls.parse_points(stream, header)
            return cls(header, points)

    @classmethod
    def parse_from_file(cls, pcd_file: str):
        """
        Parses the pointcloud from .pcd file on disk.
        :param pcd_file: The path to the .pcd file.
        :return: A PointCloud instance.
        """
        with open(pcd_file, 'rb') as stream:
            header = cls.parse_header(stream)
            points = cls.parse_points(stream, header)
            return cls(header, points)

    @staticmethod
    def parse_header(stream: IO[Any]) -> PointCloudHeader:
        """
        Parses the header of a pointcloud from byte IO stream.
        :param stream: Binary stream.
        :return: A PointCloudHeader instance.
        """
        headers_list = []
        while True:
            line = stream.readline().decode('utf8').strip()
            if line.startswith('#'):
                continue
            columns = line.split()
            key = columns[0].lower()
            val = columns[1:] if len(columns) > 2 else columns[1]
            headers_list.append((key, val))

            if key == 'data':
                break

        headers = dict(headers_list)
        headers['size'] = list(map(int, headers['size']))
        headers['count'] = list(map(int, headers['count']))
        headers['width'] = int(headers['width'])
        headers['height'] = int(headers['height'])
        headers['viewpoint'] = list(map(int, headers['viewpoint']))
        headers['points'] = int(headers['points'])
        header = PointCloudHeader(**headers)

        if any([c != 1 for c in header.count]):
            raise RuntimeError('"count" has to be 1')

        if not len(header.fields) == len(header.size) == len(header.type) == len(header.count):
            raise RuntimeError('fields/size/type/count field number are inconsistent')

        return header

    @staticmethod
    def parse_points(stream: IO[Any], header: PointCloudHeader):
        """
        Parses points from byte IO stream.
        :param stream: Byte stream that holds the points.
        :param header: <np.ndarray, X, N>. A numpy array that has X columns(features), N points.
        :return: Points of Point Cloud.
        """
        if header.data != 'binary':
            raise RuntimeError('Un-supported data foramt: {}. "binary" is expected.'.format(header.data))

        # There is garbage data at the end of the stream, usually all b'\x00'.
        row_type = PointCloud.np_type(header)
        length = row_type.itemsize * header.points
        buff = stream.read(length)
        if len(buff) != length:
            raise RuntimeError('Incomplete pointcloud stream: {} bytes expected, {} got'.format(length, len(buff)))

        points = np.frombuffer(buff, row_type)

        return points

    @staticmethod
    def np_type(header: PointCloudHeader) -> np.dtype:  # type: ignore
        """
        Helper function that translate column types in pointcloud to np types.
        :param header: A PointCloudHeader object.
        :return: np.dtype that holds the X features.
        """
        type_mapping = {'I': 'int', 'U': 'uint', 'F': 'float'}
        np_types = [type_mapping[t] + str(int(s) * 8) for t, s in zip(header.type, header.size)]

        return np.dtype([(f, getattr(np, nt)) for f, nt in zip(header.fields, np_types)])

    def to_pcd_bin(self):
        """
        Converts pointcloud to .pcd.bin format.
        :return: <np.float32, 5, N>, the point cloud in .pcd.bin format.
        """
        lidar_fields = ['x', 'y', 'z', 'intensity', 'ring']
        return np.array([np.array(self.points[f], dtype=np.float32) for f in lidar_fields])

    def to_pcd_bin2(self):
        """
        Converts pointcloud to .pcd.bin2 format.
        :return: <np.float32, 6, N>, the point cloud in .pcd.bin2 format.
        """
        lidar_fields = ['x', 'y', 'z', 'intensity', 'ring', 'lidar_info']
        return np.array([np.array(self.points[f], dtype=np.float32) for f in lidar_fields])


@PIPELINES.register_module()
class LoadNuPlanPointsFromFile(object):
    """Load NuPlan points from file.
    """
    def __init__(self,
                 coord_type,):
        assert coord_type in ['LIDAR']
        self.coord_type = coord_type

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        pc = PointCloud.parse_from_file(pts_filename).to_pcd_bin2().T
        return pc

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_filename = results['pts_filename']
        points = self._load_points(pts_filename)

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=None)
        results['points'] = points

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


from .loading import LoadPointsFromMultiSweeps
@PIPELINES.register_module()
class LoadNuPlanPointsFromMultiSweeps(LoadPointsFromMultiSweeps):
    def __init__(self,
                 ego_mask=None,
                 hard_sweeps_timestamp=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.ego_mask = ego_mask

        # if hard_sweeps_timestamp:
        #  set timestamps of all points to {hard_sweeps_timestamp}.
        self.hard_sweeps_timestamp = hard_sweeps_timestamp

    def _load_points(self, pts_filename):
        pc = PointCloud.parse_from_file(pts_filename).to_pcd_bin2().T
        return pc

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
            points.tensor[:, -1] = self.hard_sweeps_timestamp
            results['points'] = points
        return results