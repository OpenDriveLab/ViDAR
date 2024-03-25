import mmcv
import os, sys
import tqdm

from projects.mmdet3d_plugin.datasets.pipelines.nuplan_loading import PointCloud

dt_path = sys.argv[1]
meta_file = 'data/openscene-v1.1/openscene_metadata_private_test_wm.pkl'
sensor_blob_root = 'data/openscene-v1.1/sensor_blobs/private_test_wm/'
meta_info = mmcv.load(meta_file)

current_frame_idx = 5  # -2.5s
total_frame_num = 12  # 12 frames for each scene
frame_prefix = 'frame'
frame_names = [f'{frame_prefix}_0',
               f'{frame_prefix}_1',
               f'{frame_prefix}_2',
               f'{frame_prefix}_3',
               f'{frame_prefix}_4',
               f'{frame_prefix}_5',
               f'{frame_prefix}_6',]

token_raynum_dict = dict()

cur_idx = -1
scene_token = meta_info[0]['scene_token']
cur_token = None
print(f'==== Extracting Ground-truth Informations from MetaINFO: {meta_file} ====')
for info in tqdm.tqdm(meta_info):
    cur_idx += 1

    # assert each <total_frame_num> comprises a scene.
    if cur_idx < total_frame_num:
        assert scene_token == info['scene_token']
    else:
        cur_idx = 0
        scene_token = info['scene_token']

    if cur_idx == current_frame_idx:
        cur_token = info['token']
        token_raynum_dict[cur_token] = dict()

    if cur_idx > current_frame_idx:
        # Future frames.
        pcd_path = os.path.join(sensor_blob_root, info['lidar_path'])
        pcd = PointCloud.parse_from_file(pcd_path).to_pcd_bin2().T
        token_raynum_dict[cur_token][frame_names[cur_idx - current_frame_idx]] = pcd.shape[0]

print(f'==== Validating the Submission FIle: {dt_path} ====')
dt = mmcv.load(dt_path)
required_keys = ['method', 'team', 'authors', 'email', 'institution', 'country', 'results']
for k in required_keys:
    assert k in dt, f'<{k}> is required in the submission file.'
dt_res = dt['results']
for token, results in tqdm.tqdm(token_raynum_dict.items()):
    assert token in dt_res, f'{token} is not included in the submission.'
    for i in range(1, 7):
        assert frame_names[i] in dt_res[token], \
            f'Results of <{frame_names[i]}> is missed in the scene {token}!'
        assert dt_res[token][frame_names[i]].ndim == 2 and dt_res[token][frame_names[i]].shape[-1] == 1, \
            f'Submission results must be in a shape of <N, 1>,' \
            f' but get {dt_res[token][frame_names[i]].shape} instead,' \
            f' for results of the {frame_names[i]} in scene {token}.'
        assert dt_res[token][frame_names[i]].shape[0] == token_raynum_dict[token][frame_names[i]], \
            f'The submitted results should have {token_raynum_dict[token][frame_names[i]]} depths' \
            f' corresponding to each provided ray in the METAFILE, but get only' \
            f' {dt_res[token][frame_names[i]].shape[0]} instead,' \
            f' for results of the {frame_names[i]} in scene {token}.'
print('==== Validation Succeed! ====')
