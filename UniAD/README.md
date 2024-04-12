# ViDAR-UniAD Fine-tuning

This repo contains the code and configuration files for ViDAR fine-tuning on UniAD for end-to-end autonomous driving.

## Results and Models

### Stage1: Perception training

| Downstream Model | Dataset |  pre-train | Config | Detection<br>NDS | Tracking<br>AMOTA | Mapping<br>IoU-lane | models & logs |
| :------: | :------: | :---: | :---: | :----: | :----: | :----: | :----: |
| UniAD-Stage1 (baseline) | nuScenes (100% Data) |  BEVFormer-base: [cfg](../projects/configs/bevformer/bevformer_base.py) / [model](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_r101_dcn_24ep.pth)  | [base_track_map.py](./projects/configs/stage1_track_map/base_track_map.py)  |  46.69   | 36.0 | 28.7 | - |
| ViDAR-UniAD-Stage1 | nuScenes (100% Data) |   ViDAR-BEVFormer: [cfg](../projects/configs/vidar_finetune/nusc_fullset/vidar_full_nusc_1future.py) / [model](https://github.com/OpenDriveLab/ViDAR/releases/download/v1.0.0/finetune-ViDAR-RN101-nus-full-1future.pth)   | [vidar_track_map.py](./projects/configs/stage1_track_map/vidar_track_map.py)  |  54.99 |  45.6   | 33.8 | [models](https://github.com/OpenDriveLab/ViDAR/releases/download/v1.0.0/UniAD-s1-ViDAR-RN101-nus-full-1future.pth) / [logs](https://github.com/OpenDriveLab/ViDAR/releases/download/v1.0.0/UniAD-s1-ViDAR-RN101-nus-full-1future.log) |

### Stage2: End-to-end training

| Downstream Model | Dataset |  pre-train | Config | Detection<br>NDS | Tracking<br>AMOTA | Mapping<br>IoU-lane | Motion<br>minADE |Occupancy<br>IoU-n. | Planning<br>avg.Col. | models & logs |
| :------: | :------: | :---: | :---: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | 
| UniAD-Stage2 (baseline) | nuScenes (100% Data) |  UniAD-Stage1: [cfg](./projects/configs/stage1_track_map/base_track_map.py)  | [base_e2e.py](./projects/configs/stage2_e2e/base_e2e.py)  |  49.36   | 38.3 | 31.3 | 0.75 | 62.8 | 0.27 | -|
| ViDAR-UniAD-Stage2 | nuScenes (100% Data) |   ViDAR-UniAD-Stage1: [cfg](./projects/configs/stage1_track_map/vidar_track_map.py) / [model](https://github.com/OpenDriveLab/ViDAR/releases/download/v1.0.0/UniAD-s1-ViDAR-RN101-nus-full-1future.pth)   | [vidar_e2e.py](./projects/configs/stage2_e2e/vidar_e2e.py)  |  54.06 | 43.5 | 35.2 | 0.65 |   65.7  | 0.18 | [models](https://github.com/OpenDriveLab/ViDAR/releases/download/v1.0.0/UniAD-s2-ViDAR-RN101-nus-full-1future.pth) / [logs](https://github.com/OpenDriveLab/ViDAR/releases/download/v1.0.0/UniAD-s2-ViDAR-RN101-nus-full-1future.log) |

## Getting Started

### Installation

- First, refer to [Installation](../README.md#installation-a-nameinstallationa) to install ViDAR first.
- Second, run `pip install -r requirements.txt` to install extra dependencies.

### Data preprocessing
Please refer to [Dataset](./docs/DATA_PREP.md) for data preparation before the first run.

### Training Command
```shell
# stage-1
CONFIG=./projects/configs/stage1_track_map/vidar_track_map.py
GPU_NUM=8
export PYTHONPATH=/PATH/TO/ViDAR/projects/mmdet3d_plugin/bevformer/:${PYTHONPATH}
./tools/uniad_dist_train.sh ${CONFIG} ${GPU_NUM}

# stage-2
CONFIG=./projects/configs/stage2_e2e/vidar_e2e.py
GPU_NUM=16
export PYTHONPATH=/PATH/TO/ViDAR/projects/mmdet3d_plugin/bevformer/:${PYTHONPATH}
./tools/uniad_dist_train.sh ${CONFIG} ${GPU_NUM}
```

### Eval Command
```shell
CONFIG=path/to/uniad_config.py
CKPT=path/to/checkpoint.pth
GPU_NUM=8

./tools/uniad_dist_eval.sh ${CONFIG} ${CKPT} ${GPU_NUM}
```


## Related Citations
```bibtex
@inproceedings{yang2023vidar,
    title={Visual Point Cloud Forecasting enables Scalable Autonomous Driving},
    author={Yang, Zetong and Chen, Li and Sun, Yanan and Li, Hongyang},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2024}
}

@inproceedings{hu2023_uniad,
    title={Planning-oriented Autonomous Driving}, 
    author={Yihan Hu and Jiazhi Yang and Li Chen and Keyu Li and Chonghao Sima and Xizhou Zhu and Siqi Chai and Senyao Du and Tianwei Lin and Wenhai Wang and Lewei Lu and Xiaosong Jia and Qiang Liu and Jifeng Dai and Yu Qiao and Hongyang Li},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2023}
}
```