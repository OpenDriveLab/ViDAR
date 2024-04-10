## OpenScene

You can directly follow [HERE](https://github.com/OpenDriveLab/OpenScene/blob/main/docs/getting_started.md#download-data) for OpenScene data pre-processing, or
follow the step-by-step instructions below, which is the same as above link.

**Download and Unzip Data**

- Download and unzip all the sensor data from [nuPlan](https://www.nuscenes.org/nuplan). For ViDAR, we use both
the `Camera` sensor data and the `LiDAR` sensor data. We also provide the links below:
  - `mini-set` openscene_sensor_mini_camera (84 GB): [OpenDriveLab](https://openxlab.org.cn/datasets/OpenDriveLab/OpenScene/tree/main/openscene-v1.1) / [Hugging Face](https://huggingface.co/datasets/OpenDriveLab/OpenScene/tree/main/openscene-v1.1/openscene_sensor_mini_camera)
  - `mini-set` openscene_sensor_mini_lidar (60 GB): [OpenDriveLab](https://openxlab.org.cn/datasets/OpenDriveLab/OpenScene/tree/main/openscene-v1.1) / [Hugging Face](https://huggingface.co/datasets/OpenDriveLab/OpenScene/tree/main/openscene-v1.1/openscene_sensor_mini_lidar)
  - `trainval-set` openscene_sensor_trainval_camera (1.1 TB): [OpenDriveLab](https://openxlab.org.cn/datasets/OpenDriveLab/OpenScene/tree/main/openscene-v1.1)
  - `trainval-set` openscene_sensor_trainval_lidar (822 GB): [OpenDriveLab](https://openxlab.org.cn/datasets/OpenDriveLab/OpenScene/tree/main/openscene-v1.1)
  - `test-set` openscene_sensor_test_camera (120 GB): [OpenDriveLab](https://openxlab.org.cn/datasets/OpenDriveLab/OpenScene/tree/main/openscene-v1.1) / [Hugging Face](https://huggingface.co/datasets/OpenDriveLab/OpenScene/tree/main/openscene-v1.1/openscene_sensor_test_camera)
  - `test-set` openscene_sensor_test_lidar (87 GB): [OpenDriveLab](https://openxlab.org.cn/datasets/OpenDriveLab/OpenScene/tree/main/openscene-v1.1) / [Hugging Face](https://huggingface.co/datasets/OpenDriveLab/OpenScene/tree/main/openscene-v1.1/openscene_sensor_test_lidar)
  - `private-test` openscene_sensor_private_test_wm (15 GB): [Hugging Face](https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_private_test_wm.tgz?download=true)

- Download and unzip our pre-processed `meta_data` at the following links:
  - `mini-set` openscene_metadata_mini.tgz (509.6 MB): [Google Drive](https://drive.google.com/drive/folders/1MnRwhnEBsgZxbaleHxc3Gw7Ovc4I9az1?usp=sharing) / [Hugging Face](https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_mini.tgz?download=true) 
  - `trainval-set` openscene_metadata_trainval.tgz (6.6 GB): [Google Drive](https://drive.google.com/drive/folders/1MnRwhnEBsgZxbaleHxc3Gw7Ovc4I9az1?usp=sharing) / [Hugging Face](https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_trainval.tgz?download=true)
  - `test-set` openscene_metadata_test.tgz (454 MB): [Google Drive](https://drive.google.com/drive/folders/1MnRwhnEBsgZxbaleHxc3Gw7Ovc4I9az1?usp=sharing) / [Hugging Face](https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_test.tgz?download=true)
  - `private-test-set` openscene_metadata_private_test_wm.tgz (7.3 MB): [Hugging Face](https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_private_test_wm.tgz?download=true)

After downloading and unzipping all the data, soft linking those source data files to the `ViDAR/data` folder as the following structure.

```
ViDAR
├── projects/
├── tools/
├── pretrained/
│   └── r101_dcn_fcos3d_pretrain.pth
└── data/
    └── openscene-v1.1
        ├── meta_datas
        |     ├── mini
        │     │     ├── 2021.05.12.22.00.38_veh-35_01008_01518.pkl
        │     │     ├── 2021.05.12.22.28.35_veh-35_00620_01164.pkl
        │     │     ├── ...
        │     │     └── 2021.10.11.08.31.07_veh-50_01750_01948.pkl
        |     ├── trainval
        |     ├── test
        |     └── private_test_wm
        │          └── private_test_wm.pkl
        |
        └── sensor_blobs
              ├── mini
              │    ├── 2021.05.12.22.00.38_veh-35_01008_01518                                           
              │    │    ├── CAM_F0
              │    │    │     ├── c082c104b7ac5a71.jpg
              │    │    │     ├── af380db4b4ca5d63.jpg
              │    │    │     ├── ...
              │    │    │     └── 2270fccfb44858b3.jpg
              │    │    ├── CAM_B0
              │    │    ├── CAM_L0
              │    │    ├── CAM_L1
              │    │    ├── CAM_L2
              │    │    ├── CAM_R0
              │    │    ├── CAM_R1
              │    │    ├── CAM_R2
              │    │    └── MergedPointCloud
              │    │            ├── 0079e06969ed5625.pcd
              │    │            ├── 01817973fa0957d5.pcd
              │    │            ├── ...
              │    │            └── fffb7c8e89cd54a5.pcd       
              │    ├── 2021.06.09.17.23.18_veh-38_00773_01140 
              │    ├── ...                                                                            
              │    └── 2021.10.11.08.31.07_veh-50_01750_01948
              ├── trainval
              ├── test
              └── private_test_wm
```

**Prepare OpenScene data**
```
# mini / trainval / test
python tools/collect_nuplan_data.py mini
python tools/collect_nuplan_data.py trainval
python tools/collect_nuplan_data.py test
```


## NuScenes <a name="nuscenes"></a>
Download nuScenes V1.0 full dataset data  and CAN bus expansion data [HERE](https://www.nuscenes.org/download).
Prepare nuscenes data by running the following steps.

**Download CAN bus expansion**
```
# download 'can_bus.zip'
unzip can_bus.zip 
# move can_bus to data dir
```

**Prepare nuScenes data**
```
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0 --canbus ./data
```

Using the above code will generate `nuscenes_infos_temporal_{train,val}.pkl`.

**Folder structure**
```
ViDAR
├── projects/
├── tools/
├── pretrained/
│   ├── r101_dcn_fcos3d_pretrain.pth
├── data/
│   ├── can_bus/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
|   |   ├── v1.0-trainval/
|   |   ├── nuscenes_infos_temporal_train.pkl
|   |   ├── nuscenes_infos_temporal_val.pkl
```
