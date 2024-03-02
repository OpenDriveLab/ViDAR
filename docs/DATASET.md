## OpenScene

You can directly follow [HERE](https://github.com/OpenDriveLab/OpenScene/edit/main/docs/getting_started.md#download-data) for OpenScene data pre-processing, or
follow the step-by-step instructions below, which is the same as above link.

**Download and Unzip Data**

- Download and unzip all the sensor data from [nuPlan](https://www.nuscenes.org/nuplan). For ViDAR, we use both
the `Camera` sensor data and the `LiDAR` sensor data.
- Download and unzip our pre-processed `meta_data` at the following links:
  - mini-set: [openscene_metadata_mini.tgz](https://drive.google.com/file/d/1vGaTaXUQWEo9oZgJe_pUmKXNeCVAT8ME/view?usp=drive_link) (509.6 MB) 
  - trainval-set: [openscene_metadata_trainval.tgz](https://drive.google.com/file/d/1ce3LLQDpST-QzpV1ZVZcaMnjVkZnHXUq/view?usp=drive_link) (6.6 GB)
  - test-set: [openscene_metadata_test.tgz](https://drive.google.com/file/d/1hTQ56OqaNgljE3zD5qtte91uNE9qgSMk/view?usp=drive_link) (31.3 MB)
  - private-test-set: will be announced soon.

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
        |     └── test
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
              └── test
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
