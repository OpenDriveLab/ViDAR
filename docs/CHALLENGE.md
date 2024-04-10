
<div id="top" align="center">

# Predictive World Model Challenge

**The tutorial of `Predictive World Model` track for [CVPR 2024 Autonomous Grand Challenge](https://opendrivelab.com/challenge2024).**

<p align="center">
  <img src="../assets/pred_banner.png" width="900px" >
</p>
</div>

> - Official website: :globe_with_meridians: [AGC2024](https://opendrivelab.com/challenge2024/#predictive_world_model)
> - Evaluation server: :hugs: [Hugging Face](https://huggingface.co/spaces/AGC2024-P/predictive-world-model-2024)
 
Serving as an abstract spatio-temporal representation of reality, the world model can predict future states based on the current state. The learning process of world models has the potential to provide a pre-trained foundation model for autonomous driving. Given vision-only inputs, the neural network outputs point clouds in the future to testify its predictive capability of the world.


## Table of Contents

1. [Problem Formulation](#problem-formulation)
2. [ViDAR-OpenScene-Baseline](#baseline)
3. [Evaluation: Chamfer Distance](#eval)
4. [Submission](#submission)
5. [Dataset: OpenScene](#dataset)
6. [License and Citation](#license-and-citation)
7. [Related Resources](#resources)


### Problem Formulation <a name="problem-formulation"></a>
Given a visual observation of the world for the past 3 seconds, predict the point clouds in the future 3 seconds based on the designated
future ego-vehicle pose. In other words,
given historical images in 3 seconds and corresponding history ego-vehicle pose information (from -2.5s to 0s, 6 frames under 2 Hz),
the participants are required to forecast future point clouds
in 3 seconds (from 0.5s to 3s, 6 frames under 2Hz) with specified future ego-poses.

All output point clouds should be aligned to the LiDAR coordinates of the ego-vehicle in the `n` timestamp, which spans a
range of 1 to 6 given predicting 6 future frames.

We then evaluate the predicted future point clouds by querying rays. We will provide a set of query rays (5k rays per scene) for testing propose,
and `the participants are required to estimate depth along each ray for rendering point clouds. An example of submission 
is provided.` Our evaluation toolkit will render
point clouds according to ray directions and provided depths by participants, and compute chamfer distance for points within
the range from -51.2m to 51.2m on the X- and Y-axis as the criterion.

### ViDAR-OpenScene-Baseline <a name="baseline"></a>

- Download and pre-process OpenScene dataset as illustrated at [HERE](./DATASET.md).
- Try the ViDAR model on OpenScene-mini subset:
  - OpenScene-mini-1/8-subset: [config](../projects/configs/vidar_pretrain/OpenScene/vidar_OpenScene_mini_1_8_3future.py)
  - OpenScene-mini-Full-set: [config](../projects/configs/vidar_pretrain/OpenScene/vidar_OpenScene_mini_full_3future.py)
```bash
CONFIG=/path/to/your/config
GPU_NUM=8

./tools/dist_train.sh ${CONFIG} ${GPU_NUM}
```
- To train ViDAR model on the entire OpenScene dataset:
  - OpenScene-Train-1/8-subset: [config](../projects/configs/vidar_pretrain/OpenScene/vidar_OpenScene_train_1_8_3future.py)
  - OpenScene-Train-Full-set: [config](../projects/configs/vidar_pretrain/OpenScene/vidar_OpenScene_train_full_3future.py)

To be finished in one week.

### Evaluation: Chamfer Distance <a name="eval"></a>
Chamfer Distance is used for measuring the similarity of two point sets, which represent shapes or outlines of two scenens.
It compares the similarity between predicted and ground-truth shapes by calculating the average nearest-neighbor distance between
points in one set to points in the other set, and vice versa.

For this challenge, we will compare chamfer distance between predicted point clouds and ground-truth point clouds for points
within the range of -51.2m to 51.2m. Participants are required to provide depths of specified ray directions. Our evaluation
system will render point clouds by ray directions and provided depth for chamfer distance evaluation.

### Submission <a name="submission"></a>

Download the [openscene_metadata_private_test_wm.pkl](https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_private_test_wm.tgz?download=true) (7.3 MB) and 
[sensor_data](https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_private_test_wm.tgz?download=true) (15 GB) for private test set, then prepare the
submission as the followings.

The submission should be in the following format:
```
dict {
    'method':                               <str> -- name of the method
    'team':                                 <str> -- name of the team, identical to the Google Form
    'authors':                              <list> -- list of str, authors
    'e-mail':                               <str> -- e-mail address
    'institution / company':                <str> -- institution or company
    'country / region':                     <str> -- country or region
    'results': {
        [identifier]: {                     <frame_token> -- identifier of the frame
            [frame_1]:                      <np.array> [n, 1] -- Predicted distance of each designated ray of 0.5s frame.
            [frame_2]:                      <np.array> [n, 1] -- Predicted distance of each designated ray of 1.0s frame.
            [frame_3]:                      <np.array> [n, 1] -- Predicted distance of each designated ray of 1.5s frame.
            [frame_4]:                      <np.array> [n, 1] -- Predicted distance of each designated ray of 2.0s frame.
            [frame_5]:                      <np.array> [n, 1] -- Predicted distance of each designated ray of 2.5s frame.
            [frame_6]:                      <np.array> [n, 1] -- Predicted distance of each designated ray of 3.0s frame.
        },
        [identifier]: {
        }
        ...
    }
}
```

You can also prepare your submission pickle following the following scripts. **Remember to update your information in [tools/convert_nuplan_submission_pkl.py](../tools/convert_nuplan_submission_pkl.py).**
We also provide an example [configuration](../projects/configs/vidar_pretrain/OpenScene/submit_vidar_OpenScene_mini_full_3future.py) for preparing submission.
```bash
CONFIG=path/to/vidar_config.py
CKPT=path/to/checkpoint.pth
GPU_NUM=8

# submission/root: path/to/your/submission
./tools/dist_test.sh ${CONFIG} ${CKPT} ${GPU_NUM} \
  --cfg-options 'model._submission=True' 'model._submission_path=submission/root'
  
# Convert submission to desired pickle file.
python tools/convert_nuplan_submission_pkl.py \
  submission/root \  # path to the generated submission .txt files.
  submission/dt.pkl  # path to the submitted pickle file.
```

As Hugging Face server will not return any detailed error if submission failed, please validate your
submission by our provided [script](../tools/validate_hf_submission.py) **before making a submission**:
```bash
# Validate the submission.
python tools/validate_hf_submission.py submission/dt.pkl
```

## Dataset: OpenScene <a name="dataset"></a>

<div id="top" align="center">
<p align="center">
  <img src="assets/OpenScene.gif" width="900px" >
</p>
</div>

> - [Medium Blog](https://medium.com/@opendrivelab/introducing-openscene-the-largest-benchmark-for-occupancy-prediction-in-autonomous-driving-74cfc5bbe7b6) | [Zhihu](https://zhuanlan.zhihu.com/p/647953862) (in Chinese)
> - Point of contact: [contact@opendrivelab.com](mailto:contact@opendrivelab.com)

### Description
OpenScene is the largest 3D occupancy prediction benchmark in autonomous driving. To highlight, 
we build it on top of [nuPlan](https://www.nuscenes.org/nuplan#challenge), covering a wide span of over 
**120 hours** of occupancy labels collected in various cities, from `Boston`, `Pittsburgh`, `Las Vegas` to `Singapore`.
The stats of the dataset is summarized [here](docs/dataset_stats.md).

<center>
  
|  Dataset  | Original Database |      Sensor Data (hr)    |   Flow | Semantic Category                               |
|:---------:|:-----------------:|:--------------------:|:------:|:--------------------------------------------:|
| [MonoScene](https://github.com/astra-vision/MonoScene)  |      NYUv2 / SemanticKITTI     | 5 / 6  |  :x:     | 10 / 19   |
| [Occ3D](https://github.com/Tsinghua-MARS-Lab/Occ3D)   |      nuScenes / Waymo    | 5.5 / 5.7 |  :x:    | 16 / 14 |
| [Occupancy-for-nuScenes](https://github.com/FANG-MING/occupancy-for-nuscenes)   |      nuScenes     | 5.5  |  :x:     | 16  |
| [SurroundOcc](https://github.com/weiyithu/SurroundOcc)   |      nuScenes     | 5.5  |   :x:    | 16  |
| [OpenOccupancy](https://github.com/JeffWang987/OpenOccupancy)   |      nuScenes     | 5.5  |  :x:     | 16  |
| [SSCBench](https://github.com/ai4ce/SSCBench)   |      KITTI-360 / nuScenes / Waymo     | 1.8 / 4.7 / 5.6  |   :x:     | 19 / 16 / 14  |
| [OccNet](https://github.com/OpenDriveLab/OccNet)   |      nuScenes     | 5.5  |   :x:     | 16   |
| **OpenScene** |       nuPlan      | **:boom: 120**  |   **:heavy_check_mark:**    | **`TODO`** |

</center>

> - The time span of LiDAR frames accumulated for each occupancy annotation is **20** seconds.
> - Flow: the annotation of motion direction and velocity for each occupancy grid.
> - `TODO`: Full semantic labels of grids would be released in future version

### Getting Started
- [Download Data](https://github.com/OpenDriveLab/OpenScene/blob/main/docs/getting_started.md#download-data)
- [Prepare Dataset](https://github.com/OpenDriveLab/OpenScene/blob/main/docs/getting_started.md#prepare-dataset)

## License and Citation <a name="license-and-citation"></a>
> Our dataset is based on the [nuPlan Dataset](https://www.nuscenes.org/nuplan) and therefore we distribute the data under [Creative Commons Attribution-NonCommercial-ShareAlike](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) license and [nuPlan Dataset License Agreement for Non-Commercial Use](https://www.nuscenes.org/terms-of-use). You are free to share and adapt the data, but have to give appropriate credit and may not use the work for commercial purposes.
All code within this repository is under [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

Please consider citing our paper if the project helps your research with the following BibTex:

```bibtex
@article{yang2023vidar,
  title={Visual Point Cloud Forecasting enables Scalable Autonomous Driving},
  author={Yang, Zetong and Chen, Li and Sun, Yanan and Li, Hongyang},
  journal={arXiv preprint arXiv:2312.17655},
  year={2023}
}

@misc{openscene2023,
  title={OpenScene: The Largest Up-to-Date 3D Occupancy Prediction Benchmark in Autonomous Driving},
  author={OpenScene Contributors},
  howpublished={\url{https://github.com/OpenDriveLab/OpenScene}},
  year={2023}
}

@article{sima2023_occnet,
  title={Scene as Occupancy}, 
  author={Chonghao Sima and Wenwen Tong and Tai Wang and Li Chen and Silei Wu and Hanming Deng  and Yi Gu and Lewei Lu and Ping Luo and Dahua Lin and Hongyang Li},
  journal={arXiv preprint arXiv:2306.02851},
  year={2023}
}
```

<p align="right">(<a href="#top">back to top</a>)</p>

## Related Resources  <a name="resources"></a>
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
- [DriveAGI](https://github.com/OpenDriveLab/DriveAGI)  | [OpenLane-V2](https://github.com/OpenDriveLab/OpenLane-V2) | [DriveLM](https://github.com/OpenDriveLab/DriveLM)
- [Survey on Bird's-eye-view Perception](https://github.com/OpenDriveLab/BEVPerception-Survey-Recipe) | [BEVFormer](https://github.com/fundamentalvision/BEVFormer) |  [OccNet](https://github.com/OpenDriveLab/OccNet)


<p align="right">(<a href="#top">back to top</a>)</p>










