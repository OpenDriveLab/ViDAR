# ViDAR: Visual Point Cloud Forecasting

![](./assets/teaser.png "Visual point cloud forecasting")

> **Visual Point Cloud Forecasting enables Scalable Autonomous Driving**
>
> [Zetong Yang](https://scholar.google.com/citations?user=oPiZSVYAAAAJ&hl=en), [Li Chen](https://scholar.google.com/citations?user=ulZxvY0AAAAJ&hl=en&authuser=1), [Yanan Sun](https://scholar.google.com/citations?user=6TA1oPkAAAAJ&hl=en), and [Hongyang Li](https://lihongyang.info/)
> 
> - Presented by [OpenDriveLab](https://opendrivelab.com/) and Shanghai AI Lab
> - :mailbox_with_mail: Primary contact: [Zetong Yang]((https://scholar.google.com/citations?user=oPiZSVYAAAAJ&hl=en)) ( tomztyang@gmail.com ) 
> - [arXiv paper](./assets/ViDAR.pdf) | [Blog TODO]() | [Slides TODO]()
> - [CVPR 2024 Autonomous Deiving Challenge - Predictive World Model](https://opendrivelab.com/AD24Challenge.html)


## Highlights <a name="highlights"></a>

:fire: **Visual point cloud forecasting**, a new self-supervised pre-training task for end-to-end autonomous driving, predicting 
future point clouds from historical visual inputs, joint modeling the 3D geometry and temporal dynamics for simultaneous perception, prediction, and planning.

:star2: **ViDAR**, the first visual point cloud forecasting architecture.

![method](./assets/vidar.png "Architecture of ViDAR")

:trophy: Predictive world model, with the form of visual point cloud forecasting, will be a main track in the `CVPR 2024 Autonomous Driving Challenge`. Please [stay tuned](https://opendrivelab.com/AD24Challenge.html) for further details!

## News <a name="news"></a>

- `[2023/12]` ViDAR [paper](./assets/ViDAR.pdf) released. *Code and models would be available around late January.*

## Table of Contents

1. [Highlights](#highlights)
2. [News](#news)
3. [Results and Model Zoo](#models)
4. [License and Citation](#license-and-citation)
5. [Related Resources](#resources)

## Results and Model Zoo <a name="models"></a>

|   Pre-trained Checkpoint  | Config | CD@1s | CD@2s | CD@3s |
| :------: | :---: | :----: | :----: | :----: | 
|   ViDAR-RN101   |  TODO  |  TODO   | TODO | TODO |


## License and Citation

All assets and code are under the [Apache 2.0 license](./LICENSE) unless specified otherwise.

If this work is helpful for your research, please consider citing the following BibTeX entry.

``` bibtex
@article{yang2023vidar,
  title={Visual Point Cloud Forecasting enables Scalable Autonomous Driving},
  author={Yang, Zetong and Chen, Li and Sun, Yanan and Li, Hongyang},
  journal={arXiv preprint arXiv:xx},
  year={2023}
}
```

## Related Resources <a name="resources"></a>

We acknowledge all the open-source contributors for the following projects to make this work possible:

- [BEVFormer](https://github.com/fundamentalvision/BEVFormer) | [UniAD](https://github.com/OpenDriveLab/UniAD) | [4D Occ](https://github.com/tarashakhurana/4d-occ-forecasting)

<a href="https://twitter.com/OpenDriveLab" target="_blank">
    <img alt="Twitter Follow" src="https://img.shields.io/twitter/follow/OpenDriveLab?style=social&color=brightgreen&logo=twitter" />
  </a>

- [DriveAGI](https://github.com/OpenDriveLab/DriveAGI) | [Survey on BEV Perception](https://github.com/OpenDriveLab/BEVPerception-Survey-Recipe) | [Survey on E2EAD](https://github.com/OpenDriveLab/End-to-end-Autonomous-Driving)
- [BEVFormer](https://github.com/fundamentalvision/BEVFormer) | [UniAD](https://github.com/OpenDriveLab/UniAD) | [OpenLane-V2](https://github.com/OpenDriveLab/OpenLane-V2) | [OccNet](https://github.com/OpenDriveLab/OccNet)
