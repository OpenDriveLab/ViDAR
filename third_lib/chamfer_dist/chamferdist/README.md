# chamferdist: PyTorch Chamfer distance

> **NOTE**: This implementation was stolen from the [pytorch3d](https://github.com/facebookresearch/pytorch3d) repo, and all I did was to simply repackage it.

[![krrish94](https://circleci.com/gh/krrish94/chamferdist.svg?style=svg)](<https://app.circleci.com/pipelines/github/krrish94/chamferdist>)

A simple example Pytorch module to compute Chamfer distance between two pointclouds.

### Installation

You can install the package using `pip`.

```
pip install chamferdist
```

### Building from source

In your favourite python/conda virtual environment, execute the following commands. 

> **NOTE**: This assumes you have PyTorch installed already (preferably, >= 1.5.0; untested for earlier releases).

```python
python setup.py install
```

### Running (example)

That's it! You're now ready to go. Here's a quick guide to using the package. Fire up a terminal. Import the package.

```python
>>> import torch
>>> from chamferdist import ChamferDistance
```

Create two random pointclouds. Each pointcloud is a **3D tensor** with dimensions `batchsize` x `number of points` x `number of dimensions`.

```python
>>> source_cloud = torch.randn(1, 100, 3).cuda()
>>> target_cloud = torch.randn(1, 50, 3).cuda()
```

Initialize a `ChamferDistance` object.
```python
>>> chamferDist = ChamferDistance()
```

Now, compute Chamfer distance.
```python
>>> dist_forward = chamferDist(source_cloud, target_cloud)
>>> print(dist_forward.detach().cpu().item())
```

Here, `dist` is the Chamfer distance between `source_cloud` and `target_cloud`. Note that Chamfer distance is not bidirectional (and, in stricter parlance, it is not a _distance metric_).

The Chamfer distance in the backward direction, i.e., `target_cloud` to `source_cloud` can be computed in two ways. The naive way is to simply flip the order of the arguments, i.e.,
```python
>>> dist_backward = chamferDist(target_cloud, source_cloud)
```
Another way is to use the `reverse` flag provided by the `ChamferDistance` module, i.e.,
```python
>>> dist_backward = chamferDist(source_cloud, target_cloud, reverse=True)
>>> print(dist_backward.detach().cpu().item())
```

Typically, a symmetric version of the Chamfer distance is obtained, by summing the "forward" and the "backward" Chamfer distances. This is supported by the `bidirectional` flag.
```python
>>> dist_bidirectional = chamferDist(source_cloud, target_cloud, bidirectional=True)
>>> print(dist_bidirectional.detach().cpu().item())
```

Look at the example script for more details: [example.py](example.py)


### Citing (the original implementation, PyTorch3D)

If you find this work useful, you might want to cite the *original* implementation from which this codebase was borrowed (stolen!) - PyTorch3D.

```
@article{ravi2020pytorch3d,
    author = {Nikhila Ravi and Jeremy Reizenstein and David Novotny and Taylor Gordon
                  and Wan-Yen Lo and Justin Johnson and Georgia Gkioxari},
    title = {Accelerating 3D Deep Learning with PyTorch3D},
    journal = {arXiv:2007.08501},
    year = {2020},
}
```
