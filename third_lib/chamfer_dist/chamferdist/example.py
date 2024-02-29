"""
Example usage
"""

import torch

from chamferdist import ChamferDistance


# Create two random pointclouds
# (Batchsize x Number of points x Number of dims)
source_cloud = torch.randn(1, 100, 3).cuda()
target_cloud = torch.randn(1, 50, 3).cuda()
source_cloud.requires_grad = True

# Initialize Chamfer distance module
chamferDist = ChamferDistance()
# Compute Chamfer distance
dist_forward = chamferDist(source_cloud, target_cloud)
print("Forward Chamfer distance:", dist_forward.detach().cpu().item())

# Chamfer distance depends on the direction in which it is computed (as the
# nearest neighbour varies, in each direction). One can either flip the order
# of the arguments, or simply use the "reverse" flag.
dist_backward = chamferDist(source_cloud, target_cloud, reverse=True)
print("Backward Chamfer distance:", dist_backward.detach().cpu().item())
# Or, if you rather prefer, flip the order of the arguments.
dist_backward = chamferDist(target_cloud, source_cloud)
print("Backward Chamfer distance:", dist_backward.detach().cpu().item())

# To get a symmetric measure, the simplest way is to average both the "forward"
# and "backward" distances. This is done by the "bidirectional" flag.
cdist = 0.5 * chamferDist(source_cloud, target_cloud, bidirectional=True)
cdist = 0.5 * chamferDist(target_cloud, source_cloud, bidirectional=True)
print("Bi-directional Chamfer distance:", cdist.detach().cpu().item())

# As a sanity check, chamfer distance between a pointcloud and itself must be
# zero.
dist_self = chamferDist(source_cloud, source_cloud)
print("Chamfer distance (self):", dist_self.detach().cpu().item())
dist_self = chamferDist(target_cloud, target_cloud)
print("Chamfer distance (self):", dist_self.detach().cpu().item())

# Backprop using this loss!
cdist.backward()
print(
    "Gradient norm wrt bidirectional Chamfer distance:",
    source_cloud.grad.norm().detach().cpu().item(),
)
