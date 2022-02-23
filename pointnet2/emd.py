import torch
import emd_cuda
import torch.nn as nn


class EarthMoverDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2, return_match=False):
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        assert xyz1.is_cuda and xyz2.is_cuda, "Only support cuda currently."
        match = emd_cuda.approxmatch_forward(xyz1, xyz2)
        cost = emd_cuda.matchcost_forward(xyz1, xyz2, match)
        n = xyz1.shape[1]
        m = xyz2.shape[1]
        cost = cost / max(n,m)
        ctx.save_for_backward(xyz1, xyz2, match)
        if return_match:
            return cost, match
        else:
            return cost

    @staticmethod
    def backward(ctx, grad_cost):
        xyz1, xyz2, match = ctx.saved_tensors
        grad_cost = grad_cost.contiguous()
        grad_xyz1, grad_xyz2 = emd_cuda.matchcost_backward(grad_cost, xyz1, xyz2, match)
        return grad_xyz1, grad_xyz2, None


def earth_mover_distance(xyz1, xyz2, transpose=False, return_match=False):
    """Earth Mover Distance (Approx)

    Args:
        xyz1 (torch.Tensor): (b, n, 3)
        xyz2 (torch.Tensor): (b, m, 3)
        transpose (bool): whether to transpose inputs as it might be BCN format.
            Extensions only support BNC format.

    Returns:
        cost (torch.Tensor): (b)

    """
    if xyz1.dim() == 2:
        xyz1 = xyz1.unsqueeze(0)
    if xyz2.dim() == 2:
        xyz2 = xyz2.unsqueeze(0)
    if transpose:
        xyz1 = xyz1.transpose(1, 2)
        xyz2 = xyz2.transpose(1, 2)
    if return_match:
        cost, match = EarthMoverDistanceFunction.apply(xyz1, xyz2, True)
        return cost, match
    else:
        cost = EarthMoverDistanceFunction.apply(xyz1, xyz2, False)
        return cost

class EMD_distance(nn.Module):
    def forward(self, xyz1, xyz2, transpose=False, return_match=False):
        if xyz1.dim() == 2:
            xyz1 = xyz1.unsqueeze(0)
        if xyz2.dim() == 2:
            xyz2 = xyz2.unsqueeze(0)
        if transpose:
            xyz1 = xyz1.transpose(1, 2)
            xyz2 = xyz2.transpose(1, 2)
        if return_match:
            cost, match = EarthMoverDistanceFunction.apply(xyz1, xyz2, True)
            return cost, match
        else:
            cost = EarthMoverDistanceFunction.apply(xyz1, xyz2, False)
            return cost

if __name__ == '__main__':
    import pdb
    # from pytorch3d.loss.chamfer import chamfer_distance
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
    n = 2048
    m = 2048
    return_match = False
    x = torch.rand(128, n, 3).cuda() # (B,n,3)
    y = torch.rand(128, m, 3).cuda() # (B,m,3)
    d1 = earth_mover_distance(x,y, transpose=False, return_match=return_match)

    emd_module = EMD_distance()
    emd_module = nn.DataParallel(emd_module)
    d2 = emd_module(x,y, transpose=False, return_match=return_match)
    # c_d1,_ = chamfer_distance(x,y, batch_reduction=None)
    # c_d1 = c_d1/2
    # # in the completion 3d repo, they set groud truth point cloud as y, and the generated point cloud as x
    # # m1 is of shape (B,m,n) and 
    # # if m>n, m1.sum(dim=1) = ones(B,n)
    # # if m<n, m1.sum(dim=2) = ones(B,m)
    # # if m=n, m1.sum(dim=1) = m1.sum(dim=2) = ones(B,n)
    # # assume x has less points than y
    # # then for every point in x, we assign weights to every point to y, they sum to 1. 
    # # The closer a point in y is to x, the larger its weight is 

    # x_perm = torch.randperm(n)
    # x_new = x[:,x_perm,:]
    # y_perm = torch.randperm(m)
    # y_new = y[:,y_perm,:]
    # d1_new, m1_new = earth_mover_distance(x_new,y_new, transpose=False, return_match=return_match)
    # # permutation of the order of points in x or y doesn't change the distance much

    
    # d2, m2 = earth_mover_distance(y,x, transpose=False, return_match=return_match)
    # # change the order of x and y alter the distance largely
    # d3, m3 = earth_mover_distance(x,x, transpose=False, return_match=return_match)
    # # ds is indeed close to 0 
    pdb.set_trace()

