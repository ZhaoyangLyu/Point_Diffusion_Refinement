import torch
import copy
from pointnet2_ops import pointnet2_utils

def mirror(partial, axis=1):
    # partial is of shape B,N,3
    partial_mirror = copy.deepcopy(partial)
    partial_mirror[:,:,axis] = -partial_mirror[:,:,axis]
    return partial_mirror

def down_sample_points(xyz, npoints):
    # xyz is of shape (B,N,4)
    # xyz = xyz.cuda()
    xyz_flipped = xyz.transpose(1, 2).contiguous() # shape (B,4,N)
    ori_xyz = xyz[:,:,0:3].contiguous()
    idx = pointnet2_utils.furthest_point_sample(ori_xyz, npoints)
    new_xyz = pointnet2_utils.gather_operation(xyz_flipped, idx) # shape (B,4,npoints)
    new_xyz = new_xyz.transpose(1, 2).contiguous() # shape (B,npoints, 4)
    return new_xyz
    
def mirror_and_concat(partial, axis=2, num_points=[2048, 3072]):
    B, N, _ = partial.size()
    partial_mirror = mirror(partial, axis=axis)

    device = partial.device
    dtype = partial.dtype
    partial = torch.cat([partial, torch.ones(B,N,1, device=device, dtype=dtype)], dim=2) # (B.N,4)
    partial_mirror = torch.cat([partial_mirror, torch.ones(B,N,1, device=device, dtype=dtype)*(-1)], dim=2) # (B.N,4)
    
    concat = torch.cat([partial, partial_mirror], dim=1) # (B,2N,4)
    concat = concat.cuda()
    down_sampled = [concat]
    for n in num_points:
        new_xyz = down_sample_points(concat, n)
        down_sampled.append(new_xyz)
    
    return tuple(down_sampled)

if __name__ == '__main__':
    import pdb
    B = 16
    N = 2048
    partial = torch.rand(B,N,3)
    down_sampled = mirror_and_concat(partial, axis=1, num_points=[2048, 3072])
    pdb.set_trace()