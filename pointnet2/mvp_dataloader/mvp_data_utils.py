import h5py
import numpy as np
import pandas as pd
import transforms3d
import random
import math

def augment_cloud(Ps, args, return_augmentation_params=False):
    """" Augmentation on XYZ and jittering of everything """
    # Ps is a list of point clouds

    M = transforms3d.zooms.zfdir2mat(1) # M is 3*3 identity matrix
    # scale
    if args['pc_augm_scale'] > 1:
        s = random.uniform(1/args['pc_augm_scale'], args['pc_augm_scale'])
        M = np.dot(transforms3d.zooms.zfdir2mat(s), M)

    # rotation
    if args['pc_augm_rot']:
        scale = args['pc_rot_scale'] # we assume the scale is given in degrees
        # should range from 0 to 180
        if scale > 0:
            angle = random.uniform(-math.pi, math.pi) * scale / 180.0
            M = np.dot(transforms3d.axangles.axangle2mat([0,1,0], angle), M) # y=upright assumption
            # we have verified that shapes from mvp data, the upright direction is along the y axis positive direction
    
    # mirror
    if args['pc_augm_mirror_prob'] > 0: # mirroring x&z, not y
        if random.random() < args['pc_augm_mirror_prob']/2:
            M = np.dot(transforms3d.zooms.zfdir2mat(-1, [1,0,0]), M)
        if random.random() < args['pc_augm_mirror_prob']/2:
            M = np.dot(transforms3d.zooms.zfdir2mat(-1, [0,0,1]), M)

    # translation
    translation_sigma = args.get('translation_magnitude', 0)
    translation_sigma = max(args['pc_augm_scale'], 1) * translation_sigma
    if translation_sigma > 0:
        noise = np.random.normal(scale=translation_sigma, size=(1, 3))
        noise = noise.astype(Ps[0].dtype)
        
    result = []
    for P in Ps:
        P[:,:3] = np.dot(P[:,:3], M.T)
        if translation_sigma > 0:
            P[:,:3] = P[:,:3] + noise
        if args['pc_augm_jitter']:
            sigma, clip= 0.01, 0.05 # https://github.com/charlesq34/pointnet/blob/master/provider.py#L74
            P = P + np.clip(sigma * np.random.randn(*P.shape), -1*clip, clip).astype(np.float32)
        result.append(P)

    if return_augmentation_params:
        augmentation_params = {}
        augmentation_params['M_inv'] = np.linalg.inv(M.T).astype(Ps[0].dtype)
        if translation_sigma > 0:
            augmentation_params['translation'] = noise
        else:
            augmentation_params['translation'] = np.zeros((1, 3)).astype(Ps[0].dtype)
        
        return result, augmentation_params

    return result

if __name__ == '__main__':
    import pdb
    args = {'pc_augm_scale':0, 'pc_augm_rot':False, 'pc_rot_scale':30.0, 'pc_augm_mirror_prob':0.5, 'pc_augm_jitter':False}
    N = 2048
    C = 6
    num_of_clouds = 2
    pc = []
    for _ in range(num_of_clouds):
        pc.append(np.random.rand(N,C)-0.5)
    
    result = augment_cloud(pc, args)
    pdb.set_trace()


