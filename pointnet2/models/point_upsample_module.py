import torch
import numpy as np

def point_upsample(coarse, displacement, point_upsample_factor, include_displacement_center_to_final_output,
                    output_scale_factor_value):
    # coarse is of shape B,N,3
    # displacement is of shape (B,N,3*point_upsample_factor) or (B,N,3*(point_upsample_factor+1))
    grid_scale_factor = 1 / np.sqrt(point_upsample_factor)
    grid_displacement = displacement[:,:,3:] * grid_scale_factor
    center_displacement = displacement[:,:,0:3]
    intermediate_refined_X = coarse + center_displacement * output_scale_factor_value

    B,N,_ = coarse.size() # coarse should be of size B,2048,3
    # include_displacement_center_to_final_output = pointnet_config['include_displacement_center_to_final_output']
    if include_displacement_center_to_final_output:
        grid_displacement = grid_displacement.view(B, N, point_upsample_factor-1, 3)
    else:
        grid_displacement = grid_displacement.view(B, N, point_upsample_factor, 3)
        
    upsampled_X = intermediate_refined_X.unsqueeze(2) + grid_displacement * output_scale_factor_value
    # (B, N, point_upsample_factor-1, 3) or (B, N, point_upsample_factor, 3)
    upsampled_X = upsampled_X.reshape(B, -1, 3)
    if include_displacement_center_to_final_output:
        refined_X = torch.cat([upsampled_X, intermediate_refined_X], dim=1).contiguous()
    else:
        refined_X = upsampled_X.contiguous()
    # refined_X is of shape (B, N*point_upsample_factor, 3)
    return refined_X, intermediate_refined_X