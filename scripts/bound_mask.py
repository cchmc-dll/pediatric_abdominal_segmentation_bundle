import os
import cv2
import numpy as np

def sum_filter_2d(mat, filt_size):
    # Check Dimensionality
    if mat.ndim != 2:
        raise ValueError('Sum Filter 2D expects a 2-dimensional matrix')
    
    # Ensure mat is type uint8
    mat = mat.astype(np.uint8)
    
    # Define sum filter kernel
    sum_filt = np.ones((filt_size, 1), np.float32)
    
    # Apply Separably (row first)
    sum_mat = cv2.filter2D(mat, -1, sum_filt, borderType=cv2.BORDER_CONSTANT)
    
    # Apply Separably (column)
    sum_mat = cv2.filter2D(sum_mat, -1, sum_filt.T, borderType=cv2.BORDER_CONSTANT)
    
    return sum_mat

def get_boundary_mask(mask):
    mask = mask.astype(np.uint8)
    # Check n dims
    if mask.ndim == 2:
        # Apply sum filter
        sum_mat = sum_filter_2d(mask, 3)
        
        # Get bound mask
        bound_mask = mask & (sum_mat < 9)
        
    elif mask.ndim == 3:
        # Get slices where mask exists
        idxs = np.any(mask, axis=(0, 1))
        idxs = np.where(idxs)[0]
        
        # Initialize bound mask
        bound_mask = np.zeros(mask.shape, dtype=bool)
        
        for slice_num in idxs:
            # Apply sum filter
            sum_mat = sum_filter_2d(mask[:, :, slice_num], 3)
            
            # Get boundary slice
            bound_mask[:, :, slice_num] = mask[:, :, slice_num] & (sum_mat < 9)
            
    else:
        raise ValueError('Get Boundary Mask only supports 2D and 3D masks')
        
    return bound_mask