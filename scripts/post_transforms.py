import numpy as np
import cv2
from monai.transforms import MapTransform, LabelToContour
from monai.config import KeysCollection
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from monai.data import MetaTensor

import os
import json

# Calculate segmentation volumes in ml
class CalculateVolumeFromMaskd(MapTransform):
    """
    Dictionary-based transform to calculate the volume of predicted organ masks.
    
    Args:
        keys (list): The keys corresponding to the predicted organ masks in the dictionary.
       label_names (list): The list of organ names corresponding to the masks.
    """
    def __init__(self, keys, label_names):
        super().__init__(keys)
        self.label_names = label_names
    
    def __call__(self, data):
        # Initialize a dictionary to store the volumes of each organ
        pred_volumes = {}

        for key in self.keys:
            for label_name in self.label_names.keys():
                #print('Key: ', key, ' organ_name: ', label_name)
                if label_name != 'background':
                    # Get the predicted mask from the dictionary
                    pred_mask = data[key]
                    # Calculate the voxel size in cubic millimeters (voxel size should be in the metadata)
                    # Assuming the metadata contains 'spatial_shape' with voxel dimensions in mm
                    if hasattr(pred_mask, 'affine'):
                        voxel_size = np.abs(np.linalg.det(pred_mask.affine[:3, :3]))
                    else:
                        raise ValueError("Affine transformation matrix with voxel spacing information is required.")

                    # Calculate the volume in cubic millimeters
                    label_volume_mm3 = np.sum(pred_mask == self.label_names[label_name]) * voxel_size

                    # Convert to milliliters (1 ml = 1000 mm^3)
                    label_volume_ml = label_volume_mm3 / 1000.0

                    # Store the result in the pred_volumes dictionary
                    pred_volumes[label_name] = round(label_volume_ml,0)
                    
                # Add the calculated volumes to the data dictionary
                key_name = key + '_volumes'
                
                data[key_name] = pred_volumes
            print('pred_volumes: ', pred_volumes)
        return data

class LabelToContourd(MapTransform):
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            label_image = d[key]
            assert isinstance(label_image, MetaTensor), "Input image must be a MetaTensor."

            # Initialize the contour image with the same shape as the label image
            contour_image = np.zeros_like(label_image.cpu().numpy())

            if label_image.ndim == 4:  # Check if the label image is 4D with a channel dimension
                # Process each 2D slice independently along the last axis (z-axis)
                for i in range(label_image.shape[-1]):
                    slice_image = label_image[:, :, :, i].cpu().numpy()

                    # Extract unique labels excluding background (assumed to be 0)
                    unique_labels = np.unique(slice_image)
                    unique_labels = unique_labels[unique_labels != 0]

                    slice_contour = np.zeros_like(slice_image)

                    # Generate contours for each label in the slice
                    for label in unique_labels:
                        # Create a binary mask for the current label
                        binary_mask = np.zeros_like(slice_image)
                        binary_mask[slice_image == label] = 1.0

                        # Apply LabelToContour to the 2D slice (replace this with actual contour logic)
                        thick_edges = LabelToContour()(binary_mask)

                        # Assign the label value to the contour image at the edge positions
                        slice_contour[thick_edges > 0] = label

                    # Stack the processed slice back into the 4D contour image
                    contour_image[:, :, :, i] = slice_contour
            else:
                # If the label image is not 4D, process it directly
                slice_image = label_image.cpu().numpy()
                unique_labels = np.unique(slice_image)
                unique_labels = unique_labels[unique_labels != 0]

                for label in unique_labels:
                    binary_mask = np.zeros_like(slice_image)
                    binary_mask[slice_image == label] = 1.0

                    thick_edges = LabelToContour()(binary_mask)
                    contour_image[thick_edges > 0] = label

            # Convert the contour image back to a MetaTensor with the original metadata
            contour_image_meta = MetaTensor(contour_image, meta=label_image.meta)#, affine=label_image.affine)
            
            # Store the contour MetaTensor in the output dictionary
            d[key] = contour_image_meta
            
        return d

# Using Alex's technique
# from .bound_mask import get_boundary_mask

# class LabelToContourd(MapTransform):
#     def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
#         super().__init__(keys, allow_missing_keys)

#     def __call__(self, data):
#         d = dict(data)
#         for key in self.keys:
#             label_image = d[key].cpu()
            
#             # Initialize the contour image with the same shape as the label image
#             contour_image = np.zeros_like(label_image)
            
#             # If the label image is not 4D, process it directly
#             unique_labels = np.unique(label_image)
#             unique_labels = unique_labels[unique_labels != 0]

#             for label in unique_labels:
#                 binary_mask = np.zeros_like(label_image)
#                 binary_mask[label_image == label] = 1.0

#                 thick_edges = get_boundary_mask(binary_mask[0])
#                 contour_image[0][thick_edges > 0] = label

#             # Store the contour image in the output dictionary
#             d[key] = MetaTensor(contour_image, meta=label_image.meta, affine=label_image.affine)
            
#         return d   

    
import numpy as np
from monai.transforms import MapTransform
from monai.config import KeysCollection
import matplotlib.cm as cm

class OverlayImageLabeld(MapTransform):
    def __init__(self, image_key: KeysCollection, label_key: str, overlay_key: str = "overlay", alpha: float = 0.7, allow_missing_keys: bool = False):
        super().__init__(image_key, allow_missing_keys)

        self.image_key = image_key
        self.label_key = label_key
        self.overlay_key = overlay_key
        self.alpha = alpha
        self.jet_colormap = cm.get_cmap('jet', 256)  # Get the Jet colormap with 256 discrete colors

    def apply_jet_colormap(self, label_volume):
        """
        Apply the Jet colormap to a 3D label volume using matplotlib's colormap.
        """
        assert label_volume.ndim == 3, "Label volume should have 3 dimensions (H, W, D) after removing channel."
        
        label_volume_normalized = (label_volume / label_volume.max()) * 255.0
        label_volume_uint8 = label_volume_normalized.astype(np.uint8)
        
        
        # Apply the colormap to each label
        label_rgb = self.jet_colormap(label_volume_uint8)[:, :, :, :3]  # Only take the RGB channels
    
        label_rgb = (label_rgb * 255).astype(np.uint8)
        # Rearrange axes to get (3, H, W, D)
        label_rgb = np.transpose(label_rgb, (3, 0, 1, 2))
        
        #assert label_rgb.shape == (*label_volume.shape, 3), f"Label RGB shape should be (H, W, D, 3) but got {label_rgb.shape}"
        assert label_rgb.shape == (3,*label_volume.shape), f"Label RGB shape should be (3,H, W, D) but got {label_rgb.shape}"
        
        return label_rgb

    def convert_to_rgb(self, image_volume):
        """
        Convert a single-channel grayscale 3D image to an RGB 3D image.
        """
        assert image_volume.ndim == 3, "Image volume should have 3 dimensions (H, W, D) after removing channel."
        
        image_volume_normalized = (image_volume - image_volume.min()) / (image_volume.max() - image_volume.min())
        #image_rgb = np.stack([image_volume_normalized] * 3, axis=-1)
        image_rgb = np.stack([image_volume_normalized] * 3, axis=0)
        image_rgb = (image_rgb * 255).astype(np.uint8)
        
        #assert image_rgb.shape == (*image_volume.shape, 3), f"Image RGB shape should be (H, W, D, 3) but got {image_rgb.shape}"
        assert image_rgb.shape == (3,*image_volume.shape), f"Image RGB shape should be (3,H, W, D) but got {image_rgb.shape}"
        
        return image_rgb

    def _create_overlay(self, image_volume, label_volume):
        # Convert the image volume and label volume to RGB
        image_rgb = self.convert_to_rgb(image_volume)
        label_rgb = self.apply_jet_colormap(label_volume)

        # Create an alpha-blended overlay
        overlay = image_rgb.copy()
        mask = label_volume > 0

        #overlay[mask] = (self.alpha * label_rgb[mask] + (1 - self.alpha) * overlay[mask]).astype(np.uint8)
        
         # Apply the overlay where the mask is present
        for i in range(3):  # For each color channel
            overlay[i, mask] = (self.alpha * label_rgb[i, mask] + (1 - self.alpha) * overlay[i, mask]).astype(np.uint8)
        
        assert overlay.shape == image_rgb.shape, f"Overlay shape should match image RGB shape: {overlay.shape} vs {image_rgb.shape}"
        
        return overlay

    def __call__(self, data):
        d = dict(data)
        
        # Get the image and label tensors
        image = d[self.image_key]  # Expecting shape (1, H, W, D)
        label = d[self.label_key]  # Expecting shape (1, H, W, D)

        # Ensure that the input has the correct dimensions
        assert image.shape[0] == 1 and label.shape[0] == 1, "Image and label must have a channel dimension of 1."
        assert image.shape == label.shape, f"Image and label must have the same shape: {image.shape} vs {label.shape}"

        # Remove the channel dimension for processing
        image_volume = image[0]  # Shape: (H, W, D)
        label_volume = label[0]  # Shape: (H, W, D)

        # Convert to 3D overlay
        overlay = self._create_overlay(image_volume, label_volume)

        # Add the channel dimension back
        #d[self.overlay_key] = np.expand_dims(overlay, axis=0)  # Shape: (1, H, W, D, 3)
        d[self.overlay_key] =  MetaTensor(overlay, meta=label.meta, affine=label.affine)  # Shape: (3, H, W, D)

        # Assert the final output shape
        # assert d[self.overlay_key].shape == (1, *image_volume.shape, 3), \
        #     f"Final overlay shape should be (1, H, W, D, 3) but got {d[self.overlay_key].shape}"
        
        assert d[self.overlay_key].shape == (3, *image_volume.shape), \
            f"Final overlay shape should be (3, H, W, D) but got {d[self.overlay_key].shape}"
         
        #print('overlay_image shape: ',  d[self.overlay_key].shape)
        return d



class SaveData(MapTransform):
    """
    Save the output dictionary into JSON files.

    The name of the saved file will be `{key}_{output_postfix}.json`.

    Args:
        keys: keys of the corresponding items to be saved in the dictionary.
        output_dir: directory to save the output files.
        output_postfix: a string appended to all output file names, default is `data`.
        separate_folder: whether to save each file in a separate folder. Default is `True`.
        print_log: whether to print logs when saving. Default is `True`.
    """

    def __init__(
        self,
        keys: KeysCollection,
        namekey: str = "image",
        output_dir: str = "./",
        output_postfix: str = "data",
        separate_folder: bool = False,
        print_log: bool = True,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.output_dir = output_dir
        self.output_postfix = output_postfix
        self.separate_folder = separate_folder
        self.print_log = print_log
        self.namekey = namekey

    def __call__(self, data):
        d = dict(data)
        image_name = os.path.basename(d[self.namekey].meta['filename_or_obj']).split('.')[0]
        for key in self.keys:
            # Get the data
            output_data = d[key]
            
            # Determine the file name
            file_name = f"{image_name}_{self.output_postfix}.json"
            if self.separate_folder:
                file_path = os.path.join(self.output_dir, image_name, file_name)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
            else:
                file_path = os.path.join(self.output_dir, file_name)

            # Save the dictionary as a JSON file
            with open(file_path, 'w') as f:
                json.dump(output_data, f)
            
            if self.print_log:
                print(f"Saved data to {file_path}")
            
        return d