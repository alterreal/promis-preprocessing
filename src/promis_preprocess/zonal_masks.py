import SimpleITK as sitk
from pathlib import Path
import numpy as np
import cc3d
from scipy.ndimage import binary_dilation

"""Utilities for working with zonal masks."""

def get_connected_components(image, connectivity = 26):
    """Counts and computes labels from connected components in a numpy array image"""

    # get connected componenets from image in numpy array format
    labels_out = cc3d.connected_components(image, connectivity=connectivity)

    # count connected components
    cc_count = len(np.unique(labels_out)) - 1

    if cc_count:
        return labels_out, cc_count
    else:
        return image, cc_count

def load_zone_predictions(filepath):
    """
    Loads the zonal masks for a given study.
    If input is a .mha file, reads the model-generated masks.
    If input is a .npz file, reads the model-generated probability maps.
    Returns the zonal masks or probability maps as SimpleITK images or arrays, respectively.
    """

    filepath = Path(filepath)

    if filepath.is_file() and filepath.suffix == '.mha':
        # read the model-generated probability maps
        zonal_mask = sitk.ReadImage(str(filepath))
        zonal_mask_arr = sitk.GetArrayFromImage(zonal_mask)

        # Create binary masks for TZ+CZ and PZ
        mask_tz_cz_arr = (zonal_mask_arr == 1).astype("uint8")
        mask_pz_arr = (zonal_mask_arr == 2).astype("uint8")

        # Convert back to images
        mask_tz_cz = sitk.GetImageFromArray(mask_tz_cz_arr)
        mask_pz = sitk.GetImageFromArray(mask_pz_arr)

        # Preserve spacing/origin/direction from original
        mask_tz_cz.CopyInformation(zonal_mask)
        mask_pz.CopyInformation(zonal_mask)

        return mask_tz_cz, mask_pz

    elif filepath.is_file() and filepath.suffix == '.npz':
        
        # read the model-generated probability maps
        prob_maps = np.load(filepath)

        # Fetch probability maps for TZ+CZ and PZ
        tz_cz_prob_arr = prob_maps['probabilities'][1]
        pz_prob_arr = prob_maps['probabilities'][2]

        # Convert to sitk images
        tz_cz_prob = sitk.GetImageFromArray(tz_cz_prob_arr)
        pz_prob = sitk.GetImageFromArray(pz_prob_arr)

        return tz_cz_prob, pz_prob

    else:
        raise FileNotFoundError(f"No zonal masks found for {filepath}")

def keep_components_in_contact(mask1, mask2, connectivity=26):
    """
    Keep only components from each mask that are in contact with components from the other mask.
    
    Parameters:
    -----------
    mask1 : np.ndarray
        First mask array (binary or multi-value)
    mask2 : np.ndarray
        Second mask array (binary or multi-value)
    connectivity : int, optional
        Connectivity for connected components (default: 26 for 3D)
        
    Returns:
    --------
    tuple of np.ndarray
        (mask1_filtered, mask2_filtered) where mask1_filtered contains only components 
        from mask1 that are in contact with mask2 components, and vice versa
    """
    # Convert to binary for connected component analysis
    mask1_binary = (mask1 > 0).astype(np.uint8)
    mask2_binary = (mask2 > 0).astype(np.uint8)
    
    # Get connected components for both masks
    labels1, cc_count1 = get_connected_components(mask1_binary, connectivity=connectivity)
    labels2, cc_count2 = get_connected_components(mask2_binary, connectivity=connectivity)
    
    # Initialize output masks
    mask1_result = np.zeros_like(mask1, dtype=mask1.dtype)
    mask2_result = np.zeros_like(mask2, dtype=mask2.dtype)
    
    # For mask1: keep only components that are in contact with mask2
    if cc_count1 > 0 and cc_count2 > 0:
        # Dilate mask2 to find contact points (26-connectivity structure)
        if connectivity == 26:
            structure = np.ones((3, 3, 3), dtype=bool)
        elif connectivity == 6:
            structure = np.zeros((3, 3, 3), dtype=bool)
            structure[1, 1, :] = True
            structure[1, :, 1] = True
            structure[:, 1, 1] = True
        else:
            # Default to 26-connectivity structure
            structure = np.ones((3, 3, 3), dtype=bool)
        
        mask2_dilated = binary_dilation(mask2_binary, structure=structure)
        
        # Find which components in mask1 are in contact with mask2
        unique_labels1 = np.unique(labels1[labels1 > 0])
        for label1 in unique_labels1:
            component1 = (labels1 == label1)
            # Check if this component overlaps with dilated mask2
            if np.any(component1 & mask2_dilated):
                # Keep this component, preserving original values
                mask1_result[component1] = mask1[component1]
        
        # For mask2: keep only components that are in contact with mask1
        mask1_dilated = binary_dilation(mask1_binary, structure=structure)
        unique_labels2 = np.unique(labels2[labels2 > 0])
        for label2 in unique_labels2:
            component2 = (labels2 == label2)
            # Check if this component overlaps with dilated mask1
            if np.any(component2 & mask1_dilated):
                # Keep this component, preserving original values
                mask2_result[component2] = mask2[component2]
    
    return mask1_result, mask2_result

def keep_components_in_mid_x_region(mask, margin_ratio=0.1, connectivity=26):
    """
    Keep only connected components that intersect the middle x region (with margin).
    
    Parameters:
    -----------
    mask : np.ndarray
        Input mask array (binary or multi-value), shape (D, H, W) where W is width (x-axis)
    margin_ratio : float, optional
        Margin as a ratio of image width (default: 0.1 = 10%)
    connectivity : int, optional
        Connectivity for connected components (default: 26 for 3D)
        
    Returns:
    --------
    np.ndarray
        Mask with only components that intersect the mid x region, preserving original dtype and values
    """
    # Convert to binary for connected component analysis
    mask_binary = (mask > 0).astype(np.uint8)
    labels, cc_count = get_connected_components(mask_binary, connectivity=connectivity)
    
    # Initialize output mask
    mask_result = np.zeros_like(mask, dtype=mask.dtype)
    
    if cc_count > 0:
        # Get image width (x-axis is the last dimension in 3D arrays: D, H, W)
        width = mask.shape[2]
        mid_x = width / 2.0
        margin = width * margin_ratio
        x_min = int(mid_x - margin)
        x_max = int(mid_x + margin)
        
        # Find which components intersect the mid x region
        unique_labels = np.unique(labels[labels > 0])
        for label in unique_labels:
            component = (labels == label)
            # Check if this component has any voxels in the mid x region
            # Component is shape (D, H, W), check x-axis (axis 2)
            component_slice = component[:, :, x_min:x_max+1]
            if np.any(component_slice):
                # Keep this component, preserving original values
                mask_result[component] = mask[component]
    
    return mask_result