import numpy as np
import pandas as pd
import cc3d
from scipy.ndimage import distance_transform_edt
from skimage.morphology import remove_small_holes

def get_mask_centroid_3d(mask):
    """Returns a tuple containing the centroid of a 
    3D segmentation mask (z,y,x) coordinates"""
    
    mask_indices = np.argwhere(mask > 0)
    centroid = np.mean(mask_indices, axis=0).astype(int)
    
    return centroid  

def compute_centroid_distance(mask1, mask2):
    """Computes the centroid Euclidean distance between two 3D masks in numpy format""" 
    mask1_centroid = get_mask_centroid_3d(mask1)
    mask2_centroid = get_mask_centroid_3d(mask2)
    distance = np.linalg.norm(mask1_centroid - mask2_centroid)

    return distance

def get_sector_mask(mask_prostate, mask_TZ_CZ, mask_PZ, l_region, l_side, l_zone, l_section, region_div_method = 'z_equal', z_custom_cutoffs = [0.33, 0.66]):
    """Computes a segmentation mask for a single prostate sector"""

    # Create a sector mask 
    sector_region = np.zeros(mask_prostate.shape).astype(np.uint8)
    sector_side = np.zeros(mask_prostate.shape).astype(np.uint8)
    sector_section  = np.zeros(mask_prostate.shape).astype(np.uint8)

    # Calculate the centroid of the prostate segmentation mask
    prostate_centroid = get_mask_centroid_3d(mask_prostate)
    C_x, C_y =  prostate_centroid[2], prostate_centroid[1]

    # prostate segmentation mask z coordinates
    z_prostate_coor = np.argwhere(np.any(mask_prostate, axis = (1, 2))).astype(np.uint8)

    # split prostate mask z coordinates into three parts, according to method.
    if region_div_method == 'z_equal':
        # Assume apex, mid and base regions (in this order) fill equal parts along the z-axis
        regions_coor =  np.array_split(z_prostate_coor, 3)
    elif region_div_method == 'barzell':
        regions_coor =  np.array_split(z_prostate_coor, 2)
    elif region_div_method == 'volume':
        # considers each region must have the same volume (prostate_volume/3)"
        regions_coor = get_regions_volume(mask_prostate)
    elif region_div_method == 'z_custom':

        # get prostate length along z axis
        p_z_len = len(z_prostate_coor)

        # get proportion of the prostate length corresponding to the boundary between:
        # - am: apex and mid
        # - bm: mid and base
        # note: (in our data, median prostate length is 13 slices)
        am_perc = z_custom_cutoffs[0]
        mb_perc = z_custom_cutoffs[1]

        # get the corresponding z indices
        am_index = round(p_z_len*am_perc)
        mb_index = round(p_z_len*mb_perc)

        # split z coordinates accordingly
        regions_coor = np.split(z_prostate_coor, [am_index, mb_index])     
    else:
        print('Please select a valid method to divide prostate into regions along the z axis')


    # get mask for sector region
    if l_region == 'A':
        sector_region[regions_coor[0], :, :] = 1
    elif l_region == 'M':
        sector_region[regions_coor[1], :, :] = 1
    elif l_region == 'B':
        sector_region[regions_coor[-1], :, :] = 1
    else:
        # if region is not specified, assume lesion spreads across all regions
        sector_region[:, :, :] = 1

    # Get mask for side
    if l_side == 'R':
        sector_side[:, :, :C_x] = 1
    elif l_side == 'L':
        sector_side[:, :, C_x:] = 1
    else:
        # if side is not specified, assume lesion spreads across all sides
        sector_side[:, :, :] = 1

    # get mask for section
    if l_section == 'a':
        sector_section[:, :C_y, :] = 1
    elif l_section == 'p':
        sector_section[:, C_y:, :] = 1
    else:
        # if section is not specified, assume lesion spreads across all sections
        sector_section[:, :, :] = 1  

    # get mask for zone
    if l_zone == 'TZ+CZ':
        sector_zone = mask_TZ_CZ
    elif l_zone == 'PZ':
        sector_zone = mask_PZ
    else:
        # if zone is not specified, assume lesion spreads across all zones
        sector_zone = mask_prostate

    # compute sector mask
    sector_mask = sector_region * sector_side * sector_zone * sector_section

    return sector_mask 

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
    
def get_reported_lesion_mask(mask_prostate, mask_TZ_CZ, mask_PZ, sector_label_list, region_div_method = 'z_equal', z_custom_cutoffs = [0.33, 0.66]):
    """Iterates through a list of sectors that belong to the same lesion
    Returns a location-guided lesion mask"""

    # initialize mask
    mask = np.zeros(mask_prostate.shape).astype(np.uint8)

    for labeled_sector in sector_label_list:
        sector = get_sector_mask(
                            mask_prostate, 
                            mask_TZ_CZ, 
                            mask_PZ, 
                            labeled_sector['l_region'], 
                            labeled_sector['l_side'], 
                            labeled_sector['l_zone'],
                            labeled_sector['l_section'],
                            region_div_method,
                            z_custom_cutoffs
                            )

        # combine sector masks to cover entire reported lesion location
        mask = ((mask > 0) | (sector > 0)).astype(np.uint8)

    return mask

def assign_added_pixels(added_pixels, mask1, mask2):
    """Assigns pixels in the binary mask "added_pixels" to two masks, according to nearest neighbour
    Returns the two masks with added pixels"""

    assigned_mask1 = np.copy(mask1)
    assigned_mask2 = np.copy(mask2)

    #iterate through z axis
    for z in range(added_pixels.shape[0]):
        diff = added_pixels[z] > 0  # pixels added by remove_small_holes

        if not np.any(diff):
            continue  # Skip empty slices

        # Distance transform: distance to nearest pixel in mask1/mask2
        dist1 = distance_transform_edt(~(mask1[z] > 0))
        dist2 = distance_transform_edt(~(mask2[z] > 0))

        # Compare distances and assign
        assign_to_mask1 = (dist1 < dist2) & diff
        assign_to_mask2 = (dist2 <= dist1) & diff

        assigned_mask1[z][assign_to_mask1] = 1
        assigned_mask2[z][assign_to_mask2] = 1

    return assigned_mask1, assigned_mask2

def get_regions_volume(mask_prostate):
    """Divides prostate mask into apex, mid and base along the z axis,
    considering each region must have the same volume (prostate_volume/3)"""

    # Get the foreground slices and their voxel counts
    z_slices = np.argwhere(np.any(mask_prostate, axis=(1, 2))).flatten()  # shape: (num_slices,)
    slice_volumes = np.array([np.count_nonzero(mask_prostate[z]) for z in z_slices])

    # Total prostate volume
    total_volume = np.sum(slice_volumes)
    target_volume = total_volume / 3

    # Accumulate slices until we reach each volume threshold
    regions = []
    current_region = []
    acc_volume = 0
    region_idx = 0

    for z, vol in zip(z_slices, slice_volumes):
        current_region.append(z)
        acc_volume += vol

        if acc_volume >= target_volume and region_idx < 2:
            regions.append(current_region)
            current_region = []
            acc_volume = 0
            region_idx += 1

    # Add the remaining slices to the last region
    regions.append(current_region)

    # `regions` now has 3 lists: [apex_zs, mid_zs, base_zs]
    return regions

def mirror_mask_y(mask, cy, cx):
    """Flips mask slice over the y axis, considering the mask's 3D centroid"""

    # Get coordinates of nonzero pixels
    coords = np.column_stack(np.nonzero(mask))

    # Mirror Y (i.e., x-axis) across cx
    mirrored_coords = coords.copy()
    mirrored_coords[:, 1] = np.round(2 * cx - coords[:, 1]).astype(int)

    # Filter out-of-bounds coords
    valid = (
        (0 <= mirrored_coords[:, 0]) & (mirrored_coords[:, 0] < mask.shape[0]) &
        (0 <= mirrored_coords[:, 1]) & (mirrored_coords[:, 1] < mask.shape[1])
    )

    mirrored_mask = np.zeros_like(mask)
    mirrored_mask[mirrored_coords[valid][:, 0], mirrored_coords[valid][:, 1]] = 1

    return mirrored_mask


def merge_zonal_masks(mask_tz_cz_arr, mask_pz_arr):
    '''Merges prostate zone masks in numpy array format to obtain a full prostate mask. Assumes
    Returns: processed symmetric zonal masks for TZ+CZ and PZ and symmetric prostate mask in numpy array format'''


    # Create prostate segmentation mask from the union of TZ+CZ and PZ
    mask_prostate_arr = ((mask_tz_cz_arr > 0) | (mask_pz_arr > 0)).astype(np.uint8)

    # Fill small holes in prostate segmentation mask
    mask_prostate_arr_filled = np.stack([
        remove_small_holes(slice.astype(bool), area_threshold=512, connectivity=2)
        for slice in mask_prostate_arr
    ])

    # get added pixels to fill mask holes and assign them to either PZ or TZ+CZ masks
    added_pixels = mask_prostate_arr_filled - mask_prostate_arr
    mask_tz_cz_arr, mask_pz_arr = assign_added_pixels(added_pixels, mask_tz_cz_arr, mask_pz_arr)

    # Flip TZ+CZ and PZ mask slices over the y-axis, considering the 3D mask centroid as reference
    prostate_centroid = get_mask_centroid_3d(mask_prostate_arr)
    C_x, C_y =  prostate_centroid[2], prostate_centroid[1]

    mask_tz_cz_arr_mirrored = np.stack([
        mirror_mask_y(slice, C_y, C_x)
        for slice in mask_tz_cz_arr
    ])

    mask_pz_arr_mirrored = np.stack([
        mirror_mask_y(slice, C_y, C_x)
        for slice in mask_pz_arr
    ])

    # compute the union from original and mirrored masks to obtain symmetric masks 
    # (radiologists are expected to provide symmetric segmentations of the prostate)
    mask_tz_cz_arr = ((mask_tz_cz_arr > 0) | (mask_tz_cz_arr_mirrored > 0)).astype(np.uint8)
    mask_pz_arr = ((mask_pz_arr > 0) | (mask_pz_arr_mirrored > 0)).astype(np.uint8)

    # Reconstruct prostate mask from new TZ+CZ and PZ masks
    mask_prostate_arr = ((mask_tz_cz_arr > 0) | (mask_pz_arr > 0)).astype(np.uint8)

    return mask_tz_cz_arr, mask_pz_arr, mask_prostate_arr

def get_lesion_sectors_mri(row, sector_mapping, score_threshold=3):
    """
    Return PI-RADS sector descriptors where the MRI score is equal to or exceeds a threshold.

    Parameters:
        row: pandas Series with sector score columns matching sector_mapping keys.
        score_threshold: numeric threshold; sectors strictly greater than this are selected.

    Returns:
        list of dicts describing region/side/zone/section for each selected sector.
    """
    # Collect sectors whose score is above the threshold
    lesion_location = []
    for sector in sector_mapping.keys():
        # Skip if the expected column is missing or NaN
        if sector not in row or pd.isna(row[sector]):
            continue
        if row[sector] >= score_threshold:
            lesion_location.append(sector_mapping[sector])
    if len(lesion_location) == 0:
        return None
    return lesion_location

def determine_zone(les_zone_series):
    """
    Determine the zone of the lesion based on MRI CRF.
    """
    
    vals = pd.unique(les_zone_series.dropna())
    if len(vals) == 0:
        return None  # remains NaN/NA
    s = set(vals)
    if s == {1}:
        return 'PZ'
    if s.issubset({2, 3}) or s == {6}:
        return 'TZ+CZ'
    return 'PZ+TZ+CZ'

def adjust_lesion_zones(row):
    """
    Update lesion zones in the lesion_location_mri column based on affected zones.
    """
    az = row.get('affected_zones')
    lesions = row.get('lesion_location_mri')

    # Nothing to do if missing
    if lesions in (None, []) or (not isinstance(az, list) and pd.isna(az)):
        return lesions

    # Determine mode: single zone vs both
    both = (isinstance(az, list) and set(az) == {'PZ', 'TZ+CZ'}) or (isinstance(az, str) and az == 'PZ+TZ+CZ')

    if both:
        out = []
        for d in lesions:
            d1 = dict(d); d1['l_zone'] = 'PZ'
            d2 = dict(d); d2['l_zone'] = 'TZ+CZ'
            out.extend([d1, d2])
        return out

    # Single-zone cases
    if isinstance(az, str) and az in ('PZ', 'TZ+CZ'):
        zone = az
    elif isinstance(az, list) and len(az) == 1 and az[0] in ('PZ', 'TZ+CZ'):
        zone = az[0]
    else:
        return lesions  # unknown value; leave as-is

    return [dict({**d, 'l_zone': zone}) for d in lesions]
