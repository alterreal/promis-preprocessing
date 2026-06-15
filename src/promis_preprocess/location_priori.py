import numpy as np
import pandas as pd
import cc3d
from scipy.ndimage import distance_transform_edt
from skimage.morphology import remove_small_holes
import os
import SimpleITK as sitk
from tqdm import tqdm
from .metadata_extraction import get_isup_grade
from .study_processing import select_single_series_per_type


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
        remove_small_holes(slice.astype(bool), max_size=511, connectivity=2)
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


def load_mri_crf(mri_crf_path):
    mri_crf = pd.read_excel(mri_crf_path)
    return mri_crf[mri_crf["FormCycle"] == 1].copy()


def build_mri_crf_patient(mri_crf, config):
    
    mri_crf_patient = mri_crf.groupby("patientID").first()
    sector_mapping = config["sector_mapping"]

    mri_crf_patient["lesion_location_mri"] = mri_crf_patient.apply(
        get_lesion_sectors_mri, axis=1, args=(sector_mapping, 3)
    )

    for patient_id, patient_data in mri_crf.groupby("patientID"):
        mri_crf_patient.at[patient_id, "affected_zones"] = determine_zone(
            patient_data["les_zone"]
        )

    mri_crf_patient["lesion_location_mri"] = mri_crf_patient.apply(
        adjust_lesion_zones, axis=1
    )
    return mri_crf_patient


def load_tpm_crf(tpm_crf_folder, config, isup_thres, label_type):
    all_tpm_data = []
    tpm_zone_files = config["tpm_zone_files"]

    for file in tpm_zone_files:
        tpm_crf_path = tpm_crf_folder / file
        tpm_crf = pd.read_excel(tpm_crf_path)
        tpm_crf.set_index("patientID", inplace=True)
        tpm_crf.drop(
            columns=[
                "Trial",
                "Site",
                "VisitCycle",
                "FormCycle",
                "PersonId",
                "RepeatNumber",
            ],
            inplace=True,
        )

        num_cols = tpm_crf.shape[1]
        df_1 = tpm_crf.iloc[:, : int(num_cols / 2)]
        df_2 = tpm_crf.iloc[:, int(num_cols / 2) :]

        if file == tpm_zone_files[0]:
            df_1.rename(columns={"patientID.1": "d_patid"}, inplace=True)

        df_2.columns = df_1.columns
        all_tpm_data.append(pd.concat([df_1, df_2], axis=0))

    tpm_crf = pd.concat(all_tpm_data, axis=0)
    tpm_crf["isup"] = tpm_crf.apply(
        lambda row: get_isup_grade(row["zprim1"], row["zseco1"]), axis=1
    )
    tpm_crf["positive_lesion"] = tpm_crf["isup"] >= isup_thres
    tpm_crf[f"case_{label_type.lower()}"] = tpm_crf.groupby(level=0)["positive_lesion"].any()
    return tpm_crf


def add_tpm_lesion_locations(mri_crf_patient, tpm_crf, barzell_to_pirads_map, isup_thres, label_type):
    for patient_id, patient_data in tpm_crf.groupby(level=0):
        sectors = [
            sector
            for _, zone in patient_data.iterrows()
            if pd.notna(zone["isup"]) and zone["isup"] >= isup_thres
            for sector in barzell_to_pirads_map[zone["d_zone"]]
        ]
        mri_crf_patient.at[patient_id, "lesion_location_tpm"] = sectors if sectors else None
        mri_crf_patient.at[patient_id, f"case_{label_type.lower()}"] = patient_data[f"case_{label_type.lower()}"].any()


def load_series_metadata(metadata_path, config):
    series_metadata = pd.read_parquet(metadata_path)
    series_metadata.set_index("patient_id", inplace=True)
    return select_single_series_per_type(series_metadata, config["series_to_process"])


def load_tpm_crf_summary(tpm_crf_summary_path, series_metadata, isup_thres, label_type):
    tpm_crf_summary = pd.read_excel(tpm_crf_summary_path)
    tpm_crf_summary.set_index("patientID", inplace=True)
    tpm_crf_summary["isup"] = tpm_crf_summary.apply(
        lambda row: get_isup_grade(row["sumal1"], row["sumal2"]), axis=1
    )
    tpm_crf_summary[f"case_{label_type.lower()}"] = tpm_crf_summary["isup"] >= isup_thres

    print(
        f"The summary TPM CRF contains {len(tpm_crf_summary)} patients, "
        f"of which {tpm_crf_summary[f'case_{label_type.lower()}'].sum()} have a {label_type} lesion"
    )

    tpm_crf_summary = tpm_crf_summary.loc[series_metadata.index.unique()]
    print(
        f"After discarding cases without all sequences, the summary TPM CRF contains "
        f"{len(tpm_crf_summary)} patients, of which "
        f"{tpm_crf_summary[f'case_{label_type.lower()}'].sum()} have a {label_type} lesion"
    )
    return tpm_crf_summary


def merge_case_labels(mri_crf_patient, series_metadata, tpm_crf_summary, label_type):
    mri_crf_patient = mri_crf_patient.loc[series_metadata.index.unique()]
    mask = mri_crf_patient[f"case_{label_type.lower()}"].isna() & (tpm_crf_summary[f"case_{label_type.lower()}"] == False)
    mri_crf_patient.loc[mask, f"case_{label_type.lower()}"] = tpm_crf_summary.loc[mask, f"case_{label_type.lower()}"]

    print(
        f"Dropping {pd.isna(mri_crf_patient[f'case_{label_type.lower()}']).sum()} {label_type} "
        "cases with no detailed TPM CRF data"
    )
    mri_crf_patient.dropna(subset=[f"case_{label_type.lower()}"], inplace=True)
    return mri_crf_patient


def compute_location_priors(
    mri_crf_patient,
    dicom_processed,
    zonal_masks,
    location_priori_mri,
    location_priori_tpm,
    config,
):
    location_priori_mri.mkdir(parents=True, exist_ok=True)
    location_priori_tpm.mkdir(parents=True, exist_ok=True)

    discarded_cases = []
    for p_id in tqdm(mri_crf_patient.index, desc="Computing location priors"):
        try:
            study_uid = os.listdir(dicom_processed / p_id)[0]
            mri_crf_patient.loc[p_id, "study_uid"] = study_uid

            filepath_t2 = (
                dicom_processed
                / p_id
                / study_uid
                / f"image_{config['series_to_process']['t2_axial']}.mha"
            )
            filepath_mask = zonal_masks / f"{study_uid}.mha"

            t2_img = sitk.ReadImage(str(filepath_t2))
            mask_img = sitk.ReadImage(str(filepath_mask))

            mask_arr = sitk.GetArrayFromImage(mask_img)
            mask_tz_cz_arr = (mask_arr == 1).astype(np.uint8)
            mask_pz_arr = (mask_arr == 2).astype(np.uint8)
            mask_tz_cz_arr, mask_pz_arr, mask_prostate_arr = merge_zonal_masks(
                mask_tz_cz_arr, mask_pz_arr
            )

            sector_list_mri = mri_crf_patient.loc[p_id].lesion_location_mri
            sector_list_tpm = mri_crf_patient.loc[p_id].lesion_location_tpm

            if isinstance(sector_list_mri, list):
                mask_lesion_mri = get_reported_lesion_mask(
                    mask_prostate_arr,
                    mask_tz_cz_arr,
                    mask_pz_arr,
                    sector_list_mri,
                )
            else:
                mask_lesion_mri = np.zeros_like(mask_prostate_arr)

            if isinstance(sector_list_tpm, list):
                mask_lesion_tpm = get_reported_lesion_mask(
                    mask_prostate_arr,
                    mask_tz_cz_arr,
                    mask_pz_arr,
                    sector_list_tpm,
                    region_div_method="barzell",
                )
            else:
                mask_lesion_tpm = np.zeros_like(mask_prostate_arr)

            _, cc_count_mri = get_connected_components(mask_lesion_mri, connectivity=6)
            _, cc_count_tpm = get_connected_components(mask_lesion_tpm, connectivity=6)

            mri_crf_patient.loc[p_id, "cc_count_mri"] = cc_count_mri
            mri_crf_patient.loc[p_id, "cc_count_tpm"] = cc_count_tpm

            mask_lesion_mri_arr = sitk.GetImageFromArray(mask_lesion_mri)
            mask_lesion_tpm_arr = sitk.GetImageFromArray(mask_lesion_tpm)
            mask_lesion_mri_arr.CopyInformation(t2_img)
            mask_lesion_tpm_arr.CopyInformation(t2_img)

            sitk.WriteImage(mask_lesion_mri_arr, str(location_priori_mri / f"{study_uid}.mha"))
            sitk.WriteImage(mask_lesion_tpm_arr, str(location_priori_tpm / f"{study_uid}.mha"))

        except Exception:
            discarded_cases.append(p_id)
            continue

    print(f"{len(discarded_cases)} cases were not found in the DICOM processed folder")
    return mri_crf_patient


def save_lesion_metadata(mri_crf_patient, metadata_dir, label_type):
    mri_crf_patient.dropna(subset=["study_uid"], inplace=True)
    mri_crf_patient.reset_index(inplace=True, drop=False, names=["patient_id"])
    output_path = metadata_dir / f"lesion_{label_type}_metadata.csv"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    mri_crf_patient.to_csv(output_path, index=False)
    print(f"Lesion metadata saved to {output_path}")
