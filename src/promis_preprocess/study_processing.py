import pandas as pd
from pathlib import Path
from tqdm import tqdm
import SimpleITK as sitk
import numpy as np
from .dicom_processing import create_dicom_reader, load_dicom_image_from_folder
from .zonal_masks import load_zone_predictions, keep_components_in_contact, keep_components_in_mid_x_region

"""
Study processing functions.
"""


def select_single_series_per_type(metadata, series_to_process):
    """
    For each study_id, keep only one series per type in series_to_process.keys().
    Only keeps studies that have all required series.
    """
    res = []
    series_types = list(series_to_process.keys())
    
    for study_id, group in metadata.groupby('study_id'):
        found_types = [
            seq_type
            for seq_type in series_types
            if not group[group['generic_sequence_label'] == seq_type].empty
        ]
        # Only keep studies that have all required series
        if set(found_types) == set(series_types):
            for seq_type in series_types:
                seq_group = group[group['generic_sequence_label'] == seq_type]
                # RULE: choose the last one (can adapt: e.g., first)
                selected = seq_group.iloc[-1]
                res.append(selected)
    
    return pd.DataFrame(res)


def process_and_save_studies(
    metadata, 
    series_to_process, 
    reference_series, 
    dicom_raw_path, 
    dicom_processed_path, 
    nnunet_output=False, 
    nnunet_output_path=None, 
    nnunet_series_dict=None, 
    zone_predictions_path=None
    ):
    """
    Processes and saves images in MetaImage Header (.mha) format for each study in the provided metadata DataFrame.
    All series are resampled to the reference series.
    """

    discarded_cases = []

    for study_id, study_metadata in tqdm(metadata.groupby('study_id'), 
                                       total=metadata['study_id'].nunique(), 
                                       desc="Processing studies"):

        # first check if zonal mask is valid (if provided)
        if zone_predictions_path is not None:

            mask_tz_cz, mask_pz = load_zone_predictions(zone_predictions_path / f'{study_id}.mha')
            
            mask_tz_cz_arr = sitk.GetArrayFromImage(mask_tz_cz)
            mask_pz_arr = sitk.GetArrayFromImage(mask_pz)

            # Keep only components from each mask that are in contact with components from the other mask
            mask_tz_cz_arr, mask_pz_arr = keep_components_in_contact(mask_tz_cz_arr, mask_pz_arr)

            # Keep only components that intersect the mid x region (with 10% margin)
            mask_tz_cz_arr = keep_components_in_mid_x_region(mask_tz_cz_arr, margin_ratio=0.1)
            mask_pz_arr = keep_components_in_mid_x_region(mask_pz_arr, margin_ratio=0.1)

            if len(np.unique(mask_tz_cz_arr)) != 2 or len(np.unique(mask_tz_cz_arr)) != 2:
                discarded_cases.append(study_id)
                continue

        # Prepare save path
        patient_id = study_metadata.iloc[0]['patient_id']
        
        # Load reference image
        ref_row = study_metadata[study_metadata['generic_sequence_label'] == reference_series].iloc[0]
        reference_series_path = Path(dicom_raw_path) / ref_row['folder_path']
        ref_reader = create_dicom_reader()
        reference_image = load_dicom_image_from_folder(ref_reader, reference_series_path)

        # Decide output directory based on format
        if nnunet_output:
            base_output_dir = Path(nnunet_output_path)
            base_output_dir.mkdir(parents=True, exist_ok=True)
            def make_filename(series_key):
                return f"{study_id}_{nnunet_series_dict[series_key]}.mha"
        else:
            base_output_dir = Path(dicom_processed_path) / patient_id / study_id
            base_output_dir.mkdir(parents=True, exist_ok=True)
            def make_filename(series_key):
                return f"image_{series_to_process[series_key]}.mha"

        # Save reference image
        ref_output_path = base_output_dir / make_filename(reference_series)
        sitk.WriteImage(reference_image, ref_output_path)

        # Set up resampler with specified reference image and interpolator
        resample = sitk.ResampleImageFilter()
        resample.SetReferenceImage(reference_image)
        resample.SetInterpolator(sitk.sitkNearestNeighbor)

        # Process remaining series types
        for seq_type in series_to_process:
            if seq_type == reference_series:
                continue

            add_row = study_metadata[study_metadata['generic_sequence_label'] == seq_type].iloc[0]
            additional_series_path = Path(dicom_raw_path) / add_row['folder_path']
            add_reader = create_dicom_reader()
            add_image = load_dicom_image_from_folder(add_reader, additional_series_path)
            resampled_image = resample.Execute(add_image)

            # Save resampled image
            add_output_path = base_output_dir / make_filename(seq_type)
            sitk.WriteImage(resampled_image, add_output_path)

    return discarded_cases