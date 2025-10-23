#!/usr/bin/env python3
"""
Script to process DICOM studies and resample them to a reference series.
"""

import os
import argparse
import SimpleITK as sitk
import pandas as pd
from tqdm import tqdm

from promis_preprocess.config_loader import config
from promis_preprocess.dicom_processing import create_dicom_reader, load_dicom_image_from_folder


def select_single_series_per_type(metadata):
    """
    For each study_id, keep only one series per type in SERIES_TO_PROCESS.keys().
    Only keeps studies that have all required series.
    """
    res = []
    series_types = list(config['series_to_process'].keys())
    
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
                # RULE: choose the last one (can adapt: e.g., latest, largest, etc)
                selected = seq_group.iloc[-1]
                res.append(selected)
    
    return pd.DataFrame(res)


def process_and_save_studies(metadata, series_to_process, reference_series, paths_config):
    """
    Processes and saves Medical Format images for each study in the provided metadata DataFrame.
    The reference series is used to resample all others.
    """
    
    for study_id, study_metadata in tqdm(metadata.groupby('study_id'), 
                                       total=metadata['study_id'].nunique(), 
                                       desc="Processing studies"):
        # Prepare save path
        patient_id = study_metadata.iloc[0]['patient_id']
        save_path = os.path.join(
            paths_config['dicom_processed'], f"{patient_id}/{study_id}/"
        )
        os.makedirs(save_path, exist_ok=True)

        # Load reference image
        ref_row = study_metadata[study_metadata['generic_sequence_label'] == reference_series].iloc[0]
        reference_series_path = os.path.join(paths_config['dicom_raw'], ref_row['folder_path'])
        ref_reader = create_dicom_reader()
        reference_image = load_dicom_image_from_folder(ref_reader, reference_series_path)

        # Save reference image
        ref_output_path = os.path.join(save_path, f"image_{series_to_process[reference_series]}.mha")
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
            additional_series_path = os.path.join(paths_config['dicom_raw'], add_row['folder_path'])
            add_reader = create_dicom_reader()
            add_image = load_dicom_image_from_folder(add_reader, additional_series_path)
            resampled_image = resample.Execute(add_image)

            # Save resampled image
            add_output_path = os.path.join(save_path, f"image_{series_to_process[seq_type]}.mha")
            sitk.WriteImage(resampled_image, add_output_path)


def main():
    """Main processing function."""
    parser = argparse.ArgumentParser(description='Process DICOM studies and resample to reference series')
    parser.add_argument('--metadata-path', 
                       help='Path to metadata parquet file')
    parser.add_argument('--output-path', 
                       help='Override processed output path')
    parser.add_argument('--raw-path', 
                       help='Override DICOM raw path')
    
    args = parser.parse_args()
    
    # Load configuration
    config_dict = config['paths'].copy()
    
    # Override with command line arguments
    if args.output_path:
        config_dict['dicom_processed'] = args.output_path
    if args.raw_path:
        config_dict['dicom_raw'] = args.raw_path
    if args.metadata_path:
        config_dict['metadata'] = args.metadata_path
    
    print("Starting DICOM study processing...")
    print(f"Metadata path: {config_dict['metadata']}")
    print(f"DICOM raw path: {config_dict['dicom_raw']}")
    print(f"Processed output path: {config_dict['dicom_processed']}")
    print(f"Reference series: {config['reference_series']}")
    print(f"Series to process: {list(config['series_to_process'].keys())}")
    
    # Load metadata
    print("Loading metadata...")
    metadata = pd.read_parquet(os.path.join(config_dict['metadata'], 'series_metadata.parquet'))
    print(f"Loaded {len(metadata)} series from {metadata['study_id'].nunique()} studies")
    
    # Filter to studies with all required series
    print("Filtering studies with all required series...")
    metadata_filtered = select_single_series_per_type(metadata)
    print(f"Filtered to {len(metadata_filtered)} series from {metadata_filtered['study_id'].nunique()} studies")
    
    if len(metadata_filtered) == 0:
        print("No studies found with all required series types!")
        print("Available series types:")
        print(metadata['generic_sequence_label'].value_counts())
        return
    
    # Process and save studies
    print("Processing studies...")
    process_and_save_studies(metadata_filtered, config['series_to_process'], config['reference_series'], config_dict)
    
    print("Processing completed successfully!")


if __name__ == "__main__":
    main()
