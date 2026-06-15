#!/usr/bin/env python3
"""
script to process DICOM studies, resample them to a reference series and save the images in MetaImage Header (.mha) format.
"""

from pathlib import Path
import argparse
import pandas as pd

from promis_preprocess.config_loader import load_config
from promis_preprocess.study_processing import process_and_save_studies, select_single_series_per_type

def main():

    print(40 * '=')
    print('PROMIS studies processing')
    print(40 * '=')

    parser = argparse.ArgumentParser(description='Process DICOM studies, resample to reference series and save to output path')
    parser.add_argument('--metadata', 
                       type=str,
                       default=None,
                       help='Path to metadata parquet file')
    parser.add_argument('--dicom-raw', 
                       type=str,
                       default=None,
                       help='Path to DICOM raw data')
    parser.add_argument('--dicom-processed', 
                       type=str,
                       default=None,
                       help='Path to processed DICOM data')
    parser.add_argument(
                        '--zone-predictions', 
                        type=str, 
                        default=None, 
                        help='Path to the zone predictions. If provided, only cases with valid zonal mask will be processed'
                        )
    parser.add_argument('--nnunet-output', 
                       action='store_true',
                       help='If set, output in nnUNet format')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    # Override with command line arguments
    if args.dicom_processed is None:
        args.dicom_processed = Path(config['paths']['root']) / config['paths']['dicom_processed']
    if args.dicom_raw is None:
        args.dicom_raw = Path(config['paths']['root']) / config['paths']['dicom_raw']
    if args.metadata is None:
        args.metadata = Path(config['paths']['root']) / config['paths']['metadata']
    if args.zone_predictions is None:
        args.zone_predictions = Path(config['paths']['root']) / config['paths']['zonal_masks']

    # set nnunet output path
    nnunet_output_path = Path(config['paths']['root']) / config['paths']['nnunet_output']
    nnunet_output_path.mkdir(parents=True, exist_ok=True)
    
    print(">>> Starting DICOM study processing...")
    print(f"Metadata path: {args.metadata}")
    print(f"DICOM raw path: {args.dicom_raw}")
    if args.nnunet_output:
        print(f"nnUNet output: {nnunet_output_path}")
    else:
        print(f"Processed output path: {args.dicom_processed}")
    print(f"Reference series: {config['reference_series']}")
    print(f"Series to process: {list(config['series_to_process'].keys())}")
    
    # Load metadata
    print(">>> Loading metadata...")
    metadata = pd.read_parquet(args.metadata)
    print(f"Loaded {len(metadata)} series from {metadata['study_id'].nunique()} studies and {metadata['patient_id'].nunique()} patients")
    
    # Filter to studies with all required series
    print(">>> Filtering studies with all required series...")
    metadata_filtered = select_single_series_per_type(metadata, config['series_to_process'])
    print(f"Filtered to {len(metadata_filtered)} series from {metadata_filtered['study_id'].nunique()} studies and {metadata_filtered['patient_id'].nunique()} patients")
    
    if len(metadata_filtered) == 0:
        print("No studies found with all required series types!")
        print("Available series types:")
        print(metadata['generic_sequence_label'].value_counts())
        return


    # Process and save studies
    print(">>> Processing studies...")
    discarded_cases = process_and_save_studies(
        metadata_filtered, 
        config['series_to_process'], 
        config['reference_series'], 
        args.dicom_raw, args.dicom_processed, 
        nnunet_output=args.nnunet_output, 
        nnunet_output_path=nnunet_output_path, 
        nnunet_series_dict=config['nnunet_series_dict'],
        zone_predictions_path=args.zone_predictions
        )
    if args.zone_predictions is not None:
        print(f"Discarded {len(discarded_cases)} studies due to invalid zonal masks")
        
    print(">>> Processing completed successfully!")


if __name__ == "__main__":
    main()
