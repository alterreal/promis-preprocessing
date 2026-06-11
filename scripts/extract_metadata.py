#!/usr/bin/env python3
"""
script to extract metadata from DICOM files.
"""

import argparse
from pathlib import Path
from promis_preprocess.dicom_processing import extract_metadata_from_all_dicom_series
from promis_preprocess.metadata_extraction import load_series_descriptions, save_metadata_to_parquet
from promis_preprocess.analysis_utils import analyze_processing_results, log_processing_summary
from promis_preprocess.config_loader import load_config

def main():

    print(40 * '=')
    print('Extracting PROMIS metadata')
    print(40 * '=')
    
    config = load_config()

    parser = argparse.ArgumentParser(description='Extract metadata from DICOM files')
    parser.add_argument('--dicom-raw', type=str, default=None, help='Path to DICOM raw data')
    parser.add_argument('--metadata', type=str, default=None, help='Path to metadata output parquet file')
    parser.add_argument('--series-descriptions', type=str, default=None, help='Path to series descriptions file')
    args = parser.parse_args()


    if args.dicom_raw is None:
        args.dicom_raw = Path(config['paths']['root']) / config['paths']['dicom_raw']
    if args.metadata is None:
        args.metadata = Path(config['paths']['root']) / config['paths']['metadata']
    if args.series_descriptions is None:
        args.series_descriptions = Path(config['paths']['root']) / config['paths']['series_descriptions']

    # Ensure output directory exists
    Path(args.metadata).parent.mkdir(parents=True, exist_ok=True)

    
    print("\n>>> Starting DICOM metadata extraction...")
    print(f"\nDICOM path: {args.dicom_raw}")
    print(f"Output path: {args.metadata}")
    print(f"Series descriptions path: {args.series_descriptions}")
    
    # Load series descriptions
    print("\n>>> Loading series descriptions...")
    series_descriptions = load_series_descriptions(args.series_descriptions)
    
    # Process all DICOM series
    print("\n>>> Extracting metadata from DICOM series...")
    metadata, stats, log_file = extract_metadata_from_all_dicom_series(
        args.dicom_raw, 
        series_descriptions,
    )
    
    print(f'Metadata extraction complete: {stats["processed"]} processed, {stats["errors"]} errors, {stats["warnings"]} warnings')
    
    # Save metadata to parquet
    print("\n>>> Saving metadata to parquet...")
    df_metadata = save_metadata_to_parquet(metadata, args.metadata)
    
    # Generate analysis and reports
    print("\n>>> Generating analysis and reports...")
    analyze_processing_results(metadata, stats, log_file)
    log_processing_summary(stats, log_file)
    
    print("\n>>> Metadata extraction completed successfully!")


if __name__ == "__main__":
    main()
