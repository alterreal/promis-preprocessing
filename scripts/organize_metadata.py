#!/usr/bin/env python3
"""
Main script for DICOM processing pipeline.
"""

import os
import sys
from promis_preprocess.dicom_processing import process_all_dicom_series
from promis_preprocess.metadata_extraction import load_series_descriptions, save_metadata_to_parquet
from promis_preprocess.analysis_utils import analyze_processing_results, log_processing_summary, generate_summary_report
from promis_preprocess.config_loader import config


def main():
    """Main processing function."""
    # Use default config or allow override via command line
    config_dict = config['paths'].copy()
    
    # Override with command line arguments if provided
    if len(sys.argv) > 1:
        config_dict['dicom_raw'] = sys.argv[1]
    if len(sys.argv) > 2:
        config_dict['metadata'] = sys.argv[2]
    
    # Ensure output directory exists
    os.makedirs(config_dict['metadata'], exist_ok=True)
    
    print("Starting DICOM processing pipeline...")
    print(f"DICOM path: {config_dict['dicom_raw']}")
    print(f"Output path: {config_dict['metadata']}")
    
    # Load series descriptions
    print("Loading series descriptions...")
    series_descriptions = load_series_descriptions(config_dict['series_descriptions'])
    
    # Process all DICOM series
    print("Processing DICOM series...")
    metadata, stats, log_file = process_all_dicom_series(
        config_dict['dicom_raw'], 
        series_descriptions,
        config_dict['metadata']
    )
    
    print(f'Processing complete: {stats["processed"]} processed, {stats["errors"]} errors, {stats["warnings"]} warnings')
    
    # Save metadata to parquet
    print("Saving metadata...")
    df_metadata = save_metadata_to_parquet(metadata, config_dict['metadata'])
    
    # Generate analysis and reports
    print("Generating analysis and reports...")
    analyze_processing_results(metadata, stats, log_file)
    log_processing_summary(stats, log_file)
    generate_summary_report(df_metadata, config_dict['metadata'])
    
    print("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
