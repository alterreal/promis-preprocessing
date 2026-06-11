"""
Analysis and reporting utilities for DICOM processing.
"""

import pandas as pd


def analyze_processing_results(metadata, stats, log_file):
    """Analyze the results of DICOM processing."""
    print("=== Processing Analysis ===")
    print(f"Total series processed: {stats['processed']}")
    print(f"Total errors: {stats['errors']}")
    print(f"Total warnings: {stats['warnings']}")
    print(f"Detailed log available at: {log_file}")
    
    # Analyze metadata
    if metadata:
        df = pd.DataFrame(metadata)
        print(f"\nMetadata analysis:")
        print(f"  - Unique patients: {df['patient_id'].nunique()}")
        print(f"  - Unique series descriptions: {df['series_description'].nunique()}")
        print(f"  - Scanner types: {df['scanner_manufacturer'].value_counts().to_dict()}")
        print(f"  - Generic sequence labels: {df['generic_sequence_label'].value_counts().to_dict()}")


def log_processing_summary(stats, log_file):
    """Log processing summary to the main log file."""
    with open(log_file, 'a') as f:
        f.write(f"\n=== Processing Summary ===\n")
        f.write(f"Total processed: {stats['processed']}\n")
        f.write(f"Total errors: {stats['errors']}\n")
        f.write(f"Total warnings: {stats['warnings']}\n")
        f.write(f"Success rate: {stats['processed']/(stats['processed']+stats['errors'])*100:.1f}%\n")

