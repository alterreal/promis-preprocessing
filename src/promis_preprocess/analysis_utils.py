"""
Analysis and reporting utilities for DICOM processing.
"""

import pandas as pd
import os


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


'''def analyze_sequence_distribution(df_metadata, sequence_label):
    """Analyze distribution of a specific sequence type."""
    if sequence_label not in df_metadata['generic_sequence_label'].values:
        print(f"No {sequence_label} sequences found in the dataset")
        return None
    
    analysis = df_metadata[df_metadata['generic_sequence_label'] == sequence_label].groupby('patient_id').count().sort_values(by='generic_sequence_label', ascending=False)
    
    print(f"\n{sequence_label.upper()} sequences per patient (top 20):")
    print(analysis.head(20))
    
    return analysis'''


def generate_summary_report(df_metadata, output_path):
    """Generate a comprehensive summary report."""
    report_file = os.path.join(output_path, "processing_summary.txt")
    
    with open(report_file, "w") as f:
        f.write("DICOM Processing Summary Report\n")
        f.write("=" * 40 + "\n\n")
        
        f.write(f"Total series processed: {len(df_metadata)}\n")
        f.write(f"Unique patients: {df_metadata['patient_id'].nunique()}\n")
        f.write(f"Unique series descriptions: {df_metadata['series_description'].nunique()}\n\n")
        
        f.write("Scanner Types:\n")
        for scanner_type, count in df_metadata['scanner_type'].value_counts().items():
            f.write(f"  {scanner_type}: {count}\n")
        
        f.write("\nGeneric Sequence Labels:\n")
        for label, count in df_metadata['generic_sequence_label'].value_counts().items():
            f.write(f"  {label}: {count}\n")
        
        f.write("\nScanner Manufacturers:\n")
        for manufacturer, count in df_metadata['scanner_manufacturer'].value_counts().items():
            f.write(f"  {manufacturer}: {count}\n")
    
    print(f"Summary report saved to {report_file}")
