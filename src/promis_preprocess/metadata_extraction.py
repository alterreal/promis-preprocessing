"""
Metadata extraction and processing utilities.
"""

import pandas as pd
from pathlib import Path
from .config_loader import load_config

config = load_config()

def get_isup_grade(gleason_primary, gleason_secondary):
    """ Extracts the ISUP grade from the Gleason patterns."""

    if pd.isna(gleason_primary) or pd.isna(gleason_secondary):
        return None  

    gleason_score = gleason_primary + gleason_secondary

    if gleason_primary == 3 and gleason_secondary == 3:
        return 1
    elif gleason_primary == 3 and gleason_secondary == 4:
        return 2
    elif gleason_primary == 4 and gleason_secondary == 3:
        return 3
    elif gleason_score == 8:
        return 4
    elif gleason_score >= 9:
        return 5
    elif gleason_score < 6:
        return 0
    else:
        return None

def load_series_descriptions(series_descriptions_path):
    """Load and preprocess series descriptions from Excel file."""
    series_descriptions = pd.read_excel(series_descriptions_path)
    series_descriptions.columns = ['patient_id', 'series_description', 'generic_sequence_label']
    series_descriptions['series_description'] = series_descriptions['series_description'].str.replace(' ', '')
    series_descriptions = series_descriptions.iloc[1:].set_index(['patient_id', 'series_description'])
    return series_descriptions


def extract_metadata_from_reader(reader, image, path, series_descriptions, base_path):
    """Extract metadata from DICOM reader and image."""
    patient_id = reader.GetMetaData(0, config['dicom_tags']['patient_id'])
    series_description = reader.GetMetaData(0, config['dicom_tags']['series_description'])
    
    # Get generic sequence label with error handling
    try:
        generic_sequence_label = series_descriptions.loc[
            (patient_id, series_description.replace(' ', '')), 
            'generic_sequence_label'
        ]
    except KeyError:
        generic_sequence_label = 'unknown'
        print(f"Warning: No generic sequence label found for patient {patient_id}, series {series_description}")
    
    return {
        'patient_id': patient_id,
        'study_id': reader.GetMetaData(0, config['dicom_tags']['study_id']),
        'series_id': reader.GetMetaData(0, config['dicom_tags']['series_id']),
        'scanner_type': reader.GetMetaData(0, config['dicom_tags']['scanner_type']),
        'scanner_manufacturer': reader.GetMetaData(0, config['dicom_tags']['scanner_manufacturer']),
        'scanner_model': reader.GetMetaData(0, config['dicom_tags']['scanner_model']),
        'magnetic_field_strength': reader.GetMetaData(0, config['dicom_tags']['magnetic_field_strength']),
        'series_description': series_description,
        'generic_sequence_label': generic_sequence_label,
        'num_dicom_files': len(list(Path(path).iterdir())),
        'num_loaded_slices': image.GetSize()[2],
        'size': image.GetSize(),
        'pixel_spacing': image.GetSpacing(),
        'folder_path': str(Path(path).relative_to(Path(base_path)))
    }


def save_metadata_to_parquet(metadata, output_parquet):
    """Save metadata to parquet file."""
    df_metadata = pd.DataFrame(metadata)
    df_metadata.to_parquet(output_parquet)
    print(f"Metadata saved to {output_parquet}")
    print(f"DataFrame shape: {df_metadata.shape}")
    return df_metadata
