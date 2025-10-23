# PROMIS Dataset Preprocessing Pipeline

A specialized Python pipeline for preprocessing and organizing the [PROMIS (Prostate MR Image Segmentation)](https://www.reimagine-pca.org/about-7) Open Access dataset for machine learning pipelines. This tool handles DICOM medical imaging data with metadata extraction, series organization, and ML-ready data preparation.

## About PROMIS Dataset

The PROMIS dataset is a prostate MRI dataset designed for prostate cancer segmentation and analysis, available at reasonable request. It contains multi-parametric MRI sequences including T2-weighted, DWI, and ADC images from multiple patients and studies.

## Generalizability

While specifically designed for the PROMIS dataset, this pipeline can be adapted for other DICOM datasets with minimal changes. The main requirement is an Excel file (`.xlsx`) that maps series descriptions from DICOM files to generic sequence labels (e.g., `t2_axial`, `dwi_b1400_axial`, `adc_axial`). This mapping file should contain:

- **Patient ID**: Patient identifier
- **Series Description**: Original series description from DICOM metadata
- **Generic Sequence Label**: Standardized label for the sequence type

## Project Structure

```
promis_preprocess/
├── config.yaml               # YAML configuration file
├── src/promis_preprocess/    # Main package
│   ├── config_loader.py      # YAML configuration loader
│   ├── dicom_processing.py   # Core DICOM processing functions
│   ├── metadata_extraction.py # Metadata extraction utilities
│   └── analysis_utils.py     # Analysis and reporting functions
├── scripts/                  # Processing scripts
│   ├── process_studies.py    # Main study processing script
│   └── organize_metadata.py  # Metadata organization script
├── notebooks/                # Jupyter notebooks
│   └── exploration.ipynb     # Interactive analysis notebook
├── pyproject.toml           # Python dependencies
└── README.md                # This file
```

## Key Features

- **Multi-sequence Processing**: Handles pre-selected MRI sequences
- **Series Organization**: Automatically organizes DICOM series by type and study
- **Metadata Extraction**: Comprehensive extraction of DICOM metadata for ML pipelines
- **Resampling**: Resamples all sequences to a reference series for consistency
- **ML-Ready Output**: Generates organized data structure suitable for machine learning

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd promis_preprocess

# Install dependencies
pip install -e .
```

## Usage

### 1. Configure the Pipeline

You can customize which MRI sequences to process and which one to use as the reference for resampling in the `config.yaml`:

```yaml

# Series to process - customize which sequences you want
series_to_process:
  t2_axial: "T2"
  dwi_b1400_axial: "DWI"
  adc_axial: "ADC"
  # Add or remove sequences as needed for your dataset

# Reference series for resampling - choose which sequence to use as reference
reference_series: "t2_axial"
```

### 2. Process the Dataset

```bash
# Step 1: Organize metadata and extract DICOM information
python scripts/organize_metadata.py

# Step 2: Process studies and resample to reference series
python scripts/process_studies.py
```

## Output Structure

The pipeline generates an organized structure suitable for ML pipelines:

```
processed/
├── patient_001/
│   ├── study_001/
│   │   ├── image_T2.mha      # T2-weighted image
│   │   ├── image_DWI.mha     # DWI image 
│   │   └── image_ADC.mha     # ADC image
│   └── study_002/
│       └── ...
└── patient_002/
    └── ...

metadata/
├── series_metadata.parquet    # Comprehensive metadata
├── processing_log.txt         # Processing log
└── summary_report.txt         # Summary statistics
```

## ML Pipeline Integration

The processed data is ready for machine learning pipelines:

- **Consistent Format**: All images resampled to same resolution
- **Organized Structure**: Patient/Study hierarchy for easy data loading
- **Standard Formats**: MHA format compatible with medical imaging libraries
