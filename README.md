# PROMIS Dataset Preprocessing Pipeline

This is a simple Python pipeline for preprocessing the [PROMIS (Prostate MR Image Segmentation)](https://www.reimagine-pca.org/about-7) dataset. It handles DICOM medical imaging data with:

1. Metadata extraction
2. Series organization
3. Resampling
4. Lesion location priori extraction from MRI and TPM CRFs. 

## About PROMIS Dataset

The PROMIS dataset is a prostate MRI dataset available at reasonable request. It contains multi-parametric MRI sequences including T2-weighted, DWI, and ADC images from 575 patients.

## About the Location Priori

Lesion location priori computation depends on the availability of prostate zone segmentation masks, which are not provided here. If provided, the location priori will be extracted according to both MRI and TPM CRFs, separately. Although lesion location is defined in the TPM CRF according to Barzell Zones, the extracted priori are defined according to the PI-RADS v2 sectors affected by the lesion. The conversion from Barzell zones to PI-RADS sectors is done according to the mapping proposed by [Satish et al. (2021)](https://doi.org/10.1016/j.eururo.2021.05.017). Because the Barzell zonal system does not distinguish the mid-gland as a separate axial region, the resulting PI-RADS sectors are assigned only to either the apex or the base, effectively splitting the gland along a halved axial axis.

## Generalizability

While specifically designed for the PROMIS dataset, this pipeline can be adapted for other DICOM datasets with minimal changes. The main requirement is an Excel file (`.xlsx`) that maps series descriptions from DICOM files to generic sequence labels (e.g., `t2_axial`, `dwi_b1400_axial`, `adc_axial`). This mapping file should contain:

- **Patient ID**: Patient identifier
- **Series Description**: Original series description from DICOM metadata
- **Generic Sequence Label**: Standardized label for the sequence type

## Usage

### 1. Configure the Pipeline

Edit `config.yaml` to point the pipeline at your data and outputs. All paths are relative to `paths.root`, except where an absolute path is given. Each script also accepts path overrides via command-line arguments (e.g. `--dicom-raw`, `--metadata`).

You can also customize which MRI sequences to process and which one to use as the reference for resampling:

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
# Step 1: Extract metadata from DICOM seriea
uv run python scripts/extract_metadata.py

# Step 2: Process studies with all the selected sequences and resample to reference series
uv run python scripts/process_studies.py

# Step 3: Compute location priori (using PI-RADS v2 sectors) of clinically significant lesions (ISUP ≥ 2) according to MRI and TPM CRFs. 
uv run python scripts/compute_location_priori.py --label-type cspca

# in case you want to extract the location priori for ISUP ≥ 1 lesions
uv run python scripts/compute_location_priori.py --label-type pca
```

## Output Structure

process_studies.py generates an organized structure suitable for ML pipelines:

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

metadata/series_metadata.parquet    # Series metadata

```

If `--nnunet-output` is passed to `process_studies.py`, images are written in nnUNet format to the directory configured in `paths.nnunet_output`. All studies are saved in a single flat folder using the naming convention `{study_id}_{channel}.mha`, where channels are mapped via `nnunet_series_dict` in `config.yaml` (e.g. `study_001_0000.mha` for T2, `study_001_0001.mha` for DWI, `study_001_0002.mha` for ADC). Series are still resampled to the reference series before saving.


