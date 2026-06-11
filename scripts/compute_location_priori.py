#!/usr/bin/env python3
"""
Build lesion location priors (according to PI-RADS v2.1 sector map) from MRI and TPM CRF data and export lesion metadata.
"""

import argparse
from pathlib import Path


from promis_preprocess.config_loader import load_config
from promis_preprocess.location_priori import *



def main():
    print(40 * "=")
    print("Computing location priors according to PI-RADS v2.1 sector map")
    print(40 * "=")

    parser = argparse.ArgumentParser(description="Compute lesion location priors from CRF data")
    parser.add_argument(
        "--label-type",
        type=str,
        choices=["cspca", "pca"],
        default="cspca",
        help="Lesion label type (cspca: ISUP >= 2, pca: ISUP >= 1)",
    )
    parser.add_argument("--mri-crf", type=str, default=None, help="Path to MRI CRF Excel file")
    parser.add_argument("--tpm-crf-folder", type=str, default=None, help="Path to TPM CRF folder")
    parser.add_argument("--tpm-crf-summary", type=str, default=None, help="Path to TPM CRF summary Excel file")
    parser.add_argument("--metadata", type=str, default=None, help="Path to series metadata parquet file")
    parser.add_argument("--dicom-processed", type=str, default=None, help="Path to processed DICOM data")
    parser.add_argument("--zonal-masks", type=str, default=None, help="Path to zonal masks")
    parser.add_argument("--location-priori-mri", type=str, default=None, help="Path to MRI location prior output folder")
    parser.add_argument("--location-priori-tpm", type=str, default=None, help="Path to TPM location prior output folder")
    args = parser.parse_args()

    config = load_config()
    root = Path(config["paths"]["root"])

    if args.mri_crf is None:
        args.mri_crf = root / config["paths"]["mri_crf"]
    if args.tpm_crf_folder is None:
        args.tpm_crf_folder = root / config["paths"]["tpm_crf_folder"]
    if args.tpm_crf_summary is None:
        args.tpm_crf_summary = root / config["paths"]["tpm_crf_summary"]
    if args.metadata is None:
        args.metadata = root / config["paths"]["metadata"]
    if args.dicom_processed is None:
        args.dicom_processed = root / config["paths"]["dicom_processed"]
    if args.zonal_masks is None:
        args.zonal_masks = root / config["paths"]["zonal_masks"]
    if args.location_priori_mri is None:
        args.location_priori_mri = root / config["paths"][f"location_priori_{args.label_type}_mri"]
    if args.location_priori_tpm is None:
        args.location_priori_tpm = root / config["paths"][f"location_priori_{args.label_type}_tpm"]

    label_type = args.label_type

    if label_type == "cspca":
        isup_thres = 2
    elif label_type == "pca":
        isup_thres = 1
    else:
        raise ValueError(f"Invalid label type: {label_type}")

    print(f"\nLabel type: {label_type} (ISUP threshold: {isup_thres})")
    print(f"MRI CRF path: {args.mri_crf}")
    print(f"TPM CRF folder: {args.tpm_crf_folder}")
    print(f"Metadata path: {args.metadata}")
    print(f"DICOM processed path: {args.dicom_processed}")
    print(f"Location prior MRI output: {args.location_priori_mri}")
    print(f"Location prior TPM output: {args.location_priori_tpm}")

    print("\n>>> Loading MRI CRF...")
    mri_crf = load_mri_crf(args.mri_crf)
    mri_crf_patient = build_mri_crf_patient(mri_crf, config)

    print("\n>>> Loading TPM CRF...")
    tpm_crf = load_tpm_crf(args.tpm_crf_folder, config, isup_thres, label_type)
    add_tpm_lesion_locations(
        mri_crf_patient,
        tpm_crf,
        config["barzell_to_pirads_map"],
        isup_thres,
        label_type,
    )

    print("\n>>> Loading series metadata...")
    series_metadata = load_series_metadata(args.metadata, config)

    print("\n>>> Loading TPM summary CRF...")
    tpm_crf_summary = load_tpm_crf_summary(args.tpm_crf_summary, series_metadata, isup_thres, label_type)

    print("\n>>> Merging case labels...")
    mri_crf_patient = merge_case_labels(mri_crf_patient, series_metadata, tpm_crf_summary, label_type)

    print("\n>>> Computing location priors...")
    mri_crf_patient = compute_location_priors(
        mri_crf_patient,
        args.dicom_processed,
        args.zonal_masks,
        args.location_priori_mri,
        args.location_priori_tpm,
        config,
    )

    print("\n>>> Saving lesion metadata...")
    save_lesion_metadata(mri_crf_patient, Path(args.metadata).parent, label_type)

    print("\n>>> Location priori computation completed successfully!")


if __name__ == "__main__":
    main()
