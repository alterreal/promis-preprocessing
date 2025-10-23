"""
Core DICOM processing functions.
"""

import os
import SimpleITK as sitk
from tqdm import tqdm
from datetime import datetime

# disable SimpleITK warnings
sitk.ProcessObject_GlobalWarningDisplayOff()


def create_dicom_reader():
    """Create and configure a SimpleITK ImageSeriesReader."""
    reader = sitk.ImageSeriesReader()
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    return reader
    
def load_dicom_image_from_folder(reader, folder_path):
    """
    Load a DICOM series from a folder path.
    """
    dicom_names = reader.GetGDCMSeriesFileNames(folder_path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    return image


def process_dicom_series(path, series_descriptions, base_path, log_file):
    """Process a single DICOM series and return metadata or None if failed."""
    from metadata_extractor import extract_metadata_from_reader
    
    reader = create_dicom_reader()
    
    try:
        image = load_dicom_image_from_folder(reader, path)
        
        # Extract metadata 
        metadata = extract_metadata_from_reader(reader, image, path, series_descriptions, base_path)
        
        # Check for slice count mismatch 
        num_files = len(os.listdir(path))
        num_slices = image.GetSize()[2]
        if num_files != num_slices:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(log_file, 'a') as f:
                f.write(f"[{timestamp}] WARNING: {path} contains {num_files} DICOM files, but loaded series has {num_slices} slices\n")
            return metadata, 'warning'
        
        return metadata, None
        
    except Exception as e:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, 'a') as f:
            f.write(f"[{timestamp}] ERROR: {path}: {str(e)}\n")
        return None, str(e)


def process_all_dicom_series(dicom_path, series_descriptions, output_path):
    """Process all DICOM series in the given path."""
    metadata = []
    stats = {'processed': 0, 'errors': 0, 'warnings': 0}
    
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    # Create log file
    log_file = os.path.join(output_path, 'processing_log.txt')
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, 'w') as f:
        f.write(f"DICOM Processing Log - Started at {timestamp}\n")
        f.write("=" * 50 + "\n\n")
    
    # Get all DICOM directories that contain .dcm files
    dicom_dirs = []
    for root, dirs, files in os.walk(dicom_path):
        if any(f.endswith('.dcm') for f in files):
            dicom_dirs.append(root)
    
    total_dirs = len(dicom_dirs)
    
    print(f"Found {total_dirs} DICOM directories to process...")
    print(f"Logging to: {log_file}")
    
    with tqdm(total=total_dirs, desc="Processing DICOM", unit="dir") as pbar:
        for path in dicom_dirs:
            result, error = process_dicom_series(path, series_descriptions, dicom_path, log_file)
            
            if result is not None:
                metadata.append(result)
                stats['processed'] += 1
                if error == 'warning':
                    stats['warnings'] += 1
            else:
                stats['errors'] += 1
            
            # Update progress bar with current stats
            pbar.set_postfix({
                'Processed': stats['processed'],
                'Errors': stats['errors'],
                'Warnings': stats['warnings']
            })
            pbar.update(1)
    
    # Write final summary to log
    end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, 'a') as f:
        f.write(f"\nProcessing completed at {end_timestamp}\n")
        f.write(f"Final stats: {stats['processed']} processed, {stats['errors']} errors, {stats['warnings']} warnings\n")
    
    return metadata, stats, log_file
