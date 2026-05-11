# # PROBLEM: 
# # the model is failing to detect density on the VinDr images and features extraction logs density as 
# # 0.0 for all even after adding the metadata label SOPClassUID (Digital Mammography X-Ray Image 
# # Storage – for Presentation).

import os
import pydicom
import numpy as np

input_folder = r"C:\Users\Data Science\Documents\MSDS - Breast Density Classifier\breast_density_classifier\dicom images\lumina"
output_folder = r"C:\Users\Data Science\Documents\MSDS - Breast Density Classifier\breast_density_classifier\dicom images\lumina\edits"
os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):
    if file.lower().endswith((".dcm", ".dicom")):
        input_path = os.path.join(input_folder, file)
        try:
            ds = pydicom.dcmread(input_path)

            ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.1.2"
            ds.PixelSpacing = ds.ImagerPixelSpacing
            ds.Laterality = getattr(ds, 'ImageLaterality', getattr(ds, 'Laterality', 'U'))
            ds.ViewPosition = getattr(ds, 'ViewPosition', getattr(ds, 'PatientOrientation', 'U'))
 
            output_path = os.path.join(output_folder, file)
            ds.save_as(output_path, write_like_original=False)
            print(f"Processed: {file}")
        except Exception as e:
            print(f"Failed: {file} | {type(e).__name__}: {e}")

print("donezers! check the edits folder ;')")

parent_folder = r"C:\Users\Data Science\Documents\MSDS - Breast Density Classifier\breast_density_classifier\images"

num_folders = sum(
    os.path.isdir(os.path.join(parent_folder, item))
    for item in os.listdir(parent_folder))

print(f"Number of folders: {num_folders}")