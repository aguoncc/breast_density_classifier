import os
import pandas as pd
import pydicom

summary_csv = r"C:\Users\Data Science\Documents\MSDS - Breast Density Classifier\breast_density_classifier\images\density_predictions_summary.csv"
dicom_folder = r"C:\Users\Data Science\Documents\MSDS - Breast Density Classifier\breast_density_classifier\dicom images\VinDr"
annotations_csv = r"C:\Users\Data Science\Downloads\breast-level_annotations.csv"

df_summary = pd.read_csv(summary_csv)
df_annotations = pd.read_csv(annotations_csv)

clinic_birads_list = []

for patient_id in df_summary["Patient_ID"]:
    # find first DICOM file starting with patient_id
    patient_files = [f for f in os.listdir(dicom_folder) if f.startswith(patient_id)]
    if not patient_files:
        print(f"No DICOM found for {patient_id}")
        clinic_birads_list.append(None)
        continue

    dicom_path = os.path.join(dicom_folder, patient_files[0])
    ds = pydicom.dcmread(dicom_path)
    study_id = ds.StudyInstanceUID

    row = df_annotations[df_annotations["study_id"] == study_id]
    if row.empty:
        print(f"No annotation found for study_id {study_id}")
        clinic_birads_list.append(None)
        continue

    # extract breast_density from first matching row
    clinic_birads = row.iloc[0]["breast_density"]
    clinic_birads_list.append(clinic_birads)

df_summary["clinic_density"] = clinic_birads_list
output_csv = summary_csv
df_summary.to_csv(output_csv, index=False)
print(f"Updated CSV saved to {output_csv}")