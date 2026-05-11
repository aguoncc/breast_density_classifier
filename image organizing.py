import os
import shutil

input_root = r"C:\Users\Data Science\Downloads\archive\LUMINA_RAW\Benign"
output_dir = r"C:\Users\Data Science\Documents\MSDS - Breast Density Classifier\breast_density_classifier\dicom images\lumina"

os.makedirs(output_dir, exist_ok=True)

# required views per patient
required_views = {"R_CC", "R_MLO", "L_CC", "L_MLO"}

for patient_folder in os.listdir(input_root):
    patient_path = os.path.join(input_root, patient_folder)

    if not os.path.isdir(patient_path):
        continue

    # collect available views in this patient folder
    available_views = set()

    for filename in os.listdir(patient_path):
        name, ext = os.path.splitext(filename)

        # assume naming like R_CC.dcm or R_MLO.dcm
        available_views.add(name)

    # check completeness
    if not required_views.issubset(available_views):
        print(f"Skipping patient {patient_folder} (missing views: {required_views - available_views})")
        continue

    # if complete → copy files
    patient_id = f"P{patient_folder}"

    for filename in os.listdir(patient_path):
        file_path = os.path.join(patient_path, filename)

        if not os.path.isfile(file_path):
            continue

        name, ext = os.path.splitext(filename)

        new_name = f"{patient_id}_{name}{ext}"
        dest_path = os.path.join(output_dir, new_name)

        shutil.copy2(file_path, dest_path)

print("Done: only complete patients copied.")