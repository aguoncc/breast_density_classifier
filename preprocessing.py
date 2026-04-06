import os
import pydicom
import cv2

input_folder = r"C:\Users\Data Science\Documents\MSDS - Breast Density Classifier\breast_density_classifier\dicom images\VinDr"
output_folder = r"C:\Users\Data Science\Documents\MSDS - Breast Density Classifier\breast_density_classifier\images"

target_size = (2600, 2000)

os.makedirs(output_folder, exist_ok=True)


def get_laterality_and_view(ds):
    laterality = getattr(ds, "ImageLaterality", None)
    view_position = getattr(ds, "ViewPosition", None)

    if laterality in ["L", "R"] and view_position in ["CC", "MLO"]:
        return laterality, view_position

    return None, None


def preprocess_image(ds):
    img = ds.pixel_array
    img = cv2.resize(img, target_size)
    return img


def get_patient_id(ds, filename):
    pid = getattr(ds, "PatientID", None)

    if pid:
        return pid

    return filename.split("_")[0]


for filename in os.listdir(input_folder):

    if not filename.lower().endswith((".dcm", ".dicom")):
        continue

    filepath = os.path.join(input_folder, filename)
    ds = pydicom.dcmread(filepath)

    patient_id = get_patient_id(ds, filename)
    laterality, view = get_laterality_and_view(ds)

    if laterality is None or view is None:
        print(f"Skipping unknown view: {filename}")
        continue

    patient_folder = os.path.join(output_folder, patient_id)
    os.makedirs(patient_folder, exist_ok=True)

    save_filename = f"{laterality}-{view}.png"
    save_path = os.path.join(patient_folder, save_filename)

    cv2.imwrite(save_path, preprocess_image(ds))

    print(f"Saved {patient_id}/{save_filename}")