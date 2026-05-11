import pydicom


import os
import numpy as np

folder = r"C:\Users\Data Science\Documents\MSDS - Breast Density Classifier\breast_density_classifier\dicom images\lumina"

def check_metadata_flags(ds):
    flags = []

    # Implant flag (best case scenario)
    implant = ds.get("BreastImplantPresent", None)
    if implant == "YES":
        flags.append("BREAST_IMPLANT")

    # Missing compression force (suspicious)
    if not hasattr(ds, "CompressionForce"):
        flags.append("NO_COMPRESSION_FORCE")

    # Extremely low or high thickness (heuristic)
    thickness = ds.get("BodyPartThickness", None)
    if thickness:
        try:
            if float(thickness) < 10 or float(thickness) > 120:
                flags.append("ABNORMAL_THICKNESS")
        except:
            pass

    return flags

required_tags = [
    "SOPInstanceUID",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPClassUID",
    "PresentationIntentType",
    "Modality",
    "ImageLaterality",
    # "Laterality",
    # "ViewPosition",
    # "PatientOrientation",
    # "PhotometricInterpretation",
    # "BitsStored",
    # "BitsAllocated",
    "ImagerPixelSpacing",
    # "Rows",
    # "Columns",
    # "RescaleSlope",
    "BodyPartThickness",
    "Manufacturer"
]

for file in os.listdir(folder):
    if not file.lower().endswith((".dcm", ".dicom")):
        continue

    path = os.path.join(folder, file)
    ds = pydicom.dcmread(path)

    tag_values = {}
    for tag in required_tags:
        tag_values[tag] = getattr(ds, tag, "N/A")  

    print(f"{file}: {tag_values}")

# nf = pydicom.dcmread(r"C:\Users\Data Science\Downloads\Attempt2\Deep-LIBRA2\image\Newfoundland\IM-0125-0001-0001.dcm")
# cb = pydicom.dcmread(r"C:\Users\Data Science\Documents\MSDS - Breast Density Classifier\breast_density_classifier\dicom images\VinDr\T2_L_CC.dcm")

# for t in required_tags:
#     print(t)
#     # print("Newfoundland:", getattr(nf, t, None))
#     print("VinDr:", getattr(cb, t, None))
