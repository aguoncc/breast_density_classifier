import pydicom


import os
import numpy as np

folder = r"C:\Users\Data Science\Downloads"

required_tags = [
    # "SOPInstanceUID",
    # "StudyInstanceUID",
    # "SeriesInstanceUID",
    "SOPClassUID",
    "PresentationIntentType",
    # "Modality",
    "ImageLaterality",
    # "Laterality",
    # "ViewPosition",
    # "PatientOrientation",
    # "PhotometricInterpretation",
    # "BitsStored",
    # "BitsAllocated",
    # "PixelSpacing",
    # "Rows",
    # "Columns",
    # "RescaleSlope",
    # "RescaleIntercept",
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
