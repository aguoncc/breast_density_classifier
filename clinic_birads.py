import os

parent_folder = r"C:\Users\Data Science\Documents\MSDS - Breast Density Classifier\breast_density_classifier\images"

subfolders = [
    name for name in os.listdir(parent_folder)
    if os.path.isdir(os.path.join(parent_folder, name))
]

print("Number of subfolders:", len(subfolders))