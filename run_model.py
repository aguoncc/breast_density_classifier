import os
import subprocess
import pandas as pd
import numpy as np
import re
import torch

# --- SETTINGS ---
output_folder = r"C:\Users\Data Science\Documents\MSDS - Breast Density Classifier\breast_density_classifier\images"
script_path = r"C:\Users\Data Science\Documents\MSDS - Breast Density Classifier\breast_density_classifier\density_model_torch.py"
model_type = "cnn"
device_type = "gpu" if torch.cuda.is_available() else "cpu"

results = []

# --- LOOP THROUGH PATIENT FOLDERS ---
for patient_id in os.listdir(output_folder):
    patient_folder = os.path.join(output_folder, patient_id)
    if not os.path.isdir(patient_folder):
        continue

    cmd = [
        "python",
        script_path,
        model_type,
        "--image-path",
        patient_folder,
        "--device-type",
        device_type
    ]

    completed = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True 
    )

    # DEBUG: check output
    print(f"\n--- Patient {patient_id} ---")
    print("STDOUT:\n", completed.stdout)
    print("STDERR:\n", completed.stderr)

    # EXTRACT PROBABILITIES
    # Match numbers after colon or standalone decimals
    probs = re.findall(r":\s*([0-9]*\.[0-9]+)", completed.stdout)

    if len(probs) != 4:
        print(f"Skipping {patient_id}, could not find 4 probabilities. Found: {probs}")
        continue

    probs = list(map(float, probs))
    predicted_class = int(np.argmax(probs))
    birads_map = {0:"A", 1:"B", 2:"C", 3:"D"}

    results.append({
        "Patient_ID": patient_id,
        "P0_Almost_Fatty": probs[0],
        "P1_Scattered": probs[1],
        "P2_Heterogeneous": probs[2],
        "P3_Extremely_Dense": probs[3],
        "Predicted_Class": predicted_class,
        "Predicted_BIRADS": birads_map[predicted_class]
    })

# --- SAVE ONLY HIGHEST-PROBABILITY BIRADS PER PATIENT ---
if results:
    df = pd.DataFrame(results)

    # Keep only Patient_ID and Predicted_BIRADS
    df_summary = df[["Patient_ID", "Predicted_BIRADS"]]

    csv_path = os.path.join(output_folder, "density_predictions_summary.csv")
    df_summary.to_csv(csv_path, index=False)

    print(f"\nSaved highest-probability BIRADS per patient to {csv_path}")
else:
    print("No results to save!")