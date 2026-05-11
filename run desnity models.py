import os
import subprocess
import pandas as pd
import numpy as np
import re
import torch

# define use
output_folder = r"C:\Users\Data Science\Documents\MSDS - Breast Density Classifier\breast_density_classifier\images"
script_path = r"C:\Users\Data Science\Documents\MSDS - Breast Density Classifier\breast_density_classifier\density_model_torch.py"
cnn_model = "cnn"
hist_model = "histogram"
device_type = "gpu" if torch.cuda.is_available() else "cpu"

results = []

# loop thru patient folders
for patient_id in os.listdir(output_folder):
    patient_folder = os.path.join(output_folder, patient_id)
    if not os.path.isdir(patient_folder):
        continue

    cnn_cmd = [
        "python",
        script_path,
        cnn_model,
        "--image-path",
        patient_folder,
        "--device-type",
        device_type]

    cnn_completed = subprocess.run(
        cnn_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True)

    print(f"\n--- CNN Patient {patient_id} ---")
    print("STDOUT:\n", cnn_completed.stdout)
    print("STDERR:\n", cnn_completed.stderr)
    cnn_probs = re.findall(r":\s*([0-9]*\.[0-9]+)", cnn_completed.stdout)

    if len(cnn_probs) != 4:
        print(f"Skipping CNN for {patient_id}")
        continue

    cnn_probs = list(map(float, cnn_probs))
    cnn_pred = int(np.argmax(cnn_probs))


    hist_cmd = [
        "python",
        script_path,
        hist_model,
        "--image-path",
        patient_folder,
        "--device-type",
        device_type]

    hist_completed = subprocess.run(
        hist_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True)

    print(f"\n--- HISTOGRAM Patient {patient_id} ---")
    print("STDOUT:\n", hist_completed.stdout)
    print("STDERR:\n", hist_completed.stderr)
    hist_probs = re.findall(r":\s*([0-9]*\.[0-9]+)", hist_completed.stdout)
    
    if len(hist_probs) != 4:
        print(f"Skipping Histogram for {patient_id}")
        continue

    hist_probs = list(map(float, hist_probs))
    hist_pred = int(np.argmax(hist_probs))

    birads_map = {0:"A", 1:"B", 2:"C", 3:"D"}

    results.append({
        "Patient_ID": patient_id,

        # CNN outputs
        "CNN_P0_Fatty": cnn_probs[0],
        "CNN_P1_Scattered": cnn_probs[1],
        "CNN_P2_Hetero": cnn_probs[2],
        "CNN_P3_Extreme": cnn_probs[3],
        "CNN_Pred_Class": cnn_pred,
        "CNN_BIRADS": birads_map[cnn_pred],

        # Histogram outputs
        "HIST_P0_Fatty": hist_probs[0],
        "HIST_P1_Scattered": hist_probs[1],
        "HIST_P2_Hetero": hist_probs[2],
        "HIST_P3_Extreme": hist_probs[3],
        "HIST_Pred_Class": hist_pred,
        "HIST_BIRADS": birads_map[hist_pred]   
        })

df_results = pd.DataFrame(results)
csv_output_path = os.path.join(output_folder, "density_predictions.csv")
df_results.to_csv(csv_output_path, index=False)

print(f"\nResults saved to: {csv_output_path}")