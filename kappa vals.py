import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score

pred_path = r"C:\Users\Data Science\Documents\MSDS - Breast Density Classifier\breast_density_classifier\images\density_predictions.csv"
orig_path = r"C:\Users\Data Science\Downloads\Benign_Cases.csv"
# all kappas
libra_path = r"C:\Users\Data Science\Downloads\Libra Results\Density - Copy.csv"

# loading
df = pd.read_csv(pred_path)
orig = pd.read_csv(orig_path)
df_libra = pd.read_csv(libra_path)

df.columns = df.columns.str.strip()
orig.columns = orig.columns.str.strip()
df_libra.columns = df_libra.columns.str.strip()


df["ID_num"] = df["Patient_ID"].str.extract(r"P(\d+)").astype(int)
orig["ID"] = orig["ID"].ffill().astype(int)

# take first density per patient in original csv
orig_first = (
    orig.groupby("ID", as_index=False)
        .first()[["ID", "BREAST COMPOSITION"]])


# merging
merged = df.merge(
    orig_first,
    left_on="ID_num",
    right_on="ID",
    how="left"
)
print(merged.head(3))


# cleaning the merged df
merged = merged.dropna(subset=["BREAST COMPOSITION"])

merged["BREAST COMPOSITION"] = merged["BREAST COMPOSITION"].astype(int)
merged["CNN_Pred_Class"] = pd.to_numeric(merged["CNN_Pred_Class"], errors="coerce")
merged["HIST_Pred_Class"] = pd.to_numeric(merged["HIST_Pred_Class"], errors="coerce")

merged = merged.dropna(subset=["CNN_Pred_Class", "HIST_Pred_Class"])

merged["CNN_Pred_Class"] = merged["CNN_Pred_Class"].astype(int)
merged["HIST_Pred_Class"] = merged["HIST_Pred_Class"].astype(int)

print(merged.head(5))

# filter predict
df_libra["Patient_ID"] = (
    df_libra["File Analyzed"]
    .astype(str)
    .str.upper()
    .str.strip()
    .str.extract(r"(P\d+)", expand=False))

df_libra["Patient_ID"] = df_libra["Patient_ID"].astype(str).str.strip().str.upper()
merged["Patient_ID"] = merged["Patient_ID"].astype(str).str.strip().str.upper()

valid_ids = set(df_libra["Patient_ID"].dropna().unique())
print("Number of valid patients:", len(valid_ids))

df_filtered = merged[merged["Patient_ID"].isin(valid_ids)].copy()
print("Example merged IDs:")
print(merged["Patient_ID"].head(10).tolist())

print("\nExample LIBRA IDs:")
print(list(valid_ids)[:10])

print(df_filtered.head(5))
print(df_filtered.columns)


# kappa agreement
def kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")

def bootstrap_kappa(y_true, y_pred, n_boot=1000):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_true)

    scores = []

    for _ in range(n_boot):
        idx = np.random.randint(0, n, n)
        scores.append(
            cohen_kappa_score(
                y_true[idx],
                y_pred[idx],
                weights="quadratic"
            )
        )

    return np.percentile(scores, 2.5), np.percentile(scores, 97.5)


# nyu breast density kappas
y_true = df_filtered["BREAST COMPOSITION"]

cnn_k = kappa(y_true, df_filtered["CNN_Pred_Class"])
hist_k = kappa(y_true, df_filtered["HIST_Pred_Class"])

cnn_low, cnn_high = bootstrap_kappa(y_true, df_filtered["CNN_Pred_Class"])
hist_low, hist_high = bootstrap_kappa(y_true, df_filtered["HIST_Pred_Class"])


print("\n=== CNN vs ORIGINAL ===")
print(f"Kappa: {cnn_k:.4f}")
print(f"95% CI: [{cnn_low:.4f}, {cnn_high:.4f}]")

print("\n=== HIST vs ORIGINAL ===")
print(f"Kappa: {hist_k:.4f}")
print(f"95% CI: [{hist_low:.4f}, {hist_high:.4f}]")

print("\nSample size:", len(df_filtered))