import sys
import pandas as pd
import numpy as np
import pandas as pd

print("script is:", sys.executable) # import conflicts

file_path = r"C:\Users\Data Science\Downloads\Benign_Cases.csv"

# ordinal log test
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()
df["ID"] = df["ID"].fillna(method="ffill")
df["ID"] = df["ID"].astype(int).astype(str)

print(df.head())

rows = []

for _, row in df.iterrows():
    side = row["RIGHT_OR_LEFT"]

    if side == "BILATERAL":
        sides = ["R", "L"]
    elif side == "RIGHT":
        sides = ["R"]
    elif side == "LEFT":
        sides = ["L"]
    else:
        continue

    for s in sides:
        new_row = row.copy()
        new_row["File_ID"] = f"P{row['ID']}_{s}_{row['VIEW']}"
        rows.append(new_row)

df_expanded = pd.DataFrame(rows)
print(df_expanded["File_ID"].head(10))



libra = pd.read_csv(r"C:\Users\Data Science\Downloads\Libra Results\Density - Copy.csv")

libra.columns = libra.columns.str.strip()

libra = libra.rename(columns={
    "File Analyzed": "File_ID",
    "BreastDensity(%)": "density_pct"
})

# merging
merged = df_expanded.merge(libra, on="File_ID", how="inner")
print(merged.head(5))

merged["BREAST COMPOSITION"] = pd.to_numeric(
    merged["BREAST COMPOSITION"],
    errors="coerce"
)

merged = merged.dropna(subset=["BREAST COMPOSITION", "density_pct"])

from statsmodels.miscmodels.ordinal_model import OrderedModel

model = OrderedModel(
    merged["BREAST COMPOSITION"],
    merged[["density_pct"]],
    distr="logit"
)

result = model.fit(method="bfgs")
print(result.summary())
print("RUN PLSPLSPLSPLS")
print(result.params)

params = result.params
print(result.params)

print("\nMODEL PARAMETERS:")
print(result.params)

beta = result.params["density_pct"]
tau1 = result.params["1/2"]
tau2 = result.params["2/3"]
tau3 = result.params["3/4"]

print("\nTHRESHOLDS (latent scale):")
print("1/2:", tau1)
print("2/3:", tau2)
print("3/4:", tau3)


import numpy as np
import pandas as pd

# 1. Create smooth density range
density_grid = np.linspace(
    merged["density_pct"].min(),
    merged["density_pct"].max(),
    2000
)

print(f"densiy Grid {density_grid}")

exog_grid = pd.DataFrame({
    "density_pct": density_grid
})

# predict probabilities for each class
pred_probs = result.model.predict(
    result.params,
    exog=exog_grid
)

# convert probs to predicted class
pred_class1 = pred_probs.argmax(axis=1) + 1
print(f"Pred Class: {pred_class1}")
# locate change
change_points = np.where(np.diff(pred_class1) != 0)[0]
thresholds = density_grid[change_points]

print("\n==============================")
print("EMPIRICAL BI-RADS THRESHOLDS")
print("==============================")

for i, t in enumerate(thresholds, start=1):
    print(f"Cut {i}: {t:.2f}% density")

# RUN IN TERMINAL !!!
# python "c:/Users/Data Science/Documents/MSDS - Breast Density Classifier/breast_density_classifier/benign csv cleaning.py" extract threshold parameters directly