import sys
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
import pandas as pd
from sklearn.metrics import confusion_matrix

print("script running:", sys.executable) # checking due to import conflicts

file_path = r"C:\Users\Data Science\Downloads\Benign_Cases.csv"

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


############################
# libra density results
############################
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
    errors="coerce")

merged = merged.dropna(subset=["BREAST COMPOSITION", "density_pct"])
import statsmodels.api as sm
print("linear regression \n")

X = merged[["density_pct"]]
X = sm.add_constant(X)

# y = birads (ordinal treated as numeric)
y = merged["BREAST COMPOSITION"]
model = sm.OLS(y, X).fit()

print("linear regression \n")
print(model.summary())


# threshold part???
import numpy as np
from sklearn.metrics import cohen_kappa_score

# true labels
y_true = merged["BREAST COMPOSITION"].values
dens = merged["density_pct"].values

def classify_density(dens, t1, t2, t3):
    return np.digitize(dens, [t1, t2, t3]) + 1

best_kappa = -1
best_thresholds = None

t1_range = np.linspace(0, 15, 25)
t2_range = np.linspace(5, 35, 25)
t3_range = np.linspace(20, 70, 25)


for t1 in t1_range:
    for t2 in t2_range:
        if t2 <= t1:
            continue

        for t3 in t3_range:
            if t3 <= t2:
                continue

            y_pred = classify_density(dens, t1, t2, t3)
        
            counts = np.bincount(y_pred)[1:]  
            if len(counts) < 4:
                continue
            if np.min(counts) < 5:
                continue

            kappa = cohen_kappa_score(
                y_true,
                y_pred,
                weights="quadratic"
            )

            if kappa > best_kappa:
                best_kappa = kappa
                best_thresholds = (t1, t2, t3)


print("Best thresholds:", best_thresholds)
print("Best kappa:", best_kappa)

t1, t2, t3 = best_thresholds
merged["predicted_birads"] = classify_density(dens, t1, t2, t3)

print(confusion_matrix(
    merged["BREAST COMPOSITION"],
    merged["predicted_birads"]))

#############################
# apply thresholds
#############################

t1, t2, t3 = best_thresholds

def classify_density(dens, t1, t2, t3):
    return np.digitize(dens, [t1, t2, t3]) + 1

merged["predicted_birads"] = classify_density(
    merged["density_pct"].values,t1, t2, t3)

    
merged.to_csv(r"C:\Users\Data Science\Downloads\merged_results.csv", index=False)

y_true = merged["BREAST COMPOSITION"].values
y_pred = merged["predicted_birads"].values


#############################
# agreement kappa
#############################

kappa = cohen_kappa_score(
    y_true,
    y_pred,
    weights="quadratic")

print("\n========================")
print("FINAL RESULTS")
print("========================")
print("Best thresholds:", best_thresholds)
print("Quadratic weighted kappa:", round(kappa, 4))

# bootstrap ci for kappa
def bootstrap_kappa(y_true, y_pred, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot_scores = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue

        score = cohen_kappa_score(
            y_true[idx],
            y_pred[idx],
            weights="quadratic")
        boot_scores.append(score)

    return np.array(boot_scores)


boot_scores = bootstrap_kappa(y_true, y_pred)

ci_lower = np.percentile(boot_scores, 2.5)
ci_upper = np.percentile(boot_scores, 97.5)

print("\n95% Bootstrap CI for Quadratic Kappa:")
print(f"[{ci_lower:.4f}, {ci_upper:.4f}]")
print("Bootstrap mean:", round(np.mean(boot_scores), 4))


# conf matrix pt. 2 to check
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

labels = [0, 1, 2, 3]

cm = confusion_matrix(y_true, y_pred, labels=labels)
cm_df = pd.DataFrame(
    cm,
    index=[f"True {l}" for l in labels],
    columns=[f"Pred {l}" for l in labels])

print(cm_df)

# RUN IN TERMINAL !!!
# python "c:/Users/Data Science/Documents/MSDS - Breast Density Classifier/breast_density_classifier/benign csv cleaning.py" extract threshold parameters directly