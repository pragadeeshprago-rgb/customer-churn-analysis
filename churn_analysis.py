"""
Data Quality and Exploratory Analysis — Customer Churn
Dataset included with this project is synthetic and for practice only.
Run: python churn_analysis.py
Dependencies: pandas, numpy, matplotlib
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "customer_churn_raw.csv"
CLEAN = ROOT / "customer_churn_cleaned.csv"
PROFILE = ROOT / "data_quality_profile.csv"
SUMMARY = ROOT / "analysis_summary.txt"

raw = pd.read_csv(RAW)
df = raw.copy()
rows_before, cols_before = df.shape

# 1) Profile the raw data: column names, data types, missing values, duplicate rows.
raw_profile = pd.DataFrame({
    "column": raw.columns,
    "dtype": [str(raw[c].dtype) for c in raw.columns],
    "missing_count": [int(raw[c].isna().sum()) for c in raw.columns],
    "missing_percent": [round(float(raw[c].isna().mean() * 100), 2) for c in raw.columns],
    "unique_values_including_na": [int(raw[c].nunique(dropna=False)) for c in raw.columns]
})
raw_profile["exact_duplicate_rows_in_dataset"] = int(raw.duplicated().sum())
raw_profile.to_csv(PROFILE, index=False)

# Outlier screening on numeric fields using the IQR rule (screening only, not automatic deletion).
numeric_raw = raw.select_dtypes(include="number")
outlier_lines = []
for col in numeric_raw.columns:
    s = pd.to_numeric(raw[col], errors="coerce").dropna()
    if len(s) == 0:
        continue
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    count = int(((s < low) | (s > high)).sum())
    outlier_lines.append((col, round(float(low), 2), round(float(high), 2), count))

# 2) Clean with explicit, documented assumptions.
# Trim spaces, normalize blank strings, standardize known category variants.
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype("string").str.strip()
    df[col] = df[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

fixes = {
    "Gender": {"female": "Female", "male": "Male"},
    "Contract": {"month-to-month": "Month-to-month"},
    "InternetService": {"Fiber Optic": "Fiber optic", "fiber optic": "Fiber optic"},
    "PaymentMethod": {"Electronic check": "Electronic check"}
}
for col, mapping in fixes.items():
    df[col] = df[col].replace(mapping)

numeric_cols = ["SeniorCitizen", "TenureMonths", "MonthlyCharges", "TotalCharges"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

duplicates_removed = int(df.duplicated().sum())
df = df.drop_duplicates().copy()

# Domain/range assumptions: SeniorCitizen is binary 0/1; tenure cannot be negative;
# charges cannot be negative. IQR flags are reported but not automatically removed,
# because high charges can be valid customer values.
df.loc[~df["SeniorCitizen"].isin([0, 1]), "SeniorCitizen"] = np.nan
df.loc[df["TenureMonths"] < 0, "TenureMonths"] = np.nan
df.loc[df["MonthlyCharges"] < 0, "MonthlyCharges"] = np.nan
df.loc[df["TotalCharges"] < 0, "TotalCharges"] = np.nan

missing_before_imputation = df.isna().sum()
for col in numeric_cols:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())
for col in df.select_dtypes(include=["object", "string"]).columns:
    if df[col].isna().any():
        mode = df[col].mode(dropna=True)
        if not mode.empty:
            df[col] = df[col].fillna(mode.iloc[0])

# Validation checks after cleaning.
df["TenureMonths"] = df["TenureMonths"].round().clip(0, 100).astype(int)
df["SeniorCitizen"] = df["SeniorCitizen"].round().clip(0, 1).astype(int)
df["MonthlyCharges"] = df["MonthlyCharges"].round(2)
df["TotalCharges"] = df["TotalCharges"].round(2)
assert df["CustomerID"].notna().all(), "CustomerID contains missing values"
assert not df.duplicated().any(), "Duplicate rows remain"
assert df.isna().sum().sum() == 0, "Missing values remain after imputation"
assert df["SeniorCitizen"].isin([0, 1]).all(), "SeniorCitizen must be 0 or 1"
assert (df["TenureMonths"] >= 0).all(), "Negative tenure remains"
assert (df["MonthlyCharges"] >= 0).all(), "Negative monthly charges remain"
assert (df["TotalCharges"] >= 0).all(), "Negative total charges remain"
assert set(df["Churn"].dropna().unique()).issubset({"Yes", "No"}), "Unexpected churn label"
df.to_csv(CLEAN, index=False)

# 3) Summary statistics and meaningful distributions.
churn_rate = float((df["Churn"] == "Yes").mean() * 100)
churn_by_contract = df.groupby("Contract")["Churn"].apply(lambda x: (x == "Yes").mean() * 100).sort_values(ascending=False)
tenure_by_churn = df.groupby("Churn")["TenureMonths"].agg(["count", "mean", "median", "min", "max"])
charges_by_churn = df.groupby("Churn")["MonthlyCharges"].agg(["mean", "median", "min", "max"])
numeric_summary = df[["TenureMonths", "MonthlyCharges", "TotalCharges"]].describe().round(2)

# Charts
plt.figure(figsize=(6.5, 4))
df["Churn"].value_counts().reindex(["No", "Yes"]).plot(kind="bar")
plt.title("Customer Count by Churn Status")
plt.xlabel("Churn")
plt.ylabel("Customer count")
plt.tight_layout()
plt.savefig(ROOT / "churn_counts.png", dpi=150)
plt.close()

plt.figure(figsize=(7.5, 4))
churn_by_contract.sort_values().plot(kind="barh")
plt.title("Churn Rate by Contract Type")
plt.xlabel("Churn rate (%)")
plt.ylabel("Contract type")
plt.tight_layout()
plt.savefig(ROOT / "churn_by_contract.png", dpi=150)
plt.close()

plt.figure(figsize=(6.5, 4))
df.boxplot(column="TenureMonths", by="Churn")
plt.title("Tenure Distribution by Churn Status")
plt.suptitle("")
plt.xlabel("Churn")
plt.ylabel("Tenure (months)")
plt.tight_layout()
plt.savefig(ROOT / "tenure_by_churn.png", dpi=150)
plt.close()

with open(SUMMARY, "w", encoding="utf-8") as f:
    f.write("DATA QUALITY AND EXPLORATORY ANALYSIS — CUSTOMER CHURN\n")
    f.write("=" * 62 + "\n")
    f.write("DATA NOTE: Synthetic practice dataset; not real company data.\n\n")
    f.write("1. RAW DATA PROFILE\n")
    f.write(f"Rows before cleaning: {rows_before}\nColumns: {cols_before}\n")
    f.write(f"Exact duplicate rows detected: {int(raw.duplicated().sum())}\n")
    f.write("Per-column types, missing counts/percentages, and unique counts are in data_quality_profile.csv.\n\n")
    f.write("2. IQR OUTLIER SCREENING (flags only; not automatically deleted)\n")
    for col, low, high, count in outlier_lines:
        f.write(f"{col}: lower fence={low}, upper fence={high}, flagged={count}\n")
    f.write("\n3. CLEANING DECISIONS / ASSUMPTIONS\n")
    f.write("- Trim whitespace and treat blank strings as missing.\n")
    f.write("- Standardize known category variants (case/spelling/spacing).\n")
    f.write("- Convert numeric fields using safe numeric conversion.\n")
    f.write(f"- Remove {duplicates_removed} exact duplicate row(s).\n")
    f.write("- Treat SeniorCitizen values outside 0/1, negative tenure, and negative charges as invalid/missing.\n")
    f.write("- Fill numeric missing values with median; categorical missing values with mode.\n")
    f.write("- IQR outliers are reported but retained unless they violate explicit domain rules, since high values may be valid.\n")
    f.write(f"- Rows after cleaning: {len(df)}; missing cells after cleaning: {int(df.isna().sum().sum())}.\n")
    f.write("Validation checks passed: no missing cells, no duplicate rows, valid binary SeniorCitizen, nonnegative tenure/charges, valid churn labels.\n\n")
    f.write("4. SUMMARY STATISTICS\n")
    f.write(numeric_summary.to_string())
    f.write("\n\n5. CHURN PATTERNS\n")
    f.write(f"Overall churn rate: {churn_rate:.1f}%\n\n")
    f.write("Churn rate by contract (%):\n" + churn_by_contract.round(1).to_string())
    f.write("\n\nTenure by churn:\n" + tenure_by_churn.round(2).to_string())
    f.write("\n\nMonthly charges by churn:\n" + charges_by_churn.round(2).to_string())
    f.write("\n\n6. INTERPRETATION AND LIMITATIONS\n")
    f.write("In this synthetic sample, churn is highest for month-to-month contracts. This is a descriptive pattern only.\n")
    f.write("The data are synthetic and small; results are illustrative, not business-grade conclusions or causal evidence.\n")

print(f"Raw shape: {raw.shape}; duplicate rows: {int(raw.duplicated().sum())}")
print(f"Clean shape: {df.shape}; duplicate rows remaining: {int(df.duplicated().sum())}; missing cells remaining: {int(df.isna().sum().sum())}")
print(f"Overall churn rate: {churn_rate:.1f}%")
print("Churn rate by contract (%):")
print(churn_by_contract.round(1))
print("Outputs saved: customer_churn_cleaned.csv, data_quality_profile.csv, analysis_summary.txt, three PNG charts.")
