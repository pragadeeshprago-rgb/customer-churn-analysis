"""
Customer Churn Data Quality & Exploratory Analysis
Run: python churn_analysis.py
Requires: pandas, numpy, matplotlib
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RAW_FILE = ROOT / "customer_churn_raw.csv"
CLEAN_FILE = ROOT / "customer_churn_cleaned.csv"
REPORT_FILE = ROOT / "analysis_summary.txt"

df = pd.read_csv(RAW_FILE)
initial_rows = len(df)
initial_columns = len(df.columns)

# Standardize whitespace and blank strings.
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype("string").str.strip()
    df[col] = df[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

# Standardize known category variants.
category_fixes = {
    "Gender": {"female": "Female", "male": "Male"},
    "Contract": {"month-to-month": "Month-to-month", "one year": "One year", "two year": "Two year"},
    "InternetService": {"Fiber Optic": "Fiber optic", "fiber optic": "Fiber optic"},
    "PaymentMethod": {"Electronic check": "Electronic check"}
}
for col, mapping in category_fixes.items():
    df[col] = df[col].replace(mapping)

# Convert numeric columns safely.
numeric_cols = ["SeniorCitizen", "TenureMonths", "MonthlyCharges", "TotalCharges"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove exact duplicate rows.
duplicate_count = int(df.duplicated().sum())
df = df.drop_duplicates().copy()

# Flag invalid values as missing; preserve the original values in the raw file.
df.loc[~df["SeniorCitizen"].isin([0, 1]), "SeniorCitizen"] = np.nan
df.loc[df["TenureMonths"] < 0, "TenureMonths"] = np.nan
df.loc[df["MonthlyCharges"] < 0, "MonthlyCharges"] = np.nan
df.loc[df["TotalCharges"] < 0, "TotalCharges"] = np.nan

# Impute missing numeric values with median; categorical values with mode.
# This is a simple demonstration strategy; a production project should validate it with a domain owner.
missing_before = df.isna().sum()
for col in numeric_cols:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())
for col in df.select_dtypes(include=["object", "string"]).columns:
    if df[col].isna().any():
        mode = df[col].mode(dropna=True)
        if not mode.empty:
            df[col] = df[col].fillna(mode.iloc[0])

# Validate ranges and required fields.
df["TenureMonths"] = df["TenureMonths"].round().clip(0, 100).astype(int)
df["SeniorCitizen"] = df["SeniorCitizen"].round().clip(0, 1).astype(int)
df["MonthlyCharges"] = df["MonthlyCharges"].round(2)
df["TotalCharges"] = df["TotalCharges"].round(2)

# Save cleaned dataset.
df.to_csv(CLEAN_FILE, index=False)

# EDA metrics
churn_map = {"Yes": 1, "No": 0}
churn_rate = df["Churn"].map(churn_map).mean() * 100
contract_churn = (df.groupby("Contract")["Churn"]
                    .apply(lambda s: (s == "Yes").mean() * 100)
                    .sort_values(ascending=False))
tenure_summary = df.groupby("Churn")["TenureMonths"].agg(["count", "mean", "median"])
monthly_summary = df.groupby("Churn")["MonthlyCharges"].agg(["mean", "median"])
missing_after = int(df.isna().sum().sum())

# Charts
plt.figure(figsize=(7, 4))
df["Churn"].value_counts().reindex(["No", "Yes"]).plot(kind="bar")
plt.title("Customer Churn Counts")
plt.xlabel("Churn")
plt.ylabel("Number of customers")
plt.tight_layout()
plt.savefig(ROOT / "churn_counts.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 4))
contract_churn.sort_values().plot(kind="barh")
plt.title("Churn Rate by Contract Type")
plt.xlabel("Churn rate (%)")
plt.ylabel("Contract type")
plt.tight_layout()
plt.savefig(ROOT / "churn_by_contract.png", dpi=150)
plt.close()

plt.figure(figsize=(7, 4))
df.boxplot(column="TenureMonths", by="Churn")
plt.title("Tenure by Churn Status")
plt.suptitle("")
plt.xlabel("Churn")
plt.ylabel("Tenure (months)")
plt.tight_layout()
plt.savefig(ROOT / "tenure_by_churn.png", dpi=150)
plt.close()

# Write transparent report
with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("CUSTOMER CHURN DATA QUALITY & EXPLORATORY ANALYSIS\\n")
    f.write("=" * 58 + "\\n\\n")
    f.write("Dataset note: This project uses a synthetic practice dataset generated for demonstration.\\n")
    f.write(f"Raw rows: {initial_rows}\\n")
    f.write(f"Columns: {initial_columns}\\n")
    f.write(f"Exact duplicate rows removed: {duplicate_count}\\n")
    f.write("Cleaning decisions:\\n")
    f.write("- Trimmed whitespace and converted blank-like strings to missing values.\\n")
    f.write("- Standardized known category spelling/case variants.\\n")
    f.write("- Converted numeric columns to numeric types.\\n")
    f.write("- Marked invalid binary, negative tenure, and negative charge values as missing.\\n")
    f.write("- Filled numeric missing values with median and categorical missing values with mode.\\n")
    f.write(f"Rows after deduplication: {len(df)}\\n")
    f.write(f"Missing cells after cleaning: {missing_after}\\n\\n")
    f.write(f"Overall churn rate: {churn_rate:.1f}%\\n\\n")
    f.write("Churn rate by contract type (%):\\n")
    f.write(contract_churn.round(1).to_string())
    f.write("\\n\\nTenure summary by churn status:\\n")
    f.write(tenure_summary.round(2).to_string())
    f.write("\\n\\nMonthly charges by churn status:\\n")
    f.write(monthly_summary.round(2).to_string())
    f.write("\\n\\nInterpretation note: This is a small synthetic dataset; patterns are illustrative, not business-grade conclusions.\\n")

print(f"Loaded {initial_rows} raw rows and {initial_columns} columns.")
print(f"Removed {duplicate_count} exact duplicate rows; cleaned dataset has {len(df)} rows.")
print(f"Overall churn rate: {churn_rate:.1f}%")
print("Churn rate by contract type (%):")
print(contract_churn.round(1))
print(f"Saved cleaned dataset to: {CLEAN_FILE}")
print(f"Saved report to: {REPORT_FILE}")
