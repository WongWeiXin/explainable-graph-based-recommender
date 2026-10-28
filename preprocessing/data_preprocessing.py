# ==========================================
# 3.6 DATA PREPROCESSING
# Malaysia Property Dataset
# ==========================================

import sys
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
AUDIT_DIR = PROJECT_ROOT / "outputs" / "data_quality"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from duplicate_handling import remove_near_duplicates, standardise_states  # noqa: E402

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv(DATASET_DIR / "Malaysia_property_data.csv")

print("Original Dataset Shape:", df.shape)

print(df.head())

# ==========================================
# Define Relevant Columns
# ==========================================

# Numerical attributes
numeric_cols = [
    "price",
    "built_up_area",
    "land_area",
    "bedroom",
    "bathroom",
    "car_parks",
    "latitude",
    "longitude",
    "population_density"
]

# Categorical attributes
categorical_cols = [
    "property_type",
    "furnishing",
    "tenure",
    "land_title",
    "lot_type",
    "title_type",
    "unit_type",
    "facing_direction",
    "occupancy",
    "area",
    "district",
    "state"
]

# ==========================================
# 3.6.1 DATA CLEANING
# ==========================================

print("\n===== DATA CLEANING =====")

# Stage 1: remove exact duplicate rows
df = df.drop_duplicates()

# Stage 2: remove repeated listing links
# original_link represents unique listing source

df = df.drop_duplicates(
    subset=["original_link"]
)

# Remove records with missing critical attributes
critical_columns = [
    "price",
    "property_type",
    "built_up_area",
    "bedroom",
    "bathroom",
    "area",
    "state",
    "latitude",
    "longitude"
]

df = df.dropna(
    subset=critical_columns
)

df = df.dropna(
    subset=critical_columns
)

# Remove invalid numerical values
invalid_check_columns = [
    "price",
    "built_up_area",
    "bedroom",
    "bathroom"
]

for col in invalid_check_columns:

    df = df[
        df[col] > 0
    ]

# Reset index
df = df.reset_index(drop=True)

print("After Cleaning:", df.shape)

# ==========================================
# SAVE CLEANED RAW DATA
# ==========================================

df.to_csv(
    DATASET_DIR / "Malaysia_Property_Cleaned_Raw.csv",
    index=False
)

# ==========================================
# 3.6.2 DATA TRANSFORMATION
# ==========================================

print("\n===== DATA TRANSFORMATION =====")

# -------------------------------
# Handle Missing Numerical Values
# -------------------------------

for col in numeric_cols:

    if col in df.columns:

        df[col] = df[col].fillna(
            df[col].median()
        )

# -------------------------------
# Handle Missing Categorical Values
# -------------------------------

for col in categorical_cols:

    if col in df.columns:

        df[col] = df[col].fillna(
            df[col].mode()[0]
        )

# ==========================================
# 3.6.3 DATA INDEXING
# ==========================================

print("\n===== DATA INDEXING =====")

# Create unique property identifier
df.insert(
    0,
    "Property_ID",
    range(len(df))
)

print(df[["Property_ID"]].head())

# ==========================================
# 3.6.3a STATE NAMES + NEAR-DUPLICATES
# Runs after Property_ID (kept rows keep their ID)
# and before splitting (no copies across partitions)
# ==========================================

print("\n===== STATE STANDARDISATION =====")

df, state_changes = standardise_states(df)

for change, rows in state_changes.items():
    print(f"  {change}: {rows}")

print("\n===== STAGE 3: NEAR-DUPLICATE HANDLING =====")

df, duplicate_summary = remove_near_duplicates(df, audit_dir=AUDIT_DIR)

print("Candidate pairs:", duplicate_summary["candidate_pairs"])
print("Confirmed duplicate groups:", duplicate_summary["duplicate_groups"])
print("Rows removed:", duplicate_summary["rows_removed"])
print("Unconfirmed pairs kept:", duplicate_summary["unconfirmed_pairs_kept"])
print("After Near-Duplicate Handling:", df.shape)
print("Audit files:", AUDIT_DIR)

# ==========================================
# SAVE CLEANED RAW DATA
# ==========================================

df.to_csv(
    DATASET_DIR / "Malaysia_Property_Cleaned_Raw.csv",
    index=False
)

# ==========================================
# 3.6.4 DATA SPLITTING
# ==========================================

print("\n===== DATA SPLITTING =====")
# 70% Training
# 15% Validation
# 15% Testing

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42
)

validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42
)

print(
    "Training:",
    train_df.shape
)

print(
    "Validation:",
    validation_df.shape
)

print(
    "Testing:",
    test_df.shape
)

# ==========================================
# Min-Max Normalisation
# Fit ONLY on Training Data
# ==========================================

print("\n===== MIN-MAX NORMALISATION =====")

scaler = MinMaxScaler()

available_numeric = [
    col for col in numeric_cols
    if col in df.columns
]

# Fit scaler using training data only
train_df[available_numeric] = scaler.fit_transform(
    train_df[available_numeric]
)

# Apply same transformation
validation_df[available_numeric] = scaler.transform(
    validation_df[available_numeric]
)

test_df[available_numeric] = scaler.transform(
    test_df[available_numeric]
)

# ==========================================
# Label Encoding
# Fit ONLY on Training Data
# ==========================================

print("\n===== LABEL ENCODING =====")

label_encoders = {}

available_categories = [
    col for col in categorical_cols
    if col in df.columns
]

for col in available_categories:

    encoder = LabelEncoder()

    # Training data learning
    train_df[col] = encoder.fit_transform(
        train_df[col].astype(str)
    )

    # Create mapping
    mapping = dict(
        zip(
            encoder.classes_,
            encoder.transform(
                encoder.classes_
            )
        )
    )

    # Validation transformation
    validation_df[col] = (
        validation_df[col]
        .astype(str)
        .map(mapping)
        .fillna(-1)
        .astype(int)
    )

    # Testing transformation
    test_df[col] = (
        test_df[col]
        .astype(str)
        .map(mapping)
        .fillna(-1)
        .astype(int)
    )

    label_encoders[col] = encoder

# ==========================================
# 3.6.5 FEATURE PREPARATION
# ==========================================

print("\n===== FEATURE PREPARATION =====")

feature_columns = [

    # Property characteristics

    "price",
    "built_up_area",
    "land_area",
    "bedroom",
    "bathroom",
    "car_parks",


    # Property category

    "property_type",
    "furnishing",
    "tenure",
    "land_title",


    # Location

    "area",
    "district",
    "state",
    "latitude",
    "longitude",
    "postcode",


    # Neighbourhood features

    "train_station_count_3km",
    "shopping_mall_count_3km",
    "supermarket_count_3km",
    "school_count_3km",
    "hospital_count_3km",
    "park_count_1km",
    "office_count_3km",
    "university_count_3km",
    "population_density",

    # Original
    "original_link"

]

# Keep only existing columns
feature_columns = [
    col for col in feature_columns
    if col in train_df.columns
]

X_train = train_df[feature_columns]

X_validation = validation_df[feature_columns]

X_test = test_df[feature_columns]

print(
    "Selected Features:"
)

print(feature_columns)

# ==========================================
# Save Processed Dataset
# ==========================================

train_df.to_csv(
    DATASET_DIR / "Malaysia_Property_Train.csv",
    index=False
)

validation_df.to_csv(
    DATASET_DIR / "Malaysia_Property_Validation.csv",
    index=False
)

test_df.to_csv(
    DATASET_DIR / "Malaysia_Property_Test.csv",
    index=False
)

print("\nPreprocessing Completed Successfully!")