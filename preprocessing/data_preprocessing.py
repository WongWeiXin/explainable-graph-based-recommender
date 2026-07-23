# ==========================================
# 3.6 DATA PREPROCESSING
# Malaysia Property Dataset
# ==========================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("Malaysia_property_data.csv")

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

# Remove duplicate rows
df = df.drop_duplicates()

# Remove duplicate property listings
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
    "Malaysia_Property_Train.csv",
    index=False
)

validation_df.to_csv(
    "Malaysia_Property_Validation.csv",
    index=False
)

test_df.to_csv(
    "Malaysia_Property_Test.csv",
    index=False
)

print("\nPreprocessing Completed Successfully!")