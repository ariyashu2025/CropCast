import pandas as pd
from load_data import load_dataset
from outlier_fix import clip_outliers

COLUMNS = ["Rainfall_mm", "Temperature_C", "Humidity_pct", "Soil_Moisture_pct", "Soil_pH", "Nitrogen_kg_ha",
           "Phosphorus_kg_ha", "Potassium_kg_ha", "Area_ha", "Fertilizer_kg_ha", "Pest_Level_pct",
           "Disease_Severity_pct", "NDVI"]
FEATURES = COLUMNS + ["Irrigation", "Rainfall_Temperature_Index", "NPK_Total", "Water_Stress_Index",
                      "Plant_Health_Index"]


def preprocess():
    raw = load_dataset()
    df = pd.DataFrame()

    df["Rainfall_mm"] = raw["Annual_Rainfall"] if "Annual_Rainfall" in raw.columns else 1000
    df["Temperature_C"] = raw["Temperature_C"] if "Temperature_C" in raw.columns else 26.0
    df["Humidity_pct"] = raw["Humidity_pct"] if "Humidity_pct" in raw.columns else 70.0
    df["Soil_Moisture_pct"] = raw["Soil_Moisture_pct"] if "Soil_Moisture_pct" in raw.columns else 60.0
    df["Soil_pH"] = raw["Soil_pH"] if "Soil_pH" in raw.columns else 6.5
    df["Nitrogen_kg_ha"] = raw["Nitrogen_kg_ha"] if "Nitrogen_kg_ha" in raw.columns else 120.0
    df["Phosphorus_kg_ha"] = raw["Phosphorus_kg_ha"] if "Phosphorus_kg_ha" in raw.columns else 60.0
    df["Potassium_kg_ha"] = raw["Potassium_kg_ha"] if "Potassium_kg_ha" in raw.columns else 40.0
    df["Area_ha"] = raw["Area"] if "Area" in raw.columns else 1.0
    df["Fertilizer_kg_ha"] = raw["Fertilizer"] if "Fertilizer" in raw.columns else 150.0
    df["Pest_Level_pct"] = raw["Pesticide"] if "Pesticide" in raw.columns else 15.0
    df["Disease_Severity_pct"] = raw["Disease_Severity_pct"] if "Disease_Severity_pct" in raw.columns else (
        raw["Pesticide"] * 0.1 if "Pesticide" in raw.columns else 10.0)
    df["NDVI"] = raw["NDVI"] if "NDVI" in raw.columns else 0.65

    # Target column must match 'Yield'
    df["Yield"] = raw["Yield"] if "Yield" in raw.columns else 2.5

    if "Irrigation" in raw.columns:
        df["Irrigation"] = raw["Irrigation"]
    else:
        df["Irrigation"] = "Yes"

    if "Crop" in raw.columns:
        df["Crop_Type"] = raw["Crop"]
    elif "Crop_Type" in raw.columns:
        df["Crop_Type"] = raw["Crop_Type"]
    else:
        df["Crop_Type"] = "Rice"

    if "Disease_Risk" in raw.columns:
        df["Disease_Risk"] = raw["Disease_Risk"]
    else:
        score = df["Pest_Level_pct"] + df["Disease_Severity_pct"]
        q33 = score.quantile(0.33)
        q66 = score.quantile(0.66)
        df["Disease_Risk"] = score.apply(lambda s: "High" if s >= q66 else ("Medium" if s >= q33 else "Low"))

    df = clip_outliers(df, COLUMNS)
    df["Rainfall_Temperature_Index"] = df["Rainfall_mm"] / (df["Temperature_C"] + 1)
    df["NPK_Total"] = df["Nitrogen_kg_ha"] + df["Phosphorus_kg_ha"] + df["Potassium_kg_ha"]
    df["Water_Stress_Index"] = (100 - df["Soil_Moisture_pct"]).clip(lower=0)
    df["Plant_Health_Index"] = df["NDVI"] * 100 - 0.35 * df["Disease_Severity_pct"]
    df["Irrigation"] = df["Irrigation"].map({"No": 0, "Yes": 1}).fillna(1)
    df["Disease_Risk"] = df["Disease_Risk"].map({"Low": 0, "Medium": 1, "High": 2}).fillna(0)

    crops = ["Rice", "Wheat", "Maize", "Cotton", "Groundnut"]
    df["Crop_Code"] = df["Crop_Type"].map({c: i for i, c in enumerate(crops)}).fillna(0)
    return df