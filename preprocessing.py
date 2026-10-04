import pandas as pd
from load_data import load_dataset
from outlier_fix import clip_outliers
COLUMNS=["Rainfall_mm","Temperature_C","Humidity_pct","Soil_Moisture_pct","Soil_pH","Nitrogen_kg_ha",
"Phosphorus_kg_ha","Potassium_kg_ha","Area_ha","Fertilizer_kg_ha","Pest_Level_pct","Disease_Severity_pct","NDVI"]
FEATURES=COLUMNS+["Irrigation","Rainfall_Temperature_Index","NPK_Total","Water_Stress_Index","Plant_Health_Index"]
def preprocess():
    df=clip_outliers(load_dataset(),COLUMNS)
    df["Rainfall_Temperature_Index"]=df["Rainfall_mm"]/(df["Temperature_C"]+1)
    df["NPK_Total"]=df["Nitrogen_kg_ha"]+df["Phosphorus_kg_ha"]+df["Potassium_kg_ha"]
    df["Water_Stress_Index"]=(100-df["Soil_Moisture_pct"]).clip(lower=0)
    df["Plant_Health_Index"]=df["NDVI"]*100-.35*df["Disease_Severity_pct"]
    df["Irrigation"]=df["Irrigation"].map({"No":0,"Yes":1})
    df["Disease_Risk"]=df["Disease_Risk"].map({"Low":0,"Medium":1,"High":2})
    df["Crop_Code"]=df["Crop_Type"].map({c:i for i,c in enumerate(["Rice","Wheat","Maize","Cotton","Groundnut"])})
    return df