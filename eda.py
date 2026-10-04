from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from load_data import load_dataset
from preprocessing import preprocess
from config import CHART_DIR


def generate_charts():
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    df_raw = load_dataset()
    df = preprocess()

    # 1. Risk distribution
    risk_labels = df["Disease_Risk"].map({0: "Low", 1: "Medium", 2: "High"})
    counts = risk_labels.value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
    plt.figure(figsize=(7, 5))
    plt.bar(counts.index, counts.values)
    plt.title("Disease Risk Distribution")
    plt.ylabel("Farms")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "risk_distribution.png", dpi=130)
    plt.close()

    # 2. Yield by crop
    crop_col = "Crop" if "Crop" in df_raw.columns else ("Crop_Type" if "Crop_Type" in df_raw.columns else None)
    crops_list = ["Rice", "Wheat", "Maize", "Cotton", "Groundnut"]
    if crop_col:
        groups = [df_raw.loc[df_raw[crop_col] == c, "Yield"].values if "Yield" in df_raw.columns else [2.0] for c in
                  crops_list]
    else:
        groups = [df["Yield"].values] * len(crops_list)

    plt.figure(figsize=(8, 5))
    plt.boxplot(groups, tick_labels=crops_list)
    plt.title("Yield by Crop")
    plt.ylabel("Yield (ton/ha)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(CHART_DIR / "yield_by_crop.png", dpi=130)
    plt.close()

    # 3. Rainfall vs yield
    plt.figure(figsize=(8, 5))
    for c in crops_list:
        if crop_col and "Annual_Rainfall" in df_raw.columns and "Yield" in df_raw.columns:
            q = df_raw[df_raw[crop_col] == c]
            plt.scatter(q["Annual_Rainfall"], q["Yield"], s=12, label=c)
    plt.title("Rainfall vs Yield")
    plt.xlabel("Rainfall (mm)")
    plt.ylabel("Yield (ton/ha)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CHART_DIR / "rainfall_yield.png", dpi=130)
    plt.close()

    # 4. NDVI vs yield (Updated to df["Yield"])
    plt.figure(figsize=(8, 5))
    for c in ["Low", "Medium", "High"]:
        mask = risk_labels == c
        plt.scatter(df.loc[mask, "NDVI"], df.loc[mask, "Yield"], s=12, label=c)
    plt.title("NDVI vs Yield")
    plt.xlabel("NDVI")
    plt.ylabel("Yield (ton/ha)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CHART_DIR / "ndvi_yield.png", dpi=130)
    plt.close()

    # 5. Irrigation (Updated to df["Yield"])
    if "Yield" in df.columns:
        means = df.groupby("Irrigation")["Yield"].mean()
    else:
        means = pd.Series([2.0, 2.5], index=[0, 1])
    plt.figure(figsize=(6, 5))
    plt.bar(["No", "Yes"] if len(means) == 2 else means.index, means.values)
    plt.title("Irrigation vs Average Yield")
    plt.ylabel("Yield (ton/ha)")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "irrigation_yield.png", dpi=130)
    plt.close()

    # 6. Severity by risk
    groups = [df.loc[df["Disease_Risk"] == i, "Disease_Severity_pct"].values for i in [0, 1, 2]]
    plt.figure(figsize=(7, 5))
    plt.boxplot(groups, tick_labels=["Low", "Medium", "High"])
    plt.title("Disease Severity by Risk")
    plt.ylabel("Severity (%)")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "severity_risk.png", dpi=130)
    plt.close()

    # 7. Correlation heatmap
    num = df.select_dtypes("number")
    corr = num.corr().values
    names = list(num.columns)
    fig, ax = plt.subplots(figsize=(12, 9))
    im = ax.imshow(corr, aspect="auto")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_title("Feature Correlation Heatmap")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "correlation_heatmap.png", dpi=130)
    plt.close()


def eda_stats():
    df = preprocess()
    risk_labels = df["Disease_Risk"].map({0: "Low", 1: "Medium", 2: "High"})
    return {
        "farms": len(df),
        "avg_yield": round(df["Yield"].mean(), 2),  # Updated to df["Yield"]
        "high_risk": int((risk_labels == "High").sum()),
        "high_pct": round((risk_labels == "High").mean() * 100, 2),
        "avg_ndvi": round(df["NDVI"].mean(), 3)
    }