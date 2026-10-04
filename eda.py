from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from load_data import load_dataset
from config import CHART_DIR

def generate_charts():
    CHART_DIR.mkdir(parents=True,exist_ok=True)
    df=load_dataset()

    # 1 Risk distribution
    counts=df["Disease_Risk"].value_counts().reindex(["Low","Medium","High"]).fillna(0)
    plt.figure(figsize=(7,5)); plt.bar(counts.index,counts.values)
    plt.title("Disease Risk Distribution"); plt.ylabel("Farms"); plt.tight_layout()
    plt.savefig(CHART_DIR/"risk_distribution.png",dpi=130); plt.close()

    # 2 Yield by crop
    groups=[df.loc[df.Crop_Type==c,"Yield_ton_ha"].values for c in ["Rice","Wheat","Maize","Cotton","Groundnut"]]
    plt.figure(figsize=(8,5)); plt.boxplot(groups,tick_labels=["Rice","Wheat","Maize","Cotton","Groundnut"])
    plt.title("Yield by Crop"); plt.ylabel("Yield (ton/ha)"); plt.xticks(rotation=15); plt.tight_layout()
    plt.savefig(CHART_DIR/"yield_by_crop.png",dpi=130); plt.close()

    # 3 Rainfall vs yield
    plt.figure(figsize=(8,5))
    for c in ["Rice","Wheat","Maize","Cotton","Groundnut"]:
        q=df[df.Crop_Type==c]; plt.scatter(q.Rainfall_mm,q.Yield_ton_ha,s=12,label=c)
    plt.title("Rainfall vs Yield"); plt.xlabel("Rainfall (mm)"); plt.ylabel("Yield (ton/ha)"); plt.legend()
    plt.tight_layout(); plt.savefig(CHART_DIR/"rainfall_yield.png",dpi=130); plt.close()

    # 4 NDVI vs yield
    plt.figure(figsize=(8,5))
    for c in ["Low","Medium","High"]:
        q=df[df.Disease_Risk==c]; plt.scatter(q.NDVI,q.Yield_ton_ha,s=12,label=c)
    plt.title("NDVI vs Yield"); plt.xlabel("NDVI"); plt.ylabel("Yield (ton/ha)"); plt.legend()
    plt.tight_layout(); plt.savefig(CHART_DIR/"ndvi_yield.png",dpi=130); plt.close()

    # 5 Irrigation
    means=df.groupby("Irrigation")["Yield_ton_ha"].mean()
    plt.figure(figsize=(6,5)); plt.bar(means.index,means.values)
    plt.title("Irrigation vs Average Yield"); plt.ylabel("Yield (ton/ha)"); plt.tight_layout()
    plt.savefig(CHART_DIR/"irrigation_yield.png",dpi=130); plt.close()

    # 6 Severity by risk
    groups=[df.loc[df.Disease_Risk==c,"Disease_Severity_pct"].values for c in ["Low","Medium","High"]]
    plt.figure(figsize=(7,5)); plt.boxplot(groups,tick_labels=["Low","Medium","High"])
    plt.title("Disease Severity by Risk"); plt.ylabel("Severity (%)"); plt.tight_layout()
    plt.savefig(CHART_DIR/"severity_risk.png",dpi=130); plt.close()

    # 7 Correlation heatmap
    num=df.select_dtypes("number"); corr=num.corr().values; names=list(num.columns)
    fig,ax=plt.subplots(figsize=(12,9)); im=ax.imshow(corr,aspect="auto")
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names,rotation=90,fontsize=7); ax.set_yticklabels(names,fontsize=7)
    ax.set_title("Feature Correlation Heatmap"); fig.colorbar(im,ax=ax,fraction=.03,pad=.04)
    fig.tight_layout(); fig.savefig(CHART_DIR/"correlation_heatmap.png",dpi=130); plt.close()

def eda_stats():
    df=load_dataset()
    return {"farms":len(df),"avg_yield":round(df.Yield_ton_ha.mean(),2),
            "high_risk":int((df.Disease_Risk=="High").sum()),
            "high_pct":round((df.Disease_Risk=="High").mean()*100,2),
            "avg_ndvi":round(df.NDVI.mean(),3)}