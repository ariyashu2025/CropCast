import pandas as pd
from config import DATA_PATH
def load_dataset():
    return pd.read_csv(DATA_PATH)
def dataset_summary():
    df=load_dataset()
    return {"rows":len(df),"columns":len(df.columns),"missing":int(df.isna().sum().sum()),
            "duplicates":int(df.duplicated().sum()),"numeric":len(df.select_dtypes("number").columns),
            "categorical":len(df.select_dtypes(exclude="number").columns),"preview":df.head(10).to_html(classes="table",index=False)}