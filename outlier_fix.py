def iqr_report(df,column):
    s=df[column].dropna(); q1,q3=s.quantile(.25),s.quantile(.75); iqr=q3-q1
    lo,hi=q1-1.5*iqr,q3+1.5*iqr
    return {"column":column,"q1":round(q1,3),"q3":round(q3,3),"iqr":round(iqr,3),
            "lower":round(lo,3),"upper":round(hi,3),"count":int(((s<lo)|(s>hi)).sum())}
def clip_outliers(df,columns):
    out=df.copy()
    for c in columns:
        q1,q3=out[c].quantile(.25),out[c].quantile(.75); iqr=q3-q1
        out[c]=out[c].clip(q1-1.5*iqr,q3+1.5*iqr)
    return out