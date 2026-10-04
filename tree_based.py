from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from preprocessing import preprocess,FEATURES
try:
 from xgboost import XGBClassifier
except Exception: XGBClassifier=None
try:
 from lightgbm import LGBMClassifier
except Exception: LGBMClassifier=None
def train_tree(name):
    df=preprocess(); a,b,c,d=train_test_split(df[FEATURES],df.Disease_Risk,test_size=.3,random_state=42,stratify=df.Disease_Risk)
    models={"decision_tree":(DecisionTreeClassifier(max_depth=6,random_state=42),"Decision Tree"),
            "random_forest":(RandomForestClassifier(n_estimators=160,random_state=42),"Random Forest"),
            "extra_trees":(ExtraTreesClassifier(n_estimators=160,random_state=42),"Extra Trees"),
            "gradient_boosting":(GradientBoostingClassifier(n_estimators=120,random_state=42),"Gradient Boosting"),
            "adaboost":(AdaBoostClassifier(n_estimators=120,random_state=42),"AdaBoost")}
    if name=="xgboost":
        if XGBClassifier is None:return {"error":"Install xgboost first"}
        models[name]=(XGBClassifier(n_estimators=150,max_depth=4,learning_rate=.08,random_state=42,eval_metric="mlogloss"),"XGBoost")
    if name=="lightgbm":
        if LGBMClassifier is None:return {"error":"Install lightgbm first"}
        models[name]=(LGBMClassifier(n_estimators=150,learning_rate=.08,random_state=42,verbosity=-1),"LightGBM")
    m,label=models.get(name,(None,None))
    if m is None:return {"error":"Unknown algorithm"}
    m.fit(a,c); p=m.predict(b)
    return {"model":label,"accuracy":round(accuracy_score(d,p),4),"precision":round(precision_score(d,p,average="weighted",zero_division=0),4),
            "recall":round(recall_score(d,p,average="weighted",zero_division=0),4),"f1":round(f1_score(d,p,average="weighted",zero_division=0),4)}