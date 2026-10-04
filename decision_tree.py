from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from preprocessing import preprocess,FEATURES
def train():
    df=preprocess(); a,b,c,d=train_test_split(df[FEATURES],df.Disease_Risk,test_size=.3,random_state=42,stratify=df.Disease_Risk)
    m=DecisionTreeClassifier(max_depth=6,random_state=42); m.fit(a,c); return {"model":"Decision Tree","accuracy":round(accuracy_score(d,m.predict(b)),4)}