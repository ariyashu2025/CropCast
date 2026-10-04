from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import mean_squared_error,r2_score
from math import sqrt
from preprocessing import preprocess,FEATURES
def train_yield(method="linear"):
    df=preprocess(); X=df[FEATURES]; y=df.Yield_ton_ha
    a,b,c,d=train_test_split(X,y,test_size=.30,random_state=42)
    sc=StandardScaler(); a=sc.fit_transform(a); b=sc.transform(b)
    if method=="ridge": model=Ridge(alpha=1); name="Ridge Regression (L2)"
    elif method=="lasso": model=Lasso(alpha=.01,max_iter=10000); name="Lasso Regression (L1)"
    else: model=LinearRegression(); name="Linear Regression"
    model.fit(a,c); p=model.predict(b); mse=mean_squared_error(d,p)
    return {"model":name,"mse":round(mse,4),"rmse":round(sqrt(mse),4),"r2":round(r2_score(d,p),4)}