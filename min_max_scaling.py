from sklearn.preprocessing import MinMaxScaler
def minmax_scale(X):
    s=MinMaxScaler(); return s.fit_transform(X),s