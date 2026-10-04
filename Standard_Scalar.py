from sklearn.preprocessing import StandardScaler
def standard_scale(X):
    s=StandardScaler(); return s.fit_transform(X),s