import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(path="data/employee_data.csv"):
    data = pd.read_csv(path)

    le = LabelEncoder()
    data["Department"] = le.fit_transform(data["Department"])
    data["Performance"] = le.fit_transform(data["Performance"])

    X = data.drop("Performance", axis=1)
    y = data["Performance"]

    return X, y