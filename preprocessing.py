import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def clean_data(df):
    df = df.copy()
    df.replace("?", pd.NA, inplace=True)

    cat = df.select_dtypes(include="object").columns
    num = df.select_dtypes(exclude="object").columns

    # Fill missing values
    for c in cat:
        df[c] = df[c].fillna(df[c].mode()[0])
    for c in num:
        df[c] = df[c].fillna(df[c].median())

    # Encode categorical columns
    for c in cat:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c])

    return df


def scale_data(X_train, X_test):
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)