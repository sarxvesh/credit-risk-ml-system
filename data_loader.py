from ucimlrepo import fetch_ucirepo
import pandas as pd


def load_data():
    credit_approval = fetch_ucirepo(id=27)

    X = credit_approval.data.features
    y = credit_approval.data.targets

    df = pd.concat([X, y], axis=1)
    return df