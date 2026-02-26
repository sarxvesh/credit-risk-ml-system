from data_loader import load_data
from preprocessing import clean_data, scale_data
from models import train_logistic, train_rf
from evaluation import evaluate
from risk_engine import risk_level

from sklearn.model_selection import train_test_split
import joblib


def run_pipeline():
 
    df = load_data()
    df = clean_data(df)

    X = df.drop("A16", axis=1)
    y = df["A16"]

   
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

   
    X_train, X_test = scale_data(X_train, X_test)

    log_model = train_logistic(X_train, y_train)
    rf_model = train_rf(X_train, y_train)

    log_pred = log_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)

    print("\n===== Logistic Regression =====")
    evaluate(y_test, log_pred)

    print("\n===== Random Forest =====")
    evaluate(y_test, rf_pred)


    probs = rf_model.predict_proba(X_test)[:, 1]
    risks = [risk_level(p) for p in probs[:10]]

    print("\nSample Risk Levels:")
    print(risks)

 
    joblib.dump(rf_model, "credit_risk_model.pkl")
    print("\nModel saved as credit_risk_model.pkl")



if __name__ == "__main__":
    run_pipeline()