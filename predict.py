import joblib
from risk_engine import risk_level

model = joblib.load("credit_risk_model.pkl")


def predict(sample):
    prob = model.predict_proba([sample])[0][1]
    return risk_level(prob)



if __name__ == "__main__":
   
    sample_input = [1] * 15   

    result = predict(sample_input)
    print("Predicted Risk Level:", result)