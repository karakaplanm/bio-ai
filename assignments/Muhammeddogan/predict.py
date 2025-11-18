import joblib
import pandas as pd

def predict(input_dict):
    model = joblib.load("../models/best_model.pkl")
    scaler = joblib.load("../models/scaler.pkl")

    df = pd.DataFrame([input_dict])
    df_scaled = scaler.transform(df)

    prob = model.predict_proba(df_scaled)[0][1]
    return prob
