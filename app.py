
import json, pickle
import pandas as pd
from pathlib import Path
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

MODEL_PATH = Path("models/price_model.pkl")
DATA_PATH  = Path("data/processed/cleaned_cars.csv")

model_bundle = None
df_global    = None
model_info   = {}

def load_assets():
    global model_bundle, df_global, model_info
    if MODEL_PATH.exists():
        with open(MODEL_PATH,"rb") as f:
            model_bundle = pickle.load(f)
        print("Model loaded")
    if Path("models/model_info.json").exists():
        with open("models/model_info.json") as f:
            model_info = json.load(f)
    if DATA_PATH.exists():
        df_global = pd.read_csv(DATA_PATH)
        print(f"Data loaded: {len(df_global)} rows")

load_assets()

def get_options():
    if df_global is None: return {}
    return {
        "brands":        sorted(df_global["brand"].dropna().unique().tolist()),
        "fuel_types":    sorted(df_global["fuel_type"].dropna().unique().tolist()) if "fuel_type" in df_global.columns else [],
        "transmissions": sorted(df_global["transmission"].dropna().unique().tolist()) if "transmission" in df_global.columns else [],
        "owners":        sorted(df_global["owner"].dropna().unique().tolist()) if "owner" in df_global.columns else [],
    }

@app.route("/")
def home():
    stats = {}
    if df_global is not None:
        stats = {
            "total":  f"{len(df_global):,}",
            "brands": df_global["brand"].nunique(),
            "median": f"Rs {df_global['price_inr'].median()/100_000:.1f}L",
            "r2":     f"{model_info.get('r2_score',0)*100:.1f}%",
        }
    return render_template("index.html", stats=stats)

@app.route("/predict", methods=["GET","POST"])
def predict():
    opts   = get_options()
    result = None
    if request.method == "POST":
        try:
            form         = request.form
            brand        = form.get("brand","")
            year         = int(form.get("year",2020))
            kms          = int(form.get("kms_driven",50000))
            fuel_type    = form.get("fuel_type","Petrol")
            transmission = form.get("transmission","Manual")
            owner        = form.get("owner","First Owner")
            car_age      = 2026 - year
            premium      = 1 if brand.lower() in ["bmw","audi","mercedes","porsche","jaguar"] else 0

            if model_bundle:
                model    = model_bundle["model"]
                encoders = model_bundle["encoders"]
                features = model_bundle["features"]
                row = pd.DataFrame([{"brand":brand,"car_age":car_age,"kms_driven":kms,
                                     "fuel_type":fuel_type,"transmission":transmission,
                                     "owner":owner,"premium_brand":premium}])
                for col, le in encoders.items():
                    if col in row.columns:
                        val = str(row[col].iloc[0])
                        row[col] = le.transform([val if val in le.classes_ else le.classes_[0]])
                pred   = model.predict(row[features])[0]
                result = {"price": f"Rs {pred/100_000:.2f} Lakhs",
                          "range": f"Rs {pred*0.9/100_000:.1f}L - Rs {pred*1.1/100_000:.1f}L",
                          "car":   f"{year} {brand}"}
            else:
                result = {"error": "Model not loaded. Run train_model.py first."}
        except Exception as e:
            result = {"error": str(e)}
    return render_template("predict.html", opts=opts, result=result)

@app.route("/analytics")
def analytics():
    charts = {}
    if df_global is not None:
        charts["brands"]       = df_global["brand"].value_counts().head(10).to_dict()
        if "fuel_type"     in df_global.columns: charts["fuel"]         = df_global["fuel_type"].value_counts().to_dict()
        if "transmission"  in df_global.columns: charts["transmission"] = df_global["transmission"].value_counts().to_dict()
        charts["avg_price"]    = (df_global.groupby("brand")["price_inr"].mean()
                                  .sort_values(ascending=False).head(10)
                                  .apply(lambda x: round(x/100_000,2)).to_dict())
        dep = (df_global[df_global["car_age"]<=20].groupby("car_age")["price_inr"]
               .mean().apply(lambda x: round(x/100_000,2)).to_dict())
        charts["depreciation"] = {str(k):v for k,v in dep.items()}
    return render_template("analytics.html", charts=json.dumps(charts))

if __name__ == "__main__":
    app.run(debug=True)
