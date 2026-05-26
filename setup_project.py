"""
Run this once to create all project files automatically.
    python setup_project.py
"""
import os

os.makedirs("scripts", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/final", exist_ok=True)
os.makedirs("dashboards", exist_ok=True)
os.makedirs("models", exist_ok=True)

# â”€â”€ clean_data.py â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
open("scripts/clean_data.py", "w").write('''
import re, pandas as pd
from pathlib import Path

RAW_PATH       = Path("data/raw/used_cars_raw.csv")
PROCESSED_PATH = Path("data/processed/cleaned_cars.csv")
PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)

PREMIUM_BRANDS = ["bmw","audi","mercedes","mercedes-benz","porsche",
                  "jaguar","land rover","volvo","lexus","maserati"]

def clean_price(val):
    val = str(val).lower().strip()
    if any(x in val for x in ["request","nan","none",""]):
        return None
    val = val.replace(",","").replace("â‚¹","").strip()
    if "lakh" in val or "lac" in val:
        num = re.sub(r"[^\\d.]","",val)
        return float(num)*100_000 if num else None
    if "crore" in val or "cr" in val:
        num = re.sub(r"[^\\d.]","",val)
        return float(num)*10_000_000 if num else None
    num = re.sub(r"[^\\d.]","",val)
    return float(num) if num else None

def clean_kms(val):
    val = str(val).lower().replace(",","").strip()
    num = re.sub(r"[^\\d]","",val)
    return int(num) if num else None

def clean_year(val):
    m = re.search(r"(19|20)\\d{2}",str(val))
    return int(m.group()) if m else None

def extract_brand(name):
    if not isinstance(name,str): return "Unknown"
    parts = name.strip().split()
    two_word = ["maruti suzuki","land rover","mercedes benz","mercedes-benz","rolls royce"]
    for brand in two_word:
        if name.lower().startswith(brand):
            return brand.title()
    return parts[0].title() if parts else "Unknown"

def main():
    print(f"Reading: {RAW_PATH}")
    df = pd.read_csv(RAW_PATH)
    print(f"Raw shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    rename_map = {}
    for col in df.columns:
        cl = col.lower().strip()
        if "name" in cl or "car" in cl:            rename_map[col] = "car_name"
        elif "price" in cl:                        rename_map[col] = "price_raw"
        elif "km" in cl or "odometer" in cl:       rename_map[col] = "kms_raw"
        elif "year" in cl:                         rename_map[col] = "year_raw"
        elif "fuel" in cl:                         rename_map[col] = "fuel_type"
        elif "transmission" in cl or "trans" in cl:rename_map[col] = "transmission"
        elif "owner" in cl:                        rename_map[col] = "owner"
        elif "city" in cl or "location" in cl:     rename_map[col] = "city"
        elif "seller" in cl:                       rename_map[col] = "seller_type"
    df.rename(columns=rename_map, inplace=True)

    df["price_inr"]  = df["price_raw"].apply(clean_price) if "price_raw" in df.columns else None
    df["kms_driven"] = df["kms_raw"].apply(clean_kms)     if "kms_raw"   in df.columns else None
    df["year"]       = df["year_raw"].apply(clean_year)   if "year_raw"  in df.columns else None

    for col in ["fuel_type","transmission","owner","city","seller_type"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
            df[col].replace({"Nan":"Unknown","None":"Unknown","":"Unknown"},inplace=True)

    if "car_name" in df.columns:
        df["brand"] = df["car_name"].apply(extract_brand)

    df["car_age"]       = 2026 - df["year"].fillna(2020)
    df["price_per_km"]  = (df["price_inr"] / df["kms_driven"].replace(0,None)).round(2)
    df["premium_brand"] = df["brand"].str.lower().isin(PREMIUM_BRANDS).astype(int) if "brand" in df.columns else 0
    df["price_flag"]    = df["price_inr"].apply(lambda x: "price_on_request" if pd.isna(x) else "ok")

    before = len(df)
    df.drop_duplicates(inplace=True)
    df = df[df["price_inr"].notna() & (df["price_inr"] > 10_000)]
    df = df[df["year"].notna() & (df["year"] >= 1990)]
    print(f"Rows after cleaning: {before} -> {len(df)}")

    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Saved cleaned data -> {PROCESSED_PATH}")
    print(df.head(3).to_string())

if __name__ == "__main__":
    main()
''')
print("âœ“ scripts/clean_data.py")

# â”€â”€ eda.py â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
open("scripts/eda.py", "w").write('''
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

CLEAN_PATH = Path("data/processed/cleaned_cars.csv")
DASH_DIR   = Path("dashboards")
DASH_DIR.mkdir(parents=True, exist_ok=True)
COLORS = ["#2563eb","#16a34a","#dc2626","#d97706","#7c3aed",
          "#0891b2","#db2777","#65a30d","#ea580c","#6366f1"]

def savefig(name):
    path = DASH_DIR / name
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return str(path)

def main():
    df = pd.read_csv(CLEAN_PATH)
    print(f"Loaded {len(df)} rows")
    charts = []

    fig, ax = plt.subplots(figsize=(10,5))
    top = df["brand"].value_counts().head(10)
    ax.barh(top.index[::-1], top.values[::-1], color=COLORS[:10])
    ax.set_title("Top 10 Car Brands", fontsize=14, fontweight="bold")
    plt.tight_layout()
    charts.append(("Top 10 Brands", savefig("01_top_brands.png")))

    fig, ax = plt.subplots(figsize=(10,5))
    prices = df[df["price_inr"] < 5_000_000]["price_inr"] / 100_000
    ax.hist(prices, bins=40, color="#2563eb", edgecolor="white")
    ax.set_title("Price Distribution (Lakhs)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    charts.append(("Price Distribution", savefig("02_price_dist.png")))

    fig, ax = plt.subplots(figsize=(10,5))
    age_price = df.groupby("car_age")["price_inr"].mean().sort_index() / 100_000
    age_price = age_price[age_price.index <= 20]
    ax.plot(age_price.index, age_price.values, color="#dc2626", linewidth=2, marker="o")
    ax.set_title("Depreciation Curve", fontsize=14, fontweight="bold")
    ax.set_xlabel("Car Age (Years)")
    ax.set_ylabel("Avg Price (Lakhs)")
    plt.tight_layout()
    charts.append(("Depreciation", savefig("03_depreciation.png")))

    if "fuel_type" in df.columns:
        fig, ax = plt.subplots(figsize=(7,7))
        fuel = df["fuel_type"].value_counts()
        ax.pie(fuel.values, labels=fuel.index, autopct="%1.1f%%", colors=COLORS[:len(fuel)])
        ax.set_title("Fuel Type Split", fontsize=14, fontweight="bold")
        plt.tight_layout()
        charts.append(("Fuel Type", savefig("04_fuel.png")))

    html_path = DASH_DIR / "eda_report.html"
    imgs = ""
    for title, path in charts:
        imgs += f\'<div class="card"><h3>{title}</h3><img src="{Path(path).name}"></div>\'
    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><title>EDA Report</title>
    <style>body{{font-family:sans-serif;background:#f1f5f9;padding:20px}}
    h1{{text-align:center;color:#1e3a5f}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:20px}}
    .card{{background:white;border-radius:12px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,.08)}}
    .card h3{{color:#1e3a5f;margin:0 0 12px}}.card img{{width:100%;border-radius:8px}}</style></head>
    <body><h1>CarWale EDA Report â€” {len(df):,} listings</h1><div class="grid">{imgs}</div></body></html>"""
    html_path.write_text(html, encoding="utf-8")
    print(f"Report saved -> {html_path}")
    print("Open dashboards/eda_report.html in your browser!")

if __name__ == "__main__":
    main()
''')
print("âœ“ scripts/eda.py")

# â”€â”€ train_model.py â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
open("scripts/train_model.py", "w").write('''
import json, pickle
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

CLEAN_PATH = Path("data/processed/cleaned_cars.csv")
MODEL_DIR  = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

FEATURES = ["brand","car_age","kms_driven","fuel_type","transmission","owner","premium_brand"]
TARGET   = "price_inr"

def main():
    df = pd.read_csv(CLEAN_PATH)
    print(f"Loaded {len(df)} rows")

    avail = [f for f in FEATURES if f in df.columns]
    df = df[avail + [TARGET]].dropna()
    print(f"Training on {len(df)} rows | Features: {avail}")

    encoders = {}
    for col in ["brand","fuel_type","transmission","owner"]:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    X = df[avail]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    print("Training model...")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    r2    = r2_score(y_test, preds)
    print(f"MAE : Rs {mae:,.0f}")
    print(f"R2  : {r2:.4f}  ({r2*100:.1f}% accuracy)")

    bundle = {"model": model, "encoders": encoders, "features": avail}
    with open(MODEL_DIR / "price_model.pkl","wb") as f:
        pickle.dump(bundle, f)

    info = {"r2_score": round(r2,4), "mae_inr": round(mae), "features": avail, "rows": len(df)}
    with open(MODEL_DIR / "model_info.json","w") as f:
        json.dump(info, f, indent=2)

    print(f"Model saved -> models/price_model.pkl")

if __name__ == "__main__":
    main()
''')
print("âœ“ scripts/train_model.py")

# â”€â”€ app.py â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
open("app.py", "w").write('''
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
            "median": f"Rs {df_global[\'price_inr\'].median()/100_000:.1f}L",
            "r2":     f"{model_info.get(\'r2_score\',0)*100:.1f}%",
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
''')
print("âœ“ app.py")

# â”€â”€ templates â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
open("templates/base.html","w").write("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{% block title %}CarIQ{% endblock %}</title>
<link href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.0/css/bootstrap.min.css" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
body{background:#f8fafc;font-family:'Segoe UI',sans-serif}
.navbar{background:#0f172a!important}
.navbar-brand{font-weight:700;font-size:1.4rem;color:white!important}
.nav-link{color:rgba(255,255,255,.75)!important}
.nav-link:hover{color:white!important}
.card{border:none;border-radius:14px;box-shadow:0 2px 12px rgba(0,0,0,.06)}
</style>
{% block extra_css %}{% endblock %}
</head>
<body>
<nav class="navbar navbar-expand-lg">
  <div class="container">
    <a class="navbar-brand" href="/">ðŸš— CarIQ</a>
    <div class="collapse navbar-collapse">
      <ul class="navbar-nav ms-auto">
        <li class="nav-item"><a class="nav-link" href="/">Home</a></li>
        <li class="nav-item"><a class="nav-link" href="/predict">Predict Price</a></li>
        <li class="nav-item"><a class="nav-link" href="/analytics">Analytics</a></li>
      </ul>
    </div>
  </div>
</nav>
{% block content %}{% endblock %}
<footer class="text-center py-4 mt-5 text-muted small">Built with Python Â· Flask Â· Scikit-learn</footer>
<script src="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.0/js/bootstrap.bundle.min.js"></script>
{% block extra_js %}{% endblock %}
</body>
</html>""")
print("âœ“ templates/base.html")

open("templates/index.html","w").write("""{% extends "base.html" %}
{% block title %}Home{% endblock %}
{% block extra_css %}
<style>
.hero{background:linear-gradient(135deg,#0f172a,#1e3a5f);color:white;padding:80px 0}
.hero h1{font-size:3rem;font-weight:800}
.stat-card{border-radius:14px;padding:28px;text-align:center;color:white}
.stat-num{font-size:2.4rem;font-weight:800}
</style>
{% endblock %}
{% block content %}
<div class="hero text-center">
  <div class="container">
    <h1>Smart Car Price Intelligence</h1>
    <p class="mb-4 opacity-75">AI-powered used car price prediction for India</p>
    <a href="/predict" class="btn btn-primary btn-lg me-2">Predict Price</a>
    <a href="/analytics" class="btn btn-outline-light btn-lg">View Analytics</a>
  </div>
</div>
{% if stats %}
<div class="container my-5">
  <div class="row g-4">
    <div class="col-md-3"><div class="stat-card" style="background:#2563eb"><div class="stat-num">{{ stats.total }}</div><div>Cars Analysed</div></div></div>
    <div class="col-md-3"><div class="stat-card" style="background:#16a34a"><div class="stat-num">{{ stats.brands }}</div><div>Unique Brands</div></div></div>
    <div class="col-md-3"><div class="stat-card" style="background:#d97706"><div class="stat-num">{{ stats.median }}</div><div>Median Price</div></div></div>
    <div class="col-md-3"><div class="stat-card" style="background:#7c3aed"><div class="stat-num">{{ stats.r2 }}</div><div>Model Accuracy</div></div></div>
  </div>
</div>
{% endif %}
{% endblock %}""")
print("âœ“ templates/index.html")

open("templates/predict.html","w").write("""{% extends "base.html" %}
{% block title %}Predict Price{% endblock %}
{% block extra_css %}
<style>
.hero{background:linear-gradient(135deg,#1e3a5f,#2563eb);color:white;padding:50px 0}
.result{background:linear-gradient(135deg,#16a34a,#15803d);color:white;border-radius:16px;padding:32px;text-align:center}
.result-price{font-size:3rem;font-weight:800}
</style>
{% endblock %}
{% block content %}
<div class="hero text-center"><h2 class="fw-bold">Predict Car Resale Price</h2></div>
<div class="container my-5">
  <div class="row justify-content-center">
    <div class="col-lg-7">
      <div class="card p-4 mb-4">
        <form method="POST">
          <div class="row g-3">
            <div class="col-md-6">
              <label class="form-label fw-bold">Brand</label>
              <select name="brand" class="form-select">{% for b in opts.brands %}<option>{{ b }}</option>{% endfor %}</select>
            </div>
            <div class="col-md-6">
              <label class="form-label fw-bold">Year</label>
              <select name="year" class="form-select">{% for y in range(2024,1999,-1) %}<option>{{ y }}</option>{% endfor %}</select>
            </div>
            <div class="col-md-6">
              <label class="form-label fw-bold">KMs Driven</label>
              <input type="number" name="kms_driven" class="form-control" placeholder="e.g. 45000">
            </div>
            <div class="col-md-6">
              <label class="form-label fw-bold">Fuel Type</label>
              <select name="fuel_type" class="form-select">{% for f in opts.fuel_types %}<option>{{ f }}</option>{% endfor %}</select>
            </div>
            <div class="col-md-6">
              <label class="form-label fw-bold">Transmission</label>
              <select name="transmission" class="form-select">{% for t in opts.transmissions %}<option>{{ t }}</option>{% endfor %}</select>
            </div>
            <div class="col-md-6">
              <label class="form-label fw-bold">Ownership</label>
              <select name="owner" class="form-select">{% for o in opts.owners %}<option>{{ o }}</option>{% endfor %}</select>
            </div>
            <div class="col-12"><button type="submit" class="btn btn-primary w-100 py-2 fw-bold">Get Price Estimate</button></div>
          </div>
        </form>
      </div>
      {% if result %}
        {% if result.error %}<div class="alert alert-danger">{{ result.error }}</div>
        {% else %}
        <div class="result">
          <div class="mb-2 opacity-75">Estimated value for {{ result.car }}</div>
          <div class="result-price">{{ result.price }}</div>
          <div class="mt-2">Market Range: {{ result.range }}</div>
        </div>
        {% endif %}
      {% endif %}
    </div>
  </div>
</div>
{% endblock %}""")
print("âœ“ templates/predict.html")

open("templates/analytics.html","w").write("""{% extends "base.html" %}
{% block title %}Analytics{% endblock %}
{% block extra_css %}
<style>
.hero{background:linear-gradient(135deg,#0f172a,#7c3aed);color:white;padding:50px 0}
.chart-card{background:white;border-radius:14px;padding:20px;box-shadow:0 2px 12px rgba(0,0,0,.06);margin-bottom:24px}
.chart-card h5{font-weight:700;color:#1e3a5f;margin-bottom:16px}
</style>
{% endblock %}
{% block content %}
<div class="hero text-center"><h2 class="fw-bold">Market Analytics</h2></div>
<div class="container my-5">
  <div class="row">
    <div class="col-lg-6"><div class="chart-card"><h5>Top Brands by Listings</h5><canvas id="brandsChart" height="250"></canvas></div></div>
    <div class="col-lg-6"><div class="chart-card"><h5>Average Price by Brand (Lakhs)</h5><canvas id="avgPriceChart" height="250"></canvas></div></div>
    <div class="col-lg-6"><div class="chart-card"><h5>Fuel Type Distribution</h5><canvas id="fuelChart" height="250"></canvas></div></div>
    <div class="col-lg-6"><div class="chart-card"><h5>Transmission Split</h5><canvas id="transChart" height="250"></canvas></div></div>
    <div class="col-12"><div class="chart-card"><h5>Depreciation Curve</h5><canvas id="depChart" height="120"></canvas></div></div>
  </div>
</div>
{% endblock %}
{% block extra_js %}
<script>
const data = {{ charts|safe }};
const C = ["#2563eb","#16a34a","#dc2626","#d97706","#7c3aed","#0891b2","#db2777","#65a30d","#ea580c","#6366f1"];
function bar(id,labels,values,color){new Chart(document.getElementById(id),{type:"bar",data:{labels,datasets:[{data:values,backgroundColor:color||C,borderRadius:6}]},options:{plugins:{legend:{display:false}},scales:{x:{grid:{display:false}}}}})}
function pie(id,labels,values){new Chart(document.getElementById(id),{type:"doughnut",data:{labels,datasets:[{data:values,backgroundColor:C,borderWidth:2,borderColor:"white"}]},options:{plugins:{legend:{position:"right"}}}})}
if(data.brands){const e=Object.entries(data.brands).slice(0,10);bar("brandsChart",e.map(x=>x[0]),e.map(x=>x[1]),"#2563eb")}
if(data.avg_price){const e=Object.entries(data.avg_price).slice(0,10);bar("avgPriceChart",e.map(x=>x[0]),e.map(x=>x[1]),"#7c3aed")}
if(data.fuel){pie("fuelChart",Object.keys(data.fuel),Object.values(data.fuel))}
if(data.transmission){pie("transChart",Object.keys(data.transmission),Object.values(data.transmission))}
if(data.depreciation){const s=Object.entries(data.depreciation).sort((a,b)=>+a[0]-+b[0]);new Chart(document.getElementById("depChart"),{type:"line",data:{labels:s.map(e=>e[0]+" yrs"),datasets:[{data:s.map(e=>e[1]),borderColor:"#dc2626",backgroundColor:"rgba(220,38,38,.08)",tension:0.4,fill:true}]},options:{plugins:{legend:{display:false}}}})}
</script>
{% endblock %}""")
print("âœ“ templates/analytics.html")

print("\nâœ… ALL FILES CREATED SUCCESSFULLY!")
print("Now run these commands one by one:")
print("  pip install -r requirements.txt")
print("  python scripts/clean_data.py")
print("  python scripts/eda.py")
print("  python scripts/train_model.py")
print("  python app.py")
