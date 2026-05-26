"""
CarIQ - Single File Version
============================
This single file runs the entire website.
No other files needed except your CSV data.

To run locally:
    pip install flask pandas scikit-learn matplotlib numpy
    python app.py

To host on Render:
    1. Upload this file to GitHub
    2. Connect to Render
    3. Start command: gunicorn app:app
"""

import os, re, json, pickle, io, base64
import pandas as pd
import numpy as np
from pathlib import Path
from flask import Flask, render_template_string, request

app = Flask(__name__)

# ── Sample data (used if no CSV is uploaded) ──────────────────────────────────
SAMPLE_DATA = [
    {"car_name":"Maruti Suzuki Swift","brand":"Maruti Suzuki","year":2020,"price_inr":550000,"kms_driven":35000,"fuel_type":"Petrol","transmission":"Manual","owner":"First Owner","car_age":6,"premium_brand":0},
    {"car_name":"Hyundai Creta","brand":"Hyundai","year":2019,"price_inr":850000,"kms_driven":45000,"fuel_type":"Petrol","transmission":"Manual","owner":"First Owner","car_age":7,"premium_brand":0},
    {"car_name":"Honda City","brand":"Honda","year":2018,"price_inr":720000,"kms_driven":60000,"fuel_type":"Petrol","transmission":"Manual","owner":"Second Owner","car_age":8,"premium_brand":0},
    {"car_name":"Toyota Fortuner","brand":"Toyota","year":2020,"price_inr":2800000,"kms_driven":30000,"fuel_type":"Diesel","transmission":"Automatic","owner":"First Owner","car_age":6,"premium_brand":0},
    {"car_name":"BMW 3 Series","brand":"BMW","year":2019,"price_inr":3200000,"kms_driven":40000,"fuel_type":"Petrol","transmission":"Automatic","owner":"First Owner","car_age":7,"premium_brand":1},
    {"car_name":"Maruti Suzuki Baleno","brand":"Maruti Suzuki","year":2021,"price_inr":620000,"kms_driven":20000,"fuel_type":"Petrol","transmission":"Manual","owner":"First Owner","car_age":5,"premium_brand":0},
    {"car_name":"Hyundai i20","brand":"Hyundai","year":2020,"price_inr":680000,"kms_driven":28000,"fuel_type":"Petrol","transmission":"Manual","owner":"First Owner","car_age":6,"premium_brand":0},
    {"car_name":"Tata Nexon","brand":"Tata","year":2021,"price_inr":920000,"kms_driven":22000,"fuel_type":"Diesel","transmission":"Manual","owner":"First Owner","car_age":5,"premium_brand":0},
    {"car_name":"Honda Jazz","brand":"Honda","year":2019,"price_inr":590000,"kms_driven":42000,"fuel_type":"Petrol","transmission":"Manual","owner":"Second Owner","car_age":7,"premium_brand":0},
    {"car_name":"Audi A4","brand":"Audi","year":2018,"price_inr":2900000,"kms_driven":55000,"fuel_type":"Petrol","transmission":"Automatic","owner":"Second Owner","car_age":8,"premium_brand":1},
    {"car_name":"Maruti Suzuki Dzire","brand":"Maruti Suzuki","year":2020,"price_inr":580000,"kms_driven":38000,"fuel_type":"Petrol","transmission":"Manual","owner":"First Owner","car_age":6,"premium_brand":0},
    {"car_name":"Hyundai Verna","brand":"Hyundai","year":2021,"price_inr":980000,"kms_driven":18000,"fuel_type":"Petrol","transmission":"Automatic","owner":"First Owner","car_age":5,"premium_brand":0},
    {"car_name":"Toyota Innova","brand":"Toyota","year":2019,"price_inr":1650000,"kms_driven":48000,"fuel_type":"Diesel","transmission":"Manual","owner":"First Owner","car_age":7,"premium_brand":0},
    {"car_name":"Kia Seltos","brand":"Kia","year":2021,"price_inr":1150000,"kms_driven":25000,"fuel_type":"Petrol","transmission":"Automatic","owner":"First Owner","car_age":5,"premium_brand":0},
    {"car_name":"Mahindra XUV500","brand":"Mahindra","year":2018,"price_inr":890000,"kms_driven":65000,"fuel_type":"Diesel","transmission":"Manual","owner":"Second Owner","car_age":8,"premium_brand":0},
]

# ── Load or create data ───────────────────────────────────────────────────────
def load_data():
    paths = [
        "data/processed/cleaned_cars.csv",
        "data/raw/used_cars_raw.csv",
        "used_cars_raw.csv",
        "cleaned_cars.csv",
    ]
    for path in paths:
        if Path(path).exists():
            try:
                df = pd.read_csv(path)
                if len(df) > 5:
                    print(f"Loaded data from {path}: {len(df)} rows")
                    return df
            except:
                pass
    print("Using sample data")
    return pd.DataFrame(SAMPLE_DATA)

df_global = load_data()

# ── Clean data if needed ──────────────────────────────────────────────────────
def ensure_columns(df):
    rename_map = {}
    for col in df.columns:
        cl = col.lower().strip()
        if "name" in cl and "car" in cl:            rename_map[col] = "car_name"
        elif cl == "name":                          rename_map[col] = "car_name"
        elif "price" in cl and "raw" not in cl:     rename_map[col] = "price_inr"
        elif "price" in cl:                         rename_map[col] = "price_raw"
        elif "km" in cl or "odometer" in cl:        rename_map[col] = "kms_driven"
        elif "year" in cl and "raw" not in cl:      rename_map[col] = "year"
        elif "fuel" in cl:                          rename_map[col] = "fuel_type"
        elif "trans" in cl:                         rename_map[col] = "transmission"
        elif "owner" in cl:                         rename_map[col] = "owner"
        elif "city" in cl or "location" in cl:      rename_map[col] = "city"
    df = df.rename(columns=rename_map)

    if "price_raw" in df.columns and "price_inr" not in df.columns:
        def cp(v):
            v = str(v).lower().replace(",","")
            if "lakh" in v or "lac" in v:
                n = re.sub(r"[^\d.]","",v)
                return float(n)*100000 if n else None
            n = re.sub(r"[^\d.]","",v)
            return float(n) if n else None
        df["price_inr"] = df["price_raw"].apply(cp)

    if "kms_raw" in df.columns and "kms_driven" not in df.columns:
        df["kms_driven"] = df["kms_raw"].apply(lambda v: int(re.sub(r"[^\d]","",str(v))) if re.sub(r"[^\d]","",str(v)) else None)

    if "price_inr" not in df.columns:
        df["price_inr"] = None
    if "kms_driven" not in df.columns:
        df["kms_driven"] = None
    if "year" not in df.columns:
        df["year"] = 2020

    df["price_inr"] = pd.to_numeric(df["price_inr"], errors="coerce")
    df["kms_driven"] = pd.to_numeric(df["kms_driven"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    if "brand" not in df.columns:
        def eb(name):
            if not isinstance(name, str): return "Unknown"
            parts = name.strip().split()
            for b in ["maruti suzuki","land rover","mercedes benz"]:
                if name.lower().startswith(b): return b.title()
            return parts[0].title() if parts else "Unknown"
        src = "car_name" if "car_name" in df.columns else df.columns[0]
        df["brand"] = df[src].apply(eb)

    df["car_age"] = 2026 - df["year"].fillna(2020)
    df["premium_brand"] = df["brand"].str.lower().isin(["bmw","audi","mercedes","porsche","jaguar","land rover","volvo","lexus"]).astype(int)

    df = df[df["price_inr"].notna() & (df["price_inr"] > 10000)]
    return df

df_global = ensure_columns(df_global)

# ── Train model ───────────────────────────────────────────────────────────────
model_bundle = None

def train_model(df):
    global model_bundle
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score

        FEATURES = ["brand","car_age","kms_driven","fuel_type","transmission","owner","premium_brand"]
        avail = [f for f in FEATURES if f in df.columns]
        data = df[avail + ["price_inr"]].dropna()
        if len(data) < 10:
            return None, 0

        encoders = {}
        for col in ["brand","fuel_type","transmission","owner"]:
            if col in data.columns:
                le = LabelEncoder()
                data = data.copy()
                data[col] = le.fit_transform(data[col].astype(str))
                encoders[col] = le

        X, y = data[avail], data["price_inr"]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(Xtr, ytr)
        r2 = r2_score(yte, model.predict(Xte))
        model_bundle = {"model": model, "encoders": encoders, "features": avail}
        print(f"Model trained. R2={r2:.3f}")
        return model_bundle, r2
    except Exception as e:
        print(f"Model training failed: {e}")
        return None, 0

model_bundle, model_r2 = train_model(df_global)

# ── Chart generator ───────────────────────────────────────────────────────────
def make_charts(df):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        charts = {}

        fig, ax = plt.subplots(figsize=(8,4))
        top = df["brand"].value_counts().head(8)
        ax.barh(top.index[::-1], top.values[::-1], color="#2563eb")
        ax.set_title("Top Brands by Listings", fontweight="bold")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        buf = io.BytesIO(); plt.savefig(buf, format="png", dpi=100); buf.seek(0)
        charts["brands"] = base64.b64encode(buf.read()).decode()
        plt.close()

        fig, ax = plt.subplots(figsize=(8,4))
        prices = df[df["price_inr"] < 5000000]["price_inr"] / 100000
        ax.hist(prices, bins=30, color="#16a34a", edgecolor="white")
        ax.set_title("Price Distribution (Lakhs)", fontweight="bold")
        ax.set_xlabel("Price (Lakhs)")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        buf = io.BytesIO(); plt.savefig(buf, format="png", dpi=100); buf.seek(0)
        charts["prices"] = base64.b64encode(buf.read()).decode()
        plt.close()

        fig, ax = plt.subplots(figsize=(8,4))
        dep = df[df["car_age"]<=15].groupby("car_age")["price_inr"].mean() / 100000
        ax.plot(dep.index, dep.values, color="#dc2626", linewidth=2, marker="o", markersize=4)
        ax.set_title("Depreciation Curve", fontweight="bold")
        ax.set_xlabel("Car Age (Years)"); ax.set_ylabel("Avg Price (Lakhs)")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        buf = io.BytesIO(); plt.savefig(buf, format="png", dpi=100); buf.seek(0)
        charts["depreciation"] = base64.b64encode(buf.read()).decode()
        plt.close()

        if "fuel_type" in df.columns:
            fig, ax = plt.subplots(figsize=(6,6))
            fuel = df["fuel_type"].value_counts()
            colors = ["#2563eb","#16a34a","#d97706","#7c3aed","#dc2626"]
            ax.pie(fuel.values, labels=fuel.index, autopct="%1.1f%%", colors=colors[:len(fuel)])
            ax.set_title("Fuel Type Split", fontweight="bold")
            plt.tight_layout()
            buf = io.BytesIO(); plt.savefig(buf, format="png", dpi=100); buf.seek(0)
            charts["fuel"] = base64.b64encode(buf.read()).decode()
            plt.close()

        return charts
    except Exception as e:
        print(f"Chart error: {e}")
        return {}

charts_cache = make_charts(df_global)

# ── Stats ─────────────────────────────────────────────────────────────────────
def get_stats():
    return {
        "total":  f"{len(df_global):,}",
        "brands": df_global["brand"].nunique() if "brand" in df_global.columns else 0,
        "median": f"Rs {df_global['price_inr'].median()/100000:.1f}L",
        "r2":     f"{model_r2*100:.1f}%",
    }

def get_options():
    opts = {"brands":[], "fuel_types":[], "transmissions":[], "owners":[]}
    if "brand"        in df_global.columns: opts["brands"]        = sorted(df_global["brand"].dropna().unique().tolist())
    if "fuel_type"    in df_global.columns: opts["fuel_types"]    = sorted(df_global["fuel_type"].dropna().unique().tolist())
    if "transmission" in df_global.columns: opts["transmissions"] = sorted(df_global["transmission"].dropna().unique().tolist())
    if "owner"        in df_global.columns: opts["owners"]        = sorted(df_global["owner"].dropna().unique().tolist())
    if not opts["fuel_types"]:    opts["fuel_types"]    = ["Petrol","Diesel","CNG","Electric"]
    if not opts["transmissions"]: opts["transmissions"] = ["Manual","Automatic"]
    if not opts["owners"]:        opts["owners"]        = ["First Owner","Second Owner","Third Owner"]
    return opts

# ── HTML Templates ────────────────────────────────────────────────────────────
BASE = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{{ title }} - CarIQ</title>
<link href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.0/css/bootstrap.min.css" rel="stylesheet">
<style>
body{background:#f8fafc;font-family:'Segoe UI',sans-serif}
.navbar{background:#0f172a!important}
.navbar-brand{font-weight:700;font-size:1.3rem;color:white!important}
.nav-link{color:rgba(255,255,255,.7)!important}
.nav-link:hover,.nav-link.active{color:white!important}
.card{border:none;border-radius:14px;box-shadow:0 2px 12px rgba(0,0,0,.06)}
footer{text-align:center;padding:20px;color:#888;font-size:13px;margin-top:40px;border-top:1px solid #e2e8f0}
</style>
{% block style %}{% endblock %}
</head>
<body>
<nav class="navbar navbar-expand-lg">
 <div class="container">
  <a class="navbar-brand" href="/">&#128664; CarIQ</a>
  <div class="navbar-nav ms-auto flex-row gap-2">
   <a class="nav-link {% if active=='home' %}active{% endif %}" href="/">Home</a>
   <a class="nav-link {% if active=='predict' %}active{% endif %}" href="/predict">Predict</a>
   <a class="nav-link {% if active=='analytics' %}active{% endif %}" href="/analytics">Analytics</a>
  </div>
 </div>
</nav>
{% block content %}{% endblock %}
<footer>Built with Python &middot; Flask &middot; Scikit-learn &middot; CarWale Data</footer>
<script src="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.0/js/bootstrap.bundle.min.js"></script>
</body>
</html>"""

HOME_TMPL = BASE.replace("{% block style %}{% endblock %}","""
<style>
.hero{background:linear-gradient(135deg,#0f172a,#1e3a5f);color:white;padding:80px 0;text-align:center}
.hero h1{font-size:2.8rem;font-weight:800}
.stat-card{border-radius:14px;padding:24px;text-align:center;color:white}
.stat-num{font-size:2rem;font-weight:800}
.feat-card{background:white;border-radius:14px;padding:24px;box-shadow:0 2px 12px rgba(0,0,0,.06);height:100%}
</style>
""").replace("{% block content %}{% endblock %}","""
<div class="hero">
 <div class="container">
  <h1>Smart Car Price Intelligence</h1>
  <p class="opacity-75 mb-4">AI-powered used car price prediction for India</p>
  <a href="/predict" class="btn btn-primary btn-lg me-2">Predict Price</a>
  <a href="/analytics" class="btn btn-outline-light btn-lg">View Analytics</a>
 </div>
</div>
<div class="container my-5">
 <div class="row g-4 mb-5">
  <div class="col-md-3"><div class="stat-card" style="background:#2563eb"><div class="stat-num">{{ stats.total }}</div><div>Cars Analysed</div></div></div>
  <div class="col-md-3"><div class="stat-card" style="background:#16a34a"><div class="stat-num">{{ stats.brands }}</div><div>Unique Brands</div></div></div>
  <div class="col-md-3"><div class="stat-card" style="background:#d97706"><div class="stat-num">{{ stats.median }}</div><div>Median Price</div></div></div>
  <div class="col-md-3"><div class="stat-card" style="background:#7c3aed"><div class="stat-num">{{ stats.r2 }}</div><div>Model Accuracy</div></div></div>
 </div>
 <div class="row g-4">
  <div class="col-md-4"><div class="feat-card"><div style="font-size:2rem">&#129302;</div><h5 class="mt-2 fw-bold">ML Price Prediction</h5><p class="text-muted">Enter brand, year, KMs and fuel type to get an instant resale price estimate.</p><a href="/predict" class="btn btn-primary btn-sm">Try it</a></div></div>
  <div class="col-md-4"><div class="feat-card"><div style="font-size:2rem">&#128202;</div><h5 class="mt-2 fw-bold">Market Analytics</h5><p class="text-muted">Explore price trends, depreciation curves, fuel splits and brand comparisons.</p><a href="/analytics" class="btn btn-primary btn-sm">Explore</a></div></div>
  <div class="col-md-4"><div class="feat-card"><div style="font-size:2rem">&#128396;</div><h5 class="mt-2 fw-bold">Real Data</h5><p class="text-muted">All data scraped live from CarWale, cleaned through a Python ETL pipeline.</p><a href="/analytics" class="btn btn-primary btn-sm">See Data</a></div></div>
 </div>
</div>
""")

PREDICT_TMPL = BASE.replace("{% block style %}{% endblock %}","""
<style>
.hero{background:linear-gradient(135deg,#1e3a5f,#2563eb);color:white;padding:40px 0;text-align:center}
.result-box{background:linear-gradient(135deg,#16a34a,#15803d);color:white;border-radius:16px;padding:32px;text-align:center}
.result-price{font-size:3rem;font-weight:800}
</style>
""").replace("{% block content %}{% endblock %}","""
<div class="hero"><h2 class="fw-bold">Predict Car Resale Price</h2><p class="opacity-75">Fill in the details for an instant ML-powered estimate</p></div>
<div class="container my-5">
 <div class="row justify-content-center">
  <div class="col-lg-7">
   <div class="card p-4 mb-4">
    <form method="POST">
     <div class="row g-3">
      <div class="col-md-6"><label class="form-label fw-bold">Brand</label>
       <select name="brand" class="form-select">{% for b in opts.brands %}<option>{{ b }}</option>{% endfor %}</select></div>
      <div class="col-md-6"><label class="form-label fw-bold">Year</label>
       <select name="year" class="form-select">{% for y in range(2024,1999,-1) %}<option>{{ y }}</option>{% endfor %}</select></div>
      <div class="col-md-6"><label class="form-label fw-bold">KMs Driven</label>
       <input type="number" name="kms_driven" class="form-control" placeholder="e.g. 45000" required></div>
      <div class="col-md-6"><label class="form-label fw-bold">Fuel Type</label>
       <select name="fuel_type" class="form-select">{% for f in opts.fuel_types %}<option>{{ f }}</option>{% endfor %}</select></div>
      <div class="col-md-6"><label class="form-label fw-bold">Transmission</label>
       <select name="transmission" class="form-select">{% for t in opts.transmissions %}<option>{{ t }}</option>{% endfor %}</select></div>
      <div class="col-md-6"><label class="form-label fw-bold">Ownership</label>
       <select name="owner" class="form-select">{% for o in opts.owners %}<option>{{ o }}</option>{% endfor %}</select></div>
      <div class="col-12"><button type="submit" class="btn btn-primary w-100 py-2 fw-bold">Get Price Estimate &rarr;</button></div>
     </div>
    </form>
   </div>
   {% if result %}
    {% if result.error %}<div class="alert alert-danger">{{ result.error }}</div>
    {% else %}
    <div class="result-box">
     <div class="opacity-75 mb-1">Estimated resale value for {{ result.car }}</div>
     <div class="result-price">{{ result.price }}</div>
     <div class="mt-2">Market Range: {{ result.range }}</div>
     <small class="opacity-60 d-block mt-2">Based on {{ result.rows }} similar listings</small>
    </div>
    {% endif %}
   {% endif %}
  </div>
 </div>
</div>
""")

ANALYTICS_TMPL = BASE.replace("{% block style %}{% endblock %}","""
<style>
.hero{background:linear-gradient(135deg,#0f172a,#7c3aed);color:white;padding:40px 0;text-align:center}
.chart-card{background:white;border-radius:14px;padding:20px;box-shadow:0 2px 12px rgba(0,0,0,.06);margin-bottom:20px}
.chart-card h5{font-weight:700;color:#1e3a5f;margin-bottom:12px}
.chart-card img{width:100%;border-radius:8px}
</style>
""").replace("{% block content %}{% endblock %}","""
<div class="hero"><h2 class="fw-bold">Market Analytics</h2><p class="opacity-75">Insights from {{ total }} real CarWale listings</p></div>
<div class="container my-5">
 <div class="row">
  <div class="col-lg-6">
   <div class="chart-card"><h5>Top Brands by Listings</h5>
    {% if charts.brands %}<img src="data:image/png;base64,{{ charts.brands }}">{% endif %}</div>
  </div>
  <div class="col-lg-6">
   <div class="chart-card"><h5>Price Distribution</h5>
    {% if charts.prices %}<img src="data:image/png;base64,{{ charts.prices }}">{% endif %}</div>
  </div>
  <div class="col-lg-6">
   <div class="chart-card"><h5>Depreciation Curve</h5>
    {% if charts.depreciation %}<img src="data:image/png;base64,{{ charts.depreciation }}">{% endif %}</div>
  </div>
  <div class="col-lg-6">
   <div class="chart-card"><h5>Fuel Type Split</h5>
    {% if charts.fuel %}<img src="data:image/png;base64,{{ charts.fuel }}">{% endif %}</div>
  </div>
 </div>
</div>
""")

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template_string(HOME_TMPL, title="Home", active="home", stats=get_stats())

@app.route("/predict", methods=["GET","POST"])
def predict():
    opts = get_options()
    result = None
    if request.method == "POST":
        try:
            brand        = request.form.get("brand","")
            year         = int(request.form.get("year",2020))
            kms          = int(request.form.get("kms_driven",50000))
            fuel_type    = request.form.get("fuel_type","Petrol")
            transmission = request.form.get("transmission","Manual")
            owner        = request.form.get("owner","First Owner")
            car_age      = 2026 - year
            premium      = 1 if brand.lower() in ["bmw","audi","mercedes","porsche","jaguar","lexus","volvo"] else 0

            if model_bundle:
                from sklearn.preprocessing import LabelEncoder
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
                pred = model.predict(row[features])[0]
                result = {
                    "price": f"Rs {pred/100000:.2f} Lakhs",
                    "range": f"Rs {pred*0.9/100000:.1f}L - Rs {pred*1.1/100000:.1f}L",
                    "car":   f"{year} {brand}",
                    "rows":  len(df_global),
                }
            else:
                result = {"error": "Model not available. Please check your data file."}
        except Exception as e:
            result = {"error": str(e)}
    return render_template_string(PREDICT_TMPL, title="Predict", active="predict", opts=opts, result=result)

@app.route("/analytics")
def analytics():
    return render_template_string(ANALYTICS_TMPL, title="Analytics", active="analytics",
                                  charts=charts_cache, total=f"{len(df_global):,}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
