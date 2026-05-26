import os, re, io, base64
import pandas as pd
from pathlib import Path
from flask import Flask, render_template_string, request

app = Flask(__name__)

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
    {"car_name":"Maruti Suzuki Wagon R","brand":"Maruti Suzuki","year":2021,"price_inr":480000,"kms_driven":15000,"fuel_type":"CNG","transmission":"Manual","owner":"First Owner","car_age":5,"premium_brand":0},
    {"car_name":"Hyundai Grand i10","brand":"Hyundai","year":2019,"price_inr":420000,"kms_driven":32000,"fuel_type":"Petrol","transmission":"Manual","owner":"First Owner","car_age":7,"premium_brand":0},
    {"car_name":"Tata Tiago","brand":"Tata","year":2020,"price_inr":390000,"kms_driven":28000,"fuel_type":"Petrol","transmission":"Manual","owner":"First Owner","car_age":6,"premium_brand":0},
    {"car_name":"Renault Kwid","brand":"Renault","year":2020,"price_inr":310000,"kms_driven":22000,"fuel_type":"Petrol","transmission":"Manual","owner":"First Owner","car_age":6,"premium_brand":0},
    {"car_name":"Volkswagen Polo","brand":"Volkswagen","year":2019,"price_inr":650000,"kms_driven":38000,"fuel_type":"Petrol","transmission":"Manual","owner":"Second Owner","car_age":7,"premium_brand":0},
]

def load_data():
    paths = ["data/processed/cleaned_cars.csv","data/raw/used_cars_raw.csv","used_cars_raw.csv","cleaned_cars.csv"]
    for path in paths:
        if Path(path).exists():
            try:
                df = pd.read_csv(path)
                if len(df) > 5:
                    print(f"Loaded {len(df)} rows from {path}")
                    return df
            except:
                pass
    print("Using sample data")
    return pd.DataFrame(SAMPLE_DATA)

def ensure_columns(df):
    rename_map = {}
    for col in df.columns:
        cl = col.lower().strip()
        if ("name" in cl and "car" in cl) or cl == "name": rename_map[col] = "car_name"
        elif "price" in cl and "raw" not in cl:            rename_map[col] = "price_inr"
        elif "price" in cl:                                rename_map[col] = "price_raw"
        elif "km" in cl or "odometer" in cl:               rename_map[col] = "kms_driven"
        elif "year" in cl and "raw" not in cl:             rename_map[col] = "year"
        elif "fuel" in cl:                                 rename_map[col] = "fuel_type"
        elif "trans" in cl:                                rename_map[col] = "transmission"
        elif "owner" in cl:                                rename_map[col] = "owner"
        elif "city" in cl or "location" in cl:             rename_map[col] = "city"
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
    for col in ["price_inr","kms_driven","year"]:
        if col not in df.columns: df[col] = None
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "year" not in df.columns or df["year"].isna().all(): df["year"] = 2020
    if "brand" not in df.columns:
        def eb(name):
            if not isinstance(name,str): return "Unknown"
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

df_global = ensure_columns(load_data())

def train_model(df):
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score
        FEATURES = ["brand","car_age","kms_driven","fuel_type","transmission","owner","premium_brand"]
        avail = [f for f in FEATURES if f in df.columns]
        data = df[avail+["price_inr"]].dropna()
        if len(data) < 10: return None, 0
        encoders = {}
        for col in ["brand","fuel_type","transmission","owner"]:
            if col in data.columns:
                le = LabelEncoder()
                data = data.copy()
                data[col] = le.fit_transform(data[col].astype(str))
                encoders[col] = le
        X, y = data[avail], data["price_inr"]
        Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
        model = RandomForestRegressor(n_estimators=100,random_state=42,n_jobs=-1)
        model.fit(Xtr,ytr)
        r2 = r2_score(yte,model.predict(Xte))
        print(f"Model R2={r2:.3f}")
        return {"model":model,"encoders":encoders,"features":avail}, r2
    except Exception as e:
        print(f"Model error: {e}")
        return None, 0

model_bundle, model_r2 = train_model(df_global)

def make_charts(df):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        charts = {}
        def save(fig):
            buf = io.BytesIO(); fig.savefig(buf,format="png",dpi=100,bbox_inches="tight"); buf.seek(0)
            data = base64.b64encode(buf.read()).decode(); plt.close(fig); return data
        fig,ax = plt.subplots(figsize=(7,3.5)); top=df["brand"].value_counts().head(8)
        ax.barh(top.index[::-1],top.values[::-1],color="#2563eb"); ax.set_title("Top Brands",fontweight="bold"); ax.spines[["top","right"]].set_visible(False); charts["brands"]=save(fig)
        fig,ax = plt.subplots(figsize=(7,3.5)); prices=df[df["price_inr"]<5000000]["price_inr"]/100000
        ax.hist(prices,bins=30,color="#16a34a",edgecolor="white"); ax.set_title("Price Distribution (Lakhs)",fontweight="bold"); ax.spines[["top","right"]].set_visible(False); charts["prices"]=save(fig)
        fig,ax = plt.subplots(figsize=(7,3.5)); dep=df[df["car_age"]<=15].groupby("car_age")["price_inr"].mean()/100000
        ax.plot(dep.index,dep.values,color="#dc2626",linewidth=2,marker="o",markersize=4); ax.set_title("Depreciation Curve",fontweight="bold"); ax.set_xlabel("Car Age (Years)"); ax.spines[["top","right"]].set_visible(False); charts["depreciation"]=save(fig)
        if "fuel_type" in df.columns:
            fig,ax = plt.subplots(figsize=(5,5)); fuel=df["fuel_type"].value_counts()
            ax.pie(fuel.values,labels=fuel.index,autopct="%1.1f%%",colors=["#2563eb","#16a34a","#d97706","#7c3aed","#dc2626"][:len(fuel)]); ax.set_title("Fuel Type Split",fontweight="bold"); charts["fuel"]=save(fig)
        return charts
    except Exception as e:
        print(f"Chart error: {e}"); return {}

charts_cache = make_charts(df_global)

def get_stats():
    return {
        "total":  f"{len(df_global):,}",
        "brands": int(df_global["brand"].nunique()) if "brand" in df_global.columns else 0,
        "median": f"Rs {df_global['price_inr'].median()/100000:.1f}L",
        "r2":     f"{model_r2*100:.1f}%",
    }

def get_options():
    opts = {"brands":[],"fuel_types":[],"transmissions":[],"owners":[]}
    if "brand"        in df_global.columns: opts["brands"]        = sorted(df_global["brand"].dropna().unique().tolist())
    if "fuel_type"    in df_global.columns: opts["fuel_types"]    = sorted(df_global["fuel_type"].dropna().unique().tolist())
    if "transmission" in df_global.columns: opts["transmissions"] = sorted(df_global["transmission"].dropna().unique().tolist())
    if "owner"        in df_global.columns: opts["owners"]        = sorted(df_global["owner"].dropna().unique().tolist())
    if not opts["fuel_types"]:    opts["fuel_types"]    = ["Petrol","Diesel","CNG","Electric"]
    if not opts["transmissions"]: opts["transmissions"] = ["Manual","Automatic"]
    if not opts["owners"]:        opts["owners"]        = ["First Owner","Second Owner","Third Owner"]
    return opts

BASE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,shrink-to-fit=no">
<title>{{ title }} - CarIQ</title>
<link href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.0/css/bootstrap.min.css" rel="stylesheet">
<style>
*{box-sizing:border-box}
body{background:#f8fafc;font-family:'Segoe UI',sans-serif;margin:0}
.navbar{background:#0f172a!important;padding:12px 0}
.navbar-brand{font-weight:800;font-size:1.3rem;color:white!important;text-decoration:none}
.nav-link{color:rgba(255,255,255,.7)!important;padding:6px 12px!important;border-radius:6px;font-size:14px}
.nav-link:hover,.nav-link.active{color:white!important;background:rgba(255,255,255,.1)!important}
.card{border:none!important;border-radius:14px;box-shadow:0 2px 12px rgba(0,0,0,.06)}
.btn-primary{background:#2563eb;border-color:#2563eb}
.btn-primary:hover{background:#1d4ed8;border-color:#1d4ed8}
footer{text-align:center;padding:24px;color:#94a3b8;font-size:13px;margin-top:40px;border-top:1px solid #e2e8f0}
@media(max-width:768px){
  .hero h1{font-size:1.8rem!important}
  .stat-num{font-size:1.6rem!important}
  .result-price{font-size:2rem!important}
  .btn-lg{padding:10px 18px;font-size:14px}
}
</style>
EXTRA_STYLE
</head>
<body>
<nav class="navbar navbar-expand-lg">
 <div class="container">
  <a class="navbar-brand" href="/">&#128664; CarIQ</a>
  <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navmenu" style="border-color:rgba(255,255,255,.3)">
   <span style="color:white;font-size:20px">&#9776;</span>
  </button>
  <div class="collapse navbar-collapse" id="navmenu">
   <ul class="navbar-nav ms-auto">
    <li class="nav-item"><a class="nav-link ACTIVE_HOME" href="/">Home</a></li>
    <li class="nav-item"><a class="nav-link ACTIVE_PREDICT" href="/predict">Predict Price</a></li>
    <li class="nav-item"><a class="nav-link ACTIVE_ANALYTICS" href="/analytics">Analytics</a></li>
   </ul>
  </div>
 </div>
</nav>
PAGE_CONTENT
<footer>Built with Python &middot; Flask &middot; Scikit-learn &middot; CarWale Data</footer>
<script src="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.0/js/bootstrap.bundle.min.js"></script>
</body>
</html>"""

HOME_STYLE = """
<style>
.hero{background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 100%);color:white;padding:70px 0 60px;text-align:center}
.hero h1{font-size:2.6rem;font-weight:800;margin-bottom:12px}
.stat-card{border-radius:14px;padding:20px;text-align:center;color:white;height:100%}
.stat-num{font-size:1.9rem;font-weight:800}
.stat-lbl{font-size:12px;opacity:.85;margin-top:4px}
.feat-card{background:white;border-radius:14px;padding:24px;box-shadow:0 2px 12px rgba(0,0,0,.06);height:100%}
.feat-icon{font-size:2rem;margin-bottom:10px}
</style>"""

HOME_CONTENT = """
<div class="hero">
 <div class="container">
  <h1>Smart Car Price Intelligence</h1>
  <p class="opacity-75 mb-4 fs-5">AI-powered used car price prediction for India</p>
  <a href="/predict" class="btn btn-primary btn-lg me-2 mb-2">&#128270; Predict Price</a>
  <a href="/analytics" class="btn btn-outline-light btn-lg mb-2">&#128202; View Analytics</a>
 </div>
</div>
<div class="container my-5">
 <div class="row g-3 mb-5">
  <div class="col-6 col-md-3"><div class="stat-card" style="background:#2563eb"><div class="stat-num">STAT_TOTAL</div><div class="stat-lbl">Cars Analysed</div></div></div>
  <div class="col-6 col-md-3"><div class="stat-card" style="background:#16a34a"><div class="stat-num">STAT_BRANDS</div><div class="stat-lbl">Unique Brands</div></div></div>
  <div class="col-6 col-md-3"><div class="stat-card" style="background:#d97706"><div class="stat-num">STAT_MEDIAN</div><div class="stat-lbl">Median Price</div></div></div>
  <div class="col-6 col-md-3"><div class="stat-card" style="background:#7c3aed"><div class="stat-num">STAT_R2</div><div class="stat-lbl">Model Accuracy</div></div></div>
 </div>
 <h4 class="fw-bold mb-4 text-center">What CarIQ Does</h4>
 <div class="row g-4">
  <div class="col-md-4">
   <div class="feat-card">
    <div class="feat-icon">&#129302;</div>
    <h5 class="fw-bold">ML Price Prediction</h5>
    <p class="text-muted small">Enter brand, year, KMs and fuel type to get an instant AI-powered resale price estimate.</p>
    <a href="/predict" class="btn btn-primary btn-sm">Try it &rarr;</a>
   </div>
  </div>
  <div class="col-md-4">
   <div class="feat-card">
    <div class="feat-icon">&#128202;</div>
    <h5 class="fw-bold">Market Analytics</h5>
    <p class="text-muted small">Explore price trends, depreciation curves, fuel splits and brand comparisons from real data.</p>
    <a href="/analytics" class="btn btn-primary btn-sm">Explore &rarr;</a>
   </div>
  </div>
  <div class="col-md-4">
   <div class="feat-card">
    <div class="feat-icon">&#128396;</div>
    <h5 class="fw-bold">Real Scraped Data</h5>
    <p class="text-muted small">All data scraped live from CarWale using Playwright, cleaned through a Python ETL pipeline.</p>
    <a href="/analytics" class="btn btn-primary btn-sm">See Data &rarr;</a>
   </div>
  </div>
 </div>
</div>"""

PREDICT_STYLE = """
<style>
.hero{background:linear-gradient(135deg,#1e3a5f,#2563eb);color:white;padding:40px 0;text-align:center}
.result-box{background:linear-gradient(135deg,#16a34a,#15803d);color:white;border-radius:16px;padding:28px;text-align:center}
.result-price{font-size:2.8rem;font-weight:800;margin:8px 0}
.form-label{font-weight:600;font-size:13px;color:#374151}
</style>"""

PREDICT_CONTENT = """
<div class="hero">
 <div class="container">
  <h2 class="fw-bold">&#128270; Predict Car Resale Price</h2>
  <p class="opacity-75 mt-2">Fill in the details for an instant ML-powered price estimate</p>
 </div>
</div>
<div class="container my-4">
 <div class="row justify-content-center">
  <div class="col-12 col-lg-7">
   <div class="card p-4 mb-4">
    <form method="POST">
     <div class="row g-3">
      <div class="col-6"><label class="form-label">Brand</label>
       <select name="brand" class="form-select form-select-sm">BRAND_OPTIONS</select></div>
      <div class="col-6"><label class="form-label">Year</label>
       <select name="year" class="form-select form-select-sm">YEAR_OPTIONS</select></div>
      <div class="col-6"><label class="form-label">KMs Driven</label>
       <input type="number" name="kms_driven" class="form-control form-control-sm" placeholder="e.g. 45000" required></div>
      <div class="col-6"><label class="form-label">Fuel Type</label>
       <select name="fuel_type" class="form-select form-select-sm">FUEL_OPTIONS</select></div>
      <div class="col-6"><label class="form-label">Transmission</label>
       <select name="transmission" class="form-select form-select-sm">TRANS_OPTIONS</select></div>
      <div class="col-6"><label class="form-label">Ownership</label>
       <select name="owner" class="form-select form-select-sm">OWNER_OPTIONS</select></div>
      <div class="col-12 mt-2">
       <button type="submit" class="btn btn-primary w-100 py-2 fw-bold">Get Price Estimate &rarr;</button>
      </div>
     </div>
    </form>
   </div>
   RESULT_HTML
  </div>
 </div>
</div>"""

ANALYTICS_STYLE = """
<style>
.hero{background:linear-gradient(135deg,#0f172a,#7c3aed);color:white;padding:40px 0;text-align:center}
.chart-card{background:white;border-radius:14px;padding:16px;box-shadow:0 2px 12px rgba(0,0,0,.06);margin-bottom:20px}
.chart-card h5{font-weight:700;color:#1e3a5f;margin-bottom:10px;font-size:14px}
.chart-card img{width:100%;border-radius:8px}
</style>"""

ANALYTICS_CONTENT = """
<div class="hero">
 <div class="container">
  <h2 class="fw-bold">&#128202; Market Analytics</h2>
  <p class="opacity-75 mt-2">Insights from TOTAL_ROWS real CarWale listings</p>
 </div>
</div>
<div class="container my-4">
 <div class="row g-3">
  <div class="col-12 col-md-6"><div class="chart-card"><h5>&#127942; Top Brands by Listings</h5>CHART_BRANDS</div></div>
  <div class="col-12 col-md-6"><div class="chart-card"><h5>&#128176; Price Distribution (Lakhs)</h5>CHART_PRICES</div></div>
  <div class="col-12 col-md-6"><div class="chart-card"><h5>&#128197; Depreciation Curve</h5>CHART_DEP</div></div>
  <div class="col-12 col-md-6"><div class="chart-card"><h5>&#9981; Fuel Type Split</h5>CHART_FUEL</div></div>
 </div>
</div>"""

def render_page(style, content, title, active):
    page = BASE_HTML.replace("EXTRA_STYLE", style).replace("PAGE_CONTENT", content).replace("{{ title }}", title)
    page = page.replace("ACTIVE_HOME","active" if active=="home" else "")
    page = page.replace("ACTIVE_PREDICT","active" if active=="predict" else "")
    page = page.replace("ACTIVE_ANALYTICS","active" if active=="analytics" else "")
    return page

@app.route("/")
def home():
    s = get_stats()
    content = HOME_CONTENT
    content = content.replace("STAT_TOTAL", s["total"]).replace("STAT_BRANDS", str(s["brands"]))
    content = content.replace("STAT_MEDIAN", s["median"]).replace("STAT_R2", s["r2"])
    return render_page(HOME_STYLE, content, "Home", "home")

@app.route("/predict", methods=["GET","POST"])
def predict():
    opts = get_options()
    brand_opts = "".join(f"<option>{b}</option>" for b in opts["brands"])
    year_opts  = "".join(f"<option>{y}</option>" for y in range(2024,1999,-1))
    fuel_opts  = "".join(f"<option>{f}</option>" for f in opts["fuel_types"])
    trans_opts = "".join(f"<option>{t}</option>" for t in opts["transmissions"])
    owner_opts = "".join(f"<option>{o}</option>" for o in opts["owners"])

    result_html = ""
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
                result_html = f"""
                <div class="result-box">
                 <div class="opacity-75 mb-1" style="font-size:14px">Estimated resale value for {year} {brand}</div>
                 <div class="result-price">Rs {pred/100000:.2f} Lakhs</div>
                 <div class="mt-2" style="font-size:14px">Market Range: Rs {pred*0.9/100000:.1f}L &ndash; Rs {pred*1.1/100000:.1f}L</div>
                 <small class="d-block mt-3 opacity-60">Based on {len(df_global):,} listings &middot; &plusmn;10% reflects market variation</small>
                </div>"""
            else:
                result_html = '<div class="alert alert-warning">Model not ready. Please check your data file.</div>'
        except Exception as e:
            result_html = f'<div class="alert alert-danger">Error: {str(e)}</div>'

    content = PREDICT_CONTENT
    content = content.replace("BRAND_OPTIONS",brand_opts).replace("YEAR_OPTIONS",year_opts)
    content = content.replace("FUEL_OPTIONS",fuel_opts).replace("TRANS_OPTIONS",trans_opts)
    content = content.replace("OWNER_OPTIONS",owner_opts).replace("RESULT_HTML",result_html)
    return render_page(PREDICT_STYLE, content, "Predict Price", "predict")

@app.route("/analytics")
def analytics():
    def img(key):
        if key in charts_cache:
            return f'<img src="data:image/png;base64,{charts_cache[key]}" style="width:100%;border-radius:8px">'
        return '<p class="text-muted small">Chart not available</p>'
    content = ANALYTICS_CONTENT
    content = content.replace("TOTAL_ROWS", f"{len(df_global):,}")
    content = content.replace("CHART_BRANDS", img("brands"))
    content = content.replace("CHART_PRICES", img("prices"))
    content = content.replace("CHART_DEP",    img("depreciation"))
    content = content.replace("CHART_FUEL",   img("fuel"))
    return render_page(ANALYTICS_STYLE, content, "Analytics", "analytics")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
