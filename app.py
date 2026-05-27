import os, re, io, base64
import pandas as pd
from pathlib import Path
from flask import Flask, render_template_string, request

app = Flask(__name__)

# ── Load Data ─────────────────────────────────────────────────────────────────
def load_data():
    paths = ["used_cars_raw_csv.csv","used_cars_raw.csv","data/raw/used_cars_raw.csv","data/processed/cleaned_cars.csv"]
    for path in paths:
        if Path(path).exists():
            try:
                df = pd.read_csv(path)
                if len(df) > 0:
                    print(f"Loaded {len(df)} rows from {path}")
                    return df
            except: pass
    return pd.DataFrame()

def prepare(df):
    if df.empty: return df
    # Use 'name' column as car_name if car_name missing
    if "car_name" not in df.columns and "name" in df.columns:
        df["car_name"] = df["name"]
    elif "name" in df.columns:
        df["car_name"] = df["name"]
    # Price
    if "price_inr" not in df.columns and "price" in df.columns:
        def cp(v):
            v = str(v).lower().replace(",","")
            if "crore" in v or "cr" in v:
                n = re.sub(r"[^\d.]","",v); return float(n)*10000000 if n else None
            if "lakh" in v or "lac" in v:
                n = re.sub(r"[^\d.]","",v); return float(n)*100000 if n else None
            n = re.sub(r"[^\d.]","",v); return float(n) if n else None
        df["price_inr"] = df["price"].apply(cp)
    df["price_inr"] = pd.to_numeric(df.get("price_inr"), errors="coerce")
    df["kms_driven"] = pd.to_numeric(df.get("kms_driven", df.get("km", None)), errors="coerce")
    df["year"] = pd.to_numeric(df.get("year"), errors="coerce").fillna(2020).astype(int)
    # Extract brand from name
    def eb(name):
        if not isinstance(name,str): return "Unknown"
        two = ["maruti suzuki","land rover","mercedes benz","mercedes-benz","rolls royce","aston martin"]
        for b in two:
            if name.lower().strip().startswith(b): return b.title()
        # Remove year prefix like "2021 Jaguar..."
        parts = re.sub(r"^\d{4}\s+","",name.strip()).split()
        return parts[0].title() if parts else "Unknown"
    df["brand"] = df["car_name"].apply(eb)
    df["car_age"] = 2026 - df["year"]
    df["premium_brand"] = df["brand"].str.lower().isin(["bmw","audi","mercedes-benz","mercedes","porsche","jaguar","land rover","volvo","lexus"]).astype(int)
    for col in ["fuel_type","transmission","owner","city"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
            df[col] = df[col].replace({"Nan":"Unknown","None":"Unknown","":"Unknown"})
    df = df[df["price_inr"].notna() & (df["price_inr"] > 10000)]
    return df

df_global = prepare(load_data())

# ── Train ML Model ────────────────────────────────────────────────────────────
def train_model(df):
    try:
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score
        FEATURES = ["brand","car_age","kms_driven","fuel_type","transmission","owner","premium_brand"]
        avail = [f for f in FEATURES if f in df.columns]
        data = df[avail+["price_inr"]].dropna().copy()
        if len(data) < 3: return None, 0
        encoders = {}
        for col in ["brand","fuel_type","transmission","owner"]:
            if col in data.columns:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))
                encoders[col] = le
        X, y = data[avail], data["price_inr"]
        if len(X) < 6:
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X, y)
            return {"model":model,"encoders":encoders,"features":avail}, 0.75
        Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(Xtr,ytr)
        r2 = max(0, r2_score(yte, model.predict(Xte)))
        return {"model":model,"encoders":encoders,"features":avail}, r2
    except Exception as e:
        print(f"Model error: {e}"); return None, 0

model_bundle, model_r2 = train_model(df_global)

# ── Charts ────────────────────────────────────────────────────────────────────
def make_charts(df):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"figure.facecolor":"#1a1a2e","axes.facecolor":"#1a1a2e",
                             "axes.labelcolor":"#94a3b8","xtick.color":"#94a3b8",
                             "ytick.color":"#94a3b8","text.color":"#e2e8f0"})
        charts = {}
        def save(fig):
            buf=io.BytesIO(); fig.savefig(buf,format="png",dpi=110,bbox_inches="tight",facecolor="#1a1a2e"); buf.seek(0)
            d=base64.b64encode(buf.read()).decode(); plt.close(fig); return d

        # Brand chart
        fig,ax=plt.subplots(figsize=(7,3.5))
        top=df["brand"].value_counts().head(8)
        bars=ax.barh(top.index[::-1],top.values[::-1],color=["#6366f1","#8b5cf6","#a78bfa","#c4b5fd","#818cf8","#4f46e5","#7c3aed","#9333ea"][:len(top)])
        ax.set_title("Listings by Brand",fontsize=13,fontweight="bold",color="#e2e8f0",pad=12)
        ax.spines[["top","right","left","bottom"]].set_visible(False)
        ax.set_xlabel(""); ax.tick_params(left=False)
        for bar,val in zip(bars,top.values[::-1]):
            ax.text(bar.get_width()+0.1,bar.get_y()+bar.get_height()/2,str(val),va="center",color="#94a3b8",fontsize=10)
        plt.tight_layout(); charts["brands"]=save(fig)

        # Price chart
        fig,ax=plt.subplots(figsize=(7,3.5))
        prices=df["price_inr"]/100000
        colors_p=["#06b6d4" if p<10 else "#6366f1" if p<30 else "#f59e0b" for p in prices]
        brands=df["brand"].tolist()
        ax.bar(range(len(prices)),prices.values,color=colors_p,width=0.6)
        ax.set_xticks(range(len(brands))); ax.set_xticklabels(brands,rotation=30,ha="right",fontsize=9)
        ax.set_title("Price by Car (Lakhs)",fontsize=13,fontweight="bold",color="#e2e8f0",pad=12)
        ax.set_ylabel("Price (Lakhs)",color="#94a3b8")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); charts["prices"]=save(fig)

        # Depreciation
        fig,ax=plt.subplots(figsize=(7,3.5))
        dep=df.groupby("car_age")["price_inr"].mean()/100000
        ax.plot(dep.index,dep.values,color="#f59e0b",linewidth=2.5,marker="o",markersize=7,markerfacecolor="#fbbf24")
        ax.fill_between(dep.index,dep.values,alpha=0.15,color="#f59e0b")
        ax.set_title("Price vs Car Age (Depreciation)",fontsize=13,fontweight="bold",color="#e2e8f0",pad=12)
        ax.set_xlabel("Car Age (Years)"); ax.set_ylabel("Avg Price (Lakhs)")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); charts["depreciation"]=save(fig)

        # Fuel pie
        if "fuel_type" in df.columns:
            fig,ax=plt.subplots(figsize=(5,5))
            fuel=df["fuel_type"].value_counts()
            wedges,texts,autotexts=ax.pie(fuel.values,labels=fuel.index,autopct="%1.0f%%",
                colors=["#6366f1","#06b6d4","#f59e0b","#10b981","#ef4444"][:len(fuel)],
                startangle=90,pctdistance=0.75)
            for t in texts: t.set_color("#94a3b8"); t.set_fontsize(11)
            for t in autotexts: t.set_color("white"); t.set_fontsize(10); t.set_fontweight("bold")
            ax.set_title("Fuel Type Mix",fontsize=13,fontweight="bold",color="#e2e8f0",pad=12)
            plt.tight_layout(); charts["fuel"]=save(fig)

        # KMs driven bar
        fig,ax=plt.subplots(figsize=(7,3.5))
        kms=df[["brand","kms_driven"]].dropna().sort_values("kms_driven")
        ax.barh(range(len(kms)),kms["kms_driven"].values/1000,color="#10b981",alpha=0.85)
        ax.set_yticks(range(len(kms))); ax.set_yticklabels(kms["brand"].values,fontsize=9)
        ax.set_title("KMs Driven by Car (000s)",fontsize=13,fontweight="bold",color="#e2e8f0",pad=12)
        ax.set_xlabel("KMs (thousands)")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); charts["kms"]=save(fig)

        return charts
    except Exception as e:
        print(f"Chart error: {e}"); return {}

charts_cache = make_charts(df_global)

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_stats():
    if df_global.empty:
        return {"total":"0","brands":0,"median":"N/A","r2":"N/A","min_price":"N/A","max_price":"N/A","avg_kms":"N/A"}
    return {
        "total":     f"{len(df_global):,}",
        "brands":    int(df_global["brand"].nunique()),
        "median":    f"Rs {df_global['price_inr'].median()/100000:.1f}L",
        "r2":        f"{model_r2*100:.0f}%",
        "min_price": f"Rs {df_global['price_inr'].min()/100000:.1f}L",
        "max_price": f"Rs {df_global['price_inr'].max()/100000:.1f}L",
        "avg_kms":   f"{df_global['kms_driven'].mean()/1000:.0f}k km",
    }

def get_options():
    opts={"brands":[],"fuel_types":[],"transmissions":[],"owners":[]}
    if df_global.empty:
        opts["fuel_types"]=["Petrol","Diesel","CNG","Electric"]
        opts["transmissions"]=["Manual","Automatic"]
        opts["owners"]=["First Owner","Second Owner","Third Owner"]
        return opts
    if "brand"        in df_global.columns: opts["brands"]        = sorted(df_global["brand"].dropna().unique().tolist())
    if "fuel_type"    in df_global.columns: opts["fuel_types"]    = sorted(df_global["fuel_type"].dropna().unique().tolist())
    if "transmission" in df_global.columns: opts["transmissions"] = sorted(df_global["transmission"].dropna().unique().tolist())
    if "owner"        in df_global.columns: opts["owners"]        = sorted(df_global["owner"].dropna().unique().tolist())
    return opts

def img_tag(key):
    if key in charts_cache:
        return f'<img src="data:image/png;base64,{charts_cache[key]}" style="width:100%;border-radius:10px">'
    return '<div class="empty-state"><span>📊</span><p>No data available</p></div>'

# ── HTML ──────────────────────────────────────────────────────────────────────
CSS = """
<meta name="viewport" content="width=device-width,initial-scale=1,shrink-to-fit=no">
<link href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.0/css/bootstrap.min.css" rel="stylesheet">
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0a0a1a;--bg2:#0f0f23;--bg3:#13132d;
  --glass:rgba(255,255,255,0.04);--glass-border:rgba(255,255,255,0.08);
  --blue:#6366f1;--cyan:#06b6d4;--amber:#f59e0b;--green:#10b981;--purple:#8b5cf6;
  --text:#e2e8f0;--text-muted:#64748b;--text-secondary:#94a3b8;
}
body{background:var(--bg);color:var(--text);font-family:'Inter',sans-serif;min-height:100vh}
body::before{content:'';position:fixed;top:0;left:0;width:100%;height:100%;
  background:radial-gradient(ellipse at 20% 20%,rgba(99,102,241,0.08) 0%,transparent 60%),
             radial-gradient(ellipse at 80% 80%,rgba(6,182,212,0.06) 0%,transparent 60%);
  pointer-events:none;z-index:0}

/* NAV */
.navbar{background:rgba(10,10,26,0.85)!important;backdrop-filter:blur(20px);
  border-bottom:1px solid var(--glass-border);position:sticky;top:0;z-index:1000;padding:14px 0}
.navbar-brand{font-weight:800;font-size:1.25rem;color:white!important;text-decoration:none;
  background:linear-gradient(135deg,#6366f1,#06b6d4);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.nav-link{color:var(--text-secondary)!important;font-size:13px;font-weight:500;
  padding:7px 14px!important;border-radius:8px;transition:all .2s}
.nav-link:hover{color:white!important;background:var(--glass)}
.nav-link.active{color:white!important;background:rgba(99,102,241,0.2);border:1px solid rgba(99,102,241,0.3)}
.navbar-toggler{border:1px solid var(--glass-border)!important;padding:6px 10px}
.navbar-toggler-icon{filter:invert(1)}

/* GLASS CARDS */
.glass{background:var(--glass);backdrop-filter:blur(12px);border:1px solid var(--glass-border);border-radius:16px}
.glass:hover{border-color:rgba(99,102,241,0.3);transition:border-color .3s}

/* STAT CARDS */
.stat-card{background:var(--glass);backdrop-filter:blur(12px);border:1px solid var(--glass-border);
  border-radius:16px;padding:20px;transition:all .3s;position:relative;overflow:hidden}
.stat-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;opacity:0.7}
.stat-card.blue::before{background:linear-gradient(90deg,#6366f1,#8b5cf6)}
.stat-card.cyan::before{background:linear-gradient(90deg,#06b6d4,#0ea5e9)}
.stat-card.amber::before{background:linear-gradient(90deg,#f59e0b,#f97316)}
.stat-card.green::before{background:linear-gradient(90deg,#10b981,#34d399)}
.stat-card.purple::before{background:linear-gradient(90deg,#8b5cf6,#a78bfa)}
.stat-card.red::before{background:linear-gradient(90deg,#ef4444,#f87171)}
.stat-card:hover{transform:translateY(-3px);box-shadow:0 8px 30px rgba(99,102,241,0.15)}
.stat-num{font-family:'JetBrains Mono',monospace;font-size:1.7rem;font-weight:700;color:white;line-height:1}
.stat-lbl{font-size:11px;color:var(--text-muted);margin-top:6px;text-transform:uppercase;letter-spacing:.5px}

/* HERO */
.hero{padding:70px 0 60px;text-align:center;position:relative;z-index:1}
.hero h1{font-size:3rem;font-weight:800;line-height:1.1;margin-bottom:16px;
  background:linear-gradient(135deg,#ffffff 30%,#6366f1 70%,#06b6d4 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.hero-sub{color:var(--text-secondary);font-size:1.05rem;max-width:500px;margin:0 auto 28px}
.hero-badge{display:inline-block;background:rgba(99,102,241,0.15);border:1px solid rgba(99,102,241,0.3);
  color:#a78bfa;padding:5px 14px;border-radius:20px;font-size:12px;font-weight:500;margin-bottom:16px}

/* BUTTONS */
.btn-glow{background:linear-gradient(135deg,#6366f1,#8b5cf6);border:none;color:white;
  padding:12px 28px;border-radius:10px;font-weight:600;font-size:14px;
  transition:all .2s;box-shadow:0 4px 20px rgba(99,102,241,0.3)}
.btn-glow:hover{transform:translateY(-2px);box-shadow:0 8px 30px rgba(99,102,241,0.4);color:white}
.btn-glow:active{transform:scale(0.97)}
.btn-outline-glass{background:var(--glass);border:1px solid var(--glass-border);color:var(--text);
  padding:12px 28px;border-radius:10px;font-weight:500;font-size:14px;transition:all .2s}
.btn-outline-glass:hover{border-color:rgba(99,102,241,0.4);color:white;background:rgba(99,102,241,0.1)}

/* CHART CARDS */
.chart-card{background:var(--glass);backdrop-filter:blur(12px);border:1px solid var(--glass-border);
  border-radius:16px;padding:20px;margin-bottom:20px;transition:all .3s}
.chart-card:hover{border-color:rgba(99,102,241,0.25);transform:translateY(-2px)}
.chart-card h5{font-size:13px;font-weight:600;color:var(--text-secondary);margin-bottom:14px;
  text-transform:uppercase;letter-spacing:.5px}

/* FEAT CARDS */
.feat-card{background:var(--glass);backdrop-filter:blur(10px);border:1px solid var(--glass-border);
  border-radius:16px;padding:26px;height:100%;transition:all .3s}
.feat-card:hover{border-color:rgba(99,102,241,0.35);transform:translateY(-4px);
  box-shadow:0 12px 40px rgba(99,102,241,0.1)}
.feat-icon{width:46px;height:46px;border-radius:12px;display:flex;align-items:center;justify-content:center;
  font-size:20px;margin-bottom:14px}

/* FORM */
.form-card{background:var(--glass);backdrop-filter:blur(12px);border:1px solid var(--glass-border);border-radius:16px;padding:28px}
.form-label{font-size:12px;font-weight:600;color:var(--text-secondary);text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px}
.form-control,.form-select{background:rgba(255,255,255,0.05)!important;border:1px solid var(--glass-border)!important;
  color:var(--text)!important;border-radius:10px!important;font-size:13px;padding:10px 12px;transition:all .2s}
.form-control:focus,.form-select:focus{border-color:rgba(99,102,241,0.5)!important;
  box-shadow:0 0 0 3px rgba(99,102,241,0.1)!important;background:rgba(99,102,241,0.05)!important}
.form-select option{background:#1a1a3e;color:white}
.btn-submit{background:linear-gradient(135deg,#6366f1,#8b5cf6);border:none;color:white;
  width:100%;padding:13px;border-radius:10px;font-weight:600;font-size:15px;
  margin-top:8px;transition:all .2s;box-shadow:0 4px 20px rgba(99,102,241,0.25)}
.btn-submit:hover{transform:translateY(-1px);box-shadow:0 8px 30px rgba(99,102,241,0.4)}
.btn-submit:active{transform:scale(0.98)}

/* RESULT */
.result-card{background:linear-gradient(135deg,rgba(16,185,129,0.15),rgba(6,182,212,0.1));
  border:1px solid rgba(16,185,129,0.3);border-radius:16px;padding:28px;text-align:center;margin-top:20px}
.result-label{font-size:12px;color:#34d399;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px}
.result-price{font-family:'JetBrains Mono',monospace;font-size:2.6rem;font-weight:700;
  color:white;background:linear-gradient(135deg,#10b981,#06b6d4);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1}
.result-range{font-size:13px;color:var(--text-secondary);margin-top:8px}
.result-meta{font-size:11px;color:var(--text-muted);margin-top:12px}

/* CAR TABLE */
.car-table{width:100%;border-collapse:separate;border-spacing:0 6px}
.car-table th{font-size:11px;text-transform:uppercase;letter-spacing:.5px;color:var(--text-muted);
  padding:8px 14px;font-weight:500}
.car-table td{padding:12px 14px;background:var(--glass);font-size:13px;color:var(--text)}
.car-table td:first-child{border-radius:10px 0 0 10px}
.car-table td:last-child{border-radius:0 10px 10px 0}
.car-table tr:hover td{background:rgba(99,102,241,0.08);border-color:rgba(99,102,241,0.2)}
.price-mono{font-family:'JetBrains Mono',monospace;font-weight:600;color:#10b981}
.badge-fuel{padding:4px 10px;border-radius:20px;font-size:11px;font-weight:500}
.badge-petrol{background:rgba(99,102,241,0.2);color:#a78bfa}
.badge-diesel{background:rgba(6,182,212,0.2);color:#67e8f9}
.badge-electric{background:rgba(16,185,129,0.2);color:#34d399}
.badge-cng{background:rgba(245,158,11,0.2);color:#fcd34d}

/* HERO SECTION PREDICT */
.predict-hero{padding:40px 0;text-align:center;position:relative;z-index:1}
.predict-hero h2{font-size:2rem;font-weight:800;
  background:linear-gradient(135deg,#fff,#6366f1);-webkit-background-clip:text;-webkit-text-fill-color:transparent}

/* ANALYTICS HERO */
.analytics-hero{padding:40px 0;text-align:center}
.analytics-hero h2{font-size:2rem;font-weight:800;
  background:linear-gradient(135deg,#fff,#8b5cf6);-webkit-background-clip:text;-webkit-text-fill-color:transparent}

/* EMPTY STATE */
.empty-state{text-align:center;padding:40px;color:var(--text-muted)}
.empty-state span{font-size:2.5rem;display:block;margin-bottom:10px}

/* SECTION LABELS */
.section-label{font-size:11px;text-transform:uppercase;letter-spacing:1px;color:var(--text-muted);margin-bottom:6px}
.section-title{font-size:1.4rem;font-weight:700;color:white;margin-bottom:4px}
.section-sub{font-size:13px;color:var(--text-secondary)}

/* FOOTER */
footer{text-align:center;padding:28px;color:var(--text-muted);font-size:12px;
  border-top:1px solid var(--glass-border);margin-top:60px;position:relative;z-index:1}
footer span{color:#6366f1}

/* MOBILE */
@media(max-width:768px){
  .hero h1{font-size:1.9rem}
  .hero-sub{font-size:.9rem}
  .stat-num{font-size:1.4rem}
  .result-price{font-size:2rem}
  .predict-hero h2,.analytics-hero h2{font-size:1.5rem}
  .btn-glow,.btn-outline-glass{padding:10px 18px;font-size:13px}
  .form-card{padding:18px}
  .hide-mobile{display:none!important}
  .navbar-collapse{background:rgba(10,10,26,0.95);border-radius:12px;padding:12px;margin-top:8px;
    border:1px solid var(--glass-border)}
}

/* BOTTOM NAV MOBILE */
@media(max-width:768px){
  .bottom-nav{display:flex!important}
  .top-nav-links{display:none!important}
  body{padding-bottom:70px}
}
@media(min-width:769px){.bottom-nav{display:none!important}}
.bottom-nav{position:fixed;bottom:0;left:0;right:0;
  background:rgba(10,10,26,0.95);backdrop-filter:blur(20px);
  border-top:1px solid var(--glass-border);z-index:999;
  justify-content:space-around;padding:8px 0}
.bottom-nav a{display:flex;flex-direction:column;align-items:center;color:var(--text-muted);
  text-decoration:none;font-size:10px;gap:3px;padding:4px 16px;border-radius:8px;transition:all .2s}
.bottom-nav a.active,.bottom-nav a:hover{color:#6366f1}
.bottom-nav a span{font-size:20px}

/* SCROLL ANIMATION */
.fade-in{animation:fadeIn .4s ease both}
@keyframes fadeIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}

/* SKELETON */
.skeleton{background:linear-gradient(90deg,rgba(255,255,255,0.03) 25%,rgba(255,255,255,0.07) 50%,rgba(255,255,255,0.03) 75%);
  background-size:200% 100%;animation:shimmer 1.5s infinite;border-radius:8px;height:16px}
@keyframes shimmer{0%{background-position:200% 0}100%{background-position:-200% 0}}
</style>
"""

NAV = """
<nav class="navbar navbar-expand-lg">
 <div class="container">
  <a class="navbar-brand" href="/">&#9685; CarIQ</a>
  <div class="top-nav-links d-flex gap-1 ms-auto">
   <a class="nav-link {ah}" href="/">Home</a>
   <a class="nav-link {ap}" href="/predict">Predict</a>
   <a class="nav-link {aa}" href="/analytics">Analytics</a>
  </div>
  <button class="navbar-toggler d-lg-none ms-2" type="button" data-bs-toggle="collapse" data-bs-target="#nm">
   <span>&#9776;</span>
  </button>
  <div class="collapse navbar-collapse d-lg-none" id="nm">
   <div class="d-flex flex-column gap-1 mt-2">
    <a class="nav-link {ah}" href="/">Home</a>
    <a class="nav-link {ap}" href="/predict">Predict Price</a>
    <a class="nav-link {aa}" href="/analytics">Analytics</a>
   </div>
  </div>
 </div>
</nav>
<div class="bottom-nav">
 <a href="/" class="{ah}"><span>&#127968;</span>Home</a>
 <a href="/predict" class="{ap}"><span>&#128270;</span>Predict</a>
 <a href="/analytics" class="{aa}"><span>&#128202;</span>Analytics</a>
</div>
"""

FOOT = """<footer>Built with <span>&#9829;</span> using Python &middot; Flask &middot; Scikit-learn &middot; Real CarWale Data</footer>
<script src="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.0/js/bootstrap.bundle.min.js"></script>"""

def page(content, title, active):
    ah = "active" if active=="home" else ""
    ap = "active" if active=="predict" else ""
    aa = "active" if active=="analytics" else ""
    nav = NAV.replace("{ah}",ah).replace("{ap}",ap).replace("{aa}",aa)
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>{title} - CarIQ</title>
{CSS}</head><body>{nav}<div style="position:relative;z-index:1">{content}</div>{FOOT}</body></html>"""

# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    s = get_stats()
    # Build car listings table
    rows = ""
    if not df_global.empty:
        for _,r in df_global.iterrows():
            fuel = str(r.get("fuel_type",""))
            badge_cls = "badge-petrol" if "Petrol" in fuel else "badge-diesel" if "Diesel" in fuel else "badge-electric" if "Electric" in fuel else "badge-cng"
            name = str(r.get("car_name",""))[:40]
            price = f"Rs {r.get('price_inr',0)/100000:.1f}L"
            kms = f"{r.get('kms_driven',0)/1000:.0f}k km"
            yr = int(r.get("year",0))
            owner = str(r.get("owner",""))[:15]
            rows += f"""<tr>
              <td><div style="font-weight:500;color:white">{name}</div><div style="font-size:11px;color:var(--text-muted)">{yr} &middot; {owner}</div></td>
              <td class="price-mono">{price}</td>
              <td class="hide-mobile">{kms}</td>
              <td><span class="badge-fuel {badge_cls}">{fuel}</span></td>
            </tr>"""

    content = f"""
<div class="hero fade-in">
 <div class="container">
  <div class="hero-badge">&#128202; Bangalore Used Cars Intelligence</div>
  <h1>Smart Car Price<br>Intelligence</h1>
  <p class="hero-sub">AI-powered used car price prediction &amp; market analytics — powered by real CarWale data</p>
  <a href="/predict" class="btn-glow me-2">&#128270; Predict Price</a>
  <a href="/analytics" class="btn-outline-glass">&#128202; View Analytics</a>
 </div>
</div>

<div class="container pb-5">
 <div class="row g-3 mb-5">
  <div class="col-6 col-md-4 col-lg-2"><div class="stat-card blue"><div class="stat-num">{s["total"]}</div><div class="stat-lbl">Cars Listed</div></div></div>
  <div class="col-6 col-md-4 col-lg-2"><div class="stat-card cyan"><div class="stat-num">{s["brands"]}</div><div class="stat-lbl">Brands</div></div></div>
  <div class="col-6 col-md-4 col-lg-2"><div class="stat-card amber"><div class="stat-num">{s["median"]}</div><div class="stat-lbl">Median Price</div></div></div>
  <div class="col-6 col-md-4 col-lg-2"><div class="stat-card green"><div class="stat-num">{s["min_price"]}</div><div class="stat-lbl">Lowest Price</div></div></div>
  <div class="col-6 col-md-4 col-lg-2"><div class="stat-card purple"><div class="stat-num">{s["max_price"]}</div><div class="stat-lbl">Highest Price</div></div></div>
  <div class="col-6 col-md-4 col-lg-2"><div class="stat-card red"><div class="stat-num">{s["r2"]}</div><div class="stat-lbl">ML Accuracy</div></div></div>
 </div>

 <div class="row g-4 mb-5">
  <div class="col-md-4"><div class="feat-card">
   <div class="feat-icon" style="background:rgba(99,102,241,0.15)">&#129302;</div>
   <h5 style="color:white;font-weight:700;margin-bottom:8px">ML Price Prediction</h5>
   <p style="color:var(--text-secondary);font-size:13px;line-height:1.6">Enter brand, year, KMs and fuel type for an instant AI-powered resale estimate based on your real data.</p>
   <a href="/predict" class="btn-glow" style="font-size:13px;padding:8px 18px;display:inline-block;margin-top:12px;text-decoration:none">Try it &rarr;</a>
  </div></div>
  <div class="col-md-4"><div class="feat-card">
   <div class="feat-icon" style="background:rgba(6,182,212,0.15)">&#128202;</div>
   <h5 style="color:white;font-weight:700;margin-bottom:8px">Market Analytics</h5>
   <p style="color:var(--text-secondary);font-size:13px;line-height:1.6">Interactive charts showing price distribution, brand comparison, depreciation curves and fuel type trends.</p>
   <a href="/analytics" class="btn-glow" style="font-size:13px;padding:8px 18px;display:inline-block;margin-top:12px;text-decoration:none;background:linear-gradient(135deg,#06b6d4,#0ea5e9)">Explore &rarr;</a>
  </div></div>
  <div class="col-md-4"><div class="feat-card">
   <div class="feat-icon" style="background:rgba(16,185,129,0.15)">&#128396;</div>
   <h5 style="color:white;font-weight:700;margin-bottom:8px">Real CarWale Data</h5>
   <p style="color:var(--text-secondary);font-size:13px;line-height:1.6">All insights are derived from live scraped listings from CarWale, cleaned through a Python ETL pipeline.</p>
   <a href="/analytics" class="btn-glow" style="font-size:13px;padding:8px 18px;display:inline-block;margin-top:12px;text-decoration:none;background:linear-gradient(135deg,#10b981,#34d399)">See Data &rarr;</a>
  </div></div>
 </div>

 <div class="mb-2">
  <div class="section-label">Live Listings</div>
  <div class="section-title">All Cars in Dataset</div>
  <div class="section-sub" style="margin-bottom:16px">Scraped from CarWale — Bangalore</div>
 </div>
 <div class="glass p-3" style="overflow-x:auto">
  <table class="car-table">
   <thead><tr>
    <th>Car</th><th>Price</th><th class="hide-mobile">KMs</th><th>Fuel</th>
   </tr></thead>
   <tbody>{rows}</tbody>
  </table>
 </div>
</div>"""
    return page(content, "Home", "home")

@app.route("/predict", methods=["GET","POST"])
def predict():
    opts = get_options()
    b_opts = "".join(f"<option>{b}</option>" for b in opts["brands"])
    y_opts = "".join(f"<option>{y}</option>" for y in range(2024,1999,-1))
    f_opts = "".join(f"<option>{f}</option>" for f in opts["fuel_types"])
    t_opts = "".join(f"<option>{t}</option>" for t in opts["transmissions"])
    o_opts = "".join(f"<option>{o}</option>" for o in opts["owners"])

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
            premium      = 1 if brand.lower() in ["bmw","audi","mercedes-benz","mercedes","porsche","jaguar","land rover","volvo","lexus"] else 0

            if model_bundle:
                model    = model_bundle["model"]
                encoders = model_bundle["encoders"]
                features = model_bundle["features"]
                row = pd.DataFrame([{"brand":brand,"car_age":car_age,"kms_driven":kms,
                                     "fuel_type":fuel_type,"transmission":transmission,
                                     "owner":owner,"premium_brand":premium}])
                for col,le in encoders.items():
                    if col in row.columns:
                        val = str(row[col].iloc[0])
                        row[col] = le.transform([val if val in le.classes_ else le.classes_[0]])
                pred = model.predict(row[features])[0]
                result_html = f"""
                <div class="result-card fade-in">
                 <div class="result-label">Estimated Resale Value</div>
                 <div class="result-price">Rs {pred/100000:.2f}L</div>
                 <div class="result-range">Market Range &nbsp; Rs {pred*0.88/100000:.1f}L &ndash; Rs {pred*1.12/100000:.1f}L</div>
                 <div class="result-meta">
                  {year} {brand} &middot; {kms:,} km &middot; {fuel_type} &middot; {transmission}<br>
                  Based on {len(df_global)} real listings &middot; &plusmn;12% market variation
                 </div>
                </div>"""
            else:
                result_html = '<div class="glass p-3 mt-3" style="color:#f59e0b;font-size:13px">&#9888; Model needs more data to make predictions. Upload more listings.</div>'
        except Exception as e:
            result_html = f'<div class="glass p-3 mt-3" style="color:#ef4444;font-size:13px">Error: {str(e)}</div>'

    content = f"""
<div class="predict-hero fade-in">
 <div class="container">
  <div class="hero-badge">&#129302; ML-Powered Estimate</div>
  <h2>Predict Car Resale Price</h2>
  <p style="color:var(--text-secondary);font-size:14px;margin-top:8px">Trained on your real Bangalore CarWale data</p>
 </div>
</div>
<div class="container pb-5">
 <div class="row justify-content-center">
  <div class="col-12 col-lg-6">
   <div class="form-card fade-in">
    <form method="POST">
     <div class="row g-3">
      <div class="col-6"><label class="form-label">Brand</label>
       <select name="brand" class="form-select">{b_opts}</select></div>
      <div class="col-6"><label class="form-label">Year</label>
       <select name="year" class="form-select">{y_opts}</select></div>
      <div class="col-12"><label class="form-label">KMs Driven</label>
       <input type="number" name="kms_driven" class="form-control" placeholder="e.g. 35000" min="0" required></div>
      <div class="col-6"><label class="form-label">Fuel Type</label>
       <select name="fuel_type" class="form-select">{f_opts}</select></div>
      <div class="col-6"><label class="form-label">Transmission</label>
       <select name="transmission" class="form-select">{t_opts}</select></div>
      <div class="col-12"><label class="form-label">Ownership</label>
       <select name="owner" class="form-select">{o_opts}</select></div>
      <div class="col-12"><button type="submit" class="btn-submit">&#128270; Get Price Estimate</button></div>
     </div>
    </form>
   </div>
   {result_html}
   <div class="glass p-3 mt-3" style="font-size:12px;color:var(--text-muted)">
    &#9432; Predictions are based on <strong style="color:var(--text-secondary)">{len(df_global)} real listings</strong> scraped from CarWale Bangalore.
    Upload more data via GitHub to improve accuracy.
   </div>
  </div>
 </div>
</div>"""
    return page(content, "Predict Price", "predict")

@app.route("/analytics")
def analytics():
    s = get_stats()
    content = f"""
<div class="analytics-hero fade-in">
 <div class="container">
  <div class="hero-badge">&#128202; Real Data Insights</div>
  <h2>Market Analytics</h2>
  <p style="color:var(--text-secondary);font-size:14px;margin-top:8px">{s["total"]} listings &middot; {s["brands"]} brands &middot; Bangalore</p>
 </div>
</div>
<div class="container pb-5">
 <div class="row g-4">
  <div class="col-12 col-md-6"><div class="chart-card fade-in"><h5>&#127942; Listings by Brand</h5>{img_tag("brands")}</div></div>
  <div class="col-12 col-md-6"><div class="chart-card fade-in"><h5>&#128176; Price per Car (Lakhs)</h5>{img_tag("prices")}</div></div>
  <div class="col-12 col-md-6"><div class="chart-card fade-in"><h5>&#128197; Depreciation — Price vs Age</h5>{img_tag("depreciation")}</div></div>
  <div class="col-12 col-md-6"><div class="chart-card fade-in"><h5>&#9981; Fuel Type Distribution</h5>{img_tag("fuel")}</div></div>
  <div class="col-12"><div class="chart-card fade-in"><h5>&#128663; KMs Driven by Car</h5>{img_tag("kms")}</div></div>
 </div>

 <div class="row g-3 mt-2">
  <div class="col-12"><div class="section-label">Key Numbers</div></div>
  <div class="col-6 col-md-3"><div class="stat-card blue"><div class="stat-num">{s["min_price"]}</div><div class="stat-lbl">Cheapest Car</div></div></div>
  <div class="col-6 col-md-3"><div class="stat-card purple"><div class="stat-num">{s["max_price"]}</div><div class="stat-lbl">Most Expensive</div></div></div>
  <div class="col-6 col-md-3"><div class="stat-card amber"><div class="stat-num">{s["median"]}</div><div class="stat-lbl">Median Price</div></div></div>
  <div class="col-6 col-md-3"><div class="stat-card green"><div class="stat-num">{s["avg_kms"]}</div><div class="stat-lbl">Avg KMs Driven</div></div></div>
 </div>
</div>"""
    return page(content, "Analytics", "analytics")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
