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
    val = val.replace(",","").replace("Rs","").replace("rs","").strip()
    if "lakh" in val or "lac" in val:
        num = re.sub(r"[^\d.]","",val)
        return float(num)*100000 if num else None
    if "crore" in val or "cr" in val:
        num = re.sub(r"[^\d.]","",val)
        return float(num)*10000000 if num else None
    num = re.sub(r"[^\d.]","",val)
    return float(num) if num else None

def clean_kms(val):
    val = str(val).lower().replace(",","").strip()
    num = re.sub(r"[^\d]","",val)
    return int(num) if num else None

def clean_year(val):
    m = re.search(r"(19|20)\d{2}",str(val))
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
        if "name" in cl or "car" in cl:             rename_map[col] = "car_name"
        elif "price" in cl:                         rename_map[col] = "price_raw"
        elif "km" in cl or "odometer" in cl:        rename_map[col] = "kms_raw"
        elif "year" in cl:                          rename_map[col] = "year_raw"
        elif "fuel" in cl:                          rename_map[col] = "fuel_type"
        elif "transmission" in cl or "trans" in cl: rename_map[col] = "transmission"
        elif "owner" in cl:                         rename_map[col] = "owner"
        elif "city" in cl or "location" in cl:      rename_map[col] = "city"
        elif "seller" in cl:                        rename_map[col] = "seller_type"
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
    df = df[df["price_inr"].notna() & (df["price_inr"] > 10000)]
    df = df[df["year"].notna() & (df["year"] >= 1990)]
    print(f"Rows after cleaning: {before} -> {len(df)}")

    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Saved -> {PROCESSED_PATH}")
    print(df.head(3).to_string())

if __name__ == "__main__":
    main()
