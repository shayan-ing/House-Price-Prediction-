import os, pickle, calendar
import pandas as pd
from flask import Flask, render_template, request, jsonify
from pandas.api.types import CategoricalDtype

BASE = os.path.dirname(os.path.abspath(__file__))
MDIR = os.path.join(BASE, "model")

# Load the model, scaler, and metadata
with open(os.path.join(MDIR, "gbr.pkl"),    "rb") as f: MODEL  = pickle.load(f)
with open(os.path.join(MDIR, "scaler.pkl"), "rb") as f: SCALER = pickle.load(f)
with open(os.path.join(MDIR, "meta.pkl"),   "rb") as f: META   = pickle.load(f)

NUM2STR = ["MSSubClass","YearBuilt","YearRemodAdd","GarageYrBlt","MoSold","YrSold"]

def predict_price(user_input: dict) -> float:
    # Start with the default averages from our training data
    row = META["default_row"].copy()
    row.update({k: v for k, v in user_input.items() if v not in (None, "")})
    df = pd.DataFrame([row])

    # --- 1. Imputation (Safe fallbacks added) ---
    for c in ["MSZoning","Utilities","Exterior1st","Exterior2nd","MasVnrType",
              "Electrical","KitchenQual","Functional","SaleType"]:
        if c not in df.columns or pd.isna(df[c].iloc[0]):
            df[c] = META["modes"].get(c, "TA") # "TA" is a safe generic fallback

    for c in ["Alley","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1",
              "BsmtFinType2","FireplaceQu","PoolQC","Fence","MiscFeature",
              "GarageType","GarageFinish","GarageQual","GarageCond"]:
        if c not in df.columns:
            df[c] = "NA"
        else:
            df[c] = df[c].fillna("NA")
            
    for c in ["MasVnrArea","BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF",
              "BsmtFullBath","BsmtHalfBath","GarageYrBlt","GarageCars","GarageArea"]:
        if c not in df.columns:
            df[c] = 0.0
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
            
    if "LotFrontage" not in df.columns:
        df["LotFrontage"] = META["means"].get("LotFrontage", 70.0)
    else:
        df["LotFrontage"] = pd.to_numeric(df["LotFrontage"], errors="coerce").fillna(META["means"].get("LotFrontage", 70.0))

    # --- 2. Type transforms ---
    df["MoSold"] = df["MoSold"].apply(
        lambda x: calendar.month_abbr[int(float(x))] if not isinstance(x, str) else x
    )
    for c in NUM2STR:
        df[c] = df[c].astype(str)
        
    for c, cats in META["ordinal_maps"].items():
        if c in df.columns:
            df[c] = df[c].astype(CategoricalDtype(categories=cats, ordered=True)).cat.codes

    # --- 3. One-Hot Encoding ---
    obj_cols = list(df.select_dtypes(include="object").columns)
    for c in obj_cols:
        cats = META["train_categories"].get(c, None)
        df[c] = pd.Categorical(df[c], categories=cats)

    enc = pd.get_dummies(df, columns=obj_cols, prefix=obj_cols, drop_first=True)
    
    # Align columns exactly to how the model was trained
    enc = enc.reindex(columns=META["feature_columns"], fill_value=0).fillna(0).astype(float)

    X = SCALER.transform(enc)
    return float(MODEL.predict(X)[0])


app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        numeric = ["OverallQual","OverallCond","YearBuilt","YearRemodAdd",
                   "GrLivArea","TotalBsmtSF","1stFlrSF","2ndFlrSF","LotArea",
                   "GarageCars","GarageArea","FullBath","HalfBath",
                   "BedroomAbvGr","Fireplaces"]
        payload = {}
        for f in numeric:
            if f in data and data[f] not in (None, ""):
                payload[f] = float(data[f])
        for f in ["Neighborhood","MSZoning","BldgType","HouseStyle","KitchenQual"]:
            if f in data and data[f]:
                payload[f] = data[f]

        price = max(predict_price(payload), 10_000)
        return jsonify(success=True, price=round(price, 2))
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify(success=False, error=str(e)), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)