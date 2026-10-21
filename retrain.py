"""
Retrain GBR on your CURRENT environment (sklearn 1.9 / pandas 3.0 / numpy 2.4)
and produce gbr.pkl + scaler.pkl + meta.pkl  -> web app no longer needs CSVs.
"""
import os, sys, pickle, calendar, warnings, sklearn
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

BASE  = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(BASE, "ML_Model", "data_set", "train.csv")
TEST  = os.path.join(BASE, "ML_Model", "data_set", "test.csv")
OUT   = os.path.join(BASE, "model")
os.makedirs(OUT, exist_ok=True)

print(f"Python  : {sys.version.split()[0]}")
print(f"sklearn : {sklearn.__version__}")
print(f"pandas  : {pd.__version__}")
print(f"numpy   : {np.__version__}\n")

# ------------------------------------------------------------------
# 1. Load and combine
# ------------------------------------------------------------------
df_train = pd.read_csv(TRAIN)
df_test  = pd.read_csv(TEST)
n_train  = len(df_train)
df = pd.concat([df_train, df_test]).set_index("Id")

# ------------------------------------------------------------------
# 2. Missing-value imputation
# ------------------------------------------------------------------
df_mvi = df.copy()

for c in ["MSZoning","Utilities","Exterior1st","Exterior2nd","MasVnrType",
          "Electrical","KitchenQual","Functional","SaleType"]:
    m = df[c].mode()
    if len(m):
        df_mvi[c] = df_mvi[c].fillna(m.iloc[0]).astype(str)

for c in ["Alley","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1",
          "BsmtFinType2","FireplaceQu","PoolQC","Fence","MiscFeature",
          "GarageType","GarageFinish","GarageQual","GarageCond"]:
    df_mvi[c] = df_mvi[c].fillna("NA").astype(str)

for c in ["MasVnrArea","BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF",
          "BsmtFullBath","BsmtHalfBath","GarageYrBlt","GarageCars","GarageArea"]:
    df_mvi[c] = pd.to_numeric(df_mvi[c], errors="coerce").fillna(0.0)

df_mvi["LotFrontage"] = pd.to_numeric(df_mvi["LotFrontage"], errors="coerce") \
                          .fillna(float(df["LotFrontage"].mean()))

# ------------------------------------------------------------------
# 3. Type transformations
# ------------------------------------------------------------------
df_mvi["MoSold"] = df_mvi["MoSold"].apply(lambda x: calendar.month_abbr[int(float(x))])
for c in ["MSSubClass","YearBuilt","YearRemodAdd","GarageYrBlt","MoSold","YrSold"]:
    df_mvi[c] = df_mvi[c].astype(str)

ORDINAL = {
    "ExterQual":    ["Po","Fa","TA","Gd","Ex"],
    "ExterCond":    ["Po","Fa","TA","Gd","Ex"],
    "BsmtQual":     ["NA","Po","Fa","TA","Gd","Ex"],
    "BsmtCond":     ["NA","Po","Fa","TA","Gd","Ex"],
    "BsmtExposure": ["NA","No","Mn","Av","Gd"],
    "BsmtFinType1": ["NA","Unf","LwQ","Rec","BLQ","ALQ","GLQ"],
    "BsmtFinType2": ["NA","Unf","LwQ","Rec","BLQ","ALQ","GLQ"],
    "HeatingQC":    ["Po","Fa","TA","Gd","Ex"],
    "KitchenQual":  ["Po","Fa","TA","Gd","Ex"],
    "FireplaceQu":  ["NA","Po","Fa","TA","Gd","Ex"],
    "GarageQual":   ["NA","Po","Fa","TA","Gd","Ex"],
    "GarageCond":   ["NA","Po","Fa","TA","Gd","Ex"],
    "PoolQC":       ["NA","Fa","TA","Gd","Ex"],
    "Functional":   ["Sal","Sev","Maj2","Maj1","Mod","Min2","Min1","Typ"],
    "GarageFinish": ["NA","Unf","RFn","Fin"],
    "PavedDrive":   ["N","P","Y"],
    "Utilities":    ["ELO","NoSeWa","NoSewr","AllPub"],
}
for c, cats in ORDINAL.items():
    df_mvi[c] = df_mvi[c].astype(CategoricalDtype(categories=cats, ordered=True)).cat.codes

# ------------------------------------------------------------------
# 4. One-hot encode remaining object cols
# ------------------------------------------------------------------
obj_cols = df_mvi.select_dtypes(include="object").columns.tolist()
train_categories = {c: sorted(df_mvi[c].dropna().unique().tolist()) for c in obj_cols}

encoded = pd.get_dummies(df_mvi, columns=obj_cols, prefix=obj_cols, drop_first=True)

# ------------------------------------------------------------------
# 5. Split, scale, train
# ------------------------------------------------------------------
X_train = encoded.iloc[:n_train].drop(columns=["SalePrice"]).astype(float)
y_train = encoded.iloc[:n_train]["SalePrice"].astype(float)

feature_columns = X_train.columns.tolist()
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = GradientBoostingRegressor()
model.fit(X_train_scaled, y_train)

r2 = r2_score(y_train, model.predict(X_train_scaled))
print(f"Training R²  : {r2:.4f}")
print(f"Features     : {len(feature_columns)}")

# ------------------------------------------------------------------
# 6. Build default row for the web app
# ------------------------------------------------------------------
modes, means = {}, {}
for c in df.columns:
    if c == "SalePrice":
        continue
    if df[c].dtype == "object":
        m = df[c].mode()
        modes[c] = m.iloc[0] if len(m) else "NA"
    else:
        means[c] = float(pd.to_numeric(df[c], errors="coerce").mean())

default_row = {}
for c in df.columns:
    if c == "SalePrice":
        continue
    if c in modes:
        default_row[c] = modes[c]
    elif c in ["MSSubClass","YearBuilt","YearRemodAdd","GarageYrBlt","MoSold","YrSold"]:
        default_row[c] = df[c].mode().iloc[0]
    else:
        default_row[c] = means.get(c, 0.0)

# ------------------------------------------------------------------
# 7. Save artefacts
# ------------------------------------------------------------------
with open(os.path.join(OUT, "gbr.pkl"),    "wb") as f: pickle.dump(model,  f)
with open(os.path.join(OUT, "scaler.pkl"), "wb") as f: pickle.dump(scaler, f)

meta = {
    "feature_columns": feature_columns,
    "train_categories": train_categories,
    "ordinal_maps": ORDINAL,
    "modes": modes,
    "means": means,
    "default_row": default_row,
    "sklearn_version": sklearn.__version__,
    "pandas_version":  pd.__version__,
    "python_version":  sys.version.split()[0],
}
with open(os.path.join(OUT, "meta.pkl"), "wb") as f: pickle.dump(meta, f)

print("\n✅ Saved model/gbr.pkl")
print("✅ Saved model/scaler.pkl")
print("✅ Saved model/meta.pkl")