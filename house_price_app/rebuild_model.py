import os
import json
import joblib
import pandas as pd
import numpy as np
from scipy.stats import skew

# Removed seaborn/matplotlib to avoid errors
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.base import BaseEstimator, TransformerMixin

# ===================================================================================
# 1. CUSTOM TRANSFORMER
# ===================================================================================
class SkewnessTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, skew_limit=0.75):
        self.skew_limit = skew_limit
        self.skewed_feats_ = None

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame): return self
        numeric_data = X.select_dtypes(include=[np.number])
        skewness = numeric_data.apply(lambda x: skew(x.dropna()))
        self.skewed_feats_ = skewness[abs(skewness) > self.skew_limit].index
        return self

    def transform(self, X):
        if self.skewed_feats_ is None: return X
        X_copy = X.copy()
        if isinstance(X_copy, pd.DataFrame):
            for feat in self.skewed_feats_:
                if feat in X_copy.columns:
                    X_copy[feat] = np.log1p(X_copy[feat])
        return X_copy

    def get_feature_names_out(self, input_features=None):
        return input_features

# ===================================================================================
# 2. DATA LOADING
# ===================================================================================
def load_and_engineer_data():
    if not os.path.exists('train.csv'):
        print("❌ CRITICAL ERROR: 'train.csv' not found.")
        print("   Make sure you downloaded the dataset and saved it in this folder.")
        return None, None
    
    train_df = pd.read_csv('train.csv')
    train_df.drop('Id', axis=1, inplace=True, errors='ignore')
    train_df = train_df[train_df.GrLivArea < 4500]
    
    y = np.log1p(train_df['SalePrice'])
    X = train_df.drop('SalePrice', axis=1)

    X['TotalSF'] = X['TotalBsmtSF'].fillna(0) + X['1stFlrSF'].fillna(0) + X['2ndFlrSF'].fillna(0)
    X['TotalBath'] = X['FullBath'].fillna(0) + (0.5 * X['HalfBath'].fillna(0)) + \
                     X['BsmtFullBath'].fillna(0) + (0.5 * X['BsmtHalfBath'].fillna(0))
    X['HouseAge'] = X['YrSold'] - X['YearBuilt']

    cols_to_drop = ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'FullBath', 'HalfBath', 
                    'BsmtFullBath', 'BsmtHalfBath', 'YearBuilt', 'YrSold']
    X.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    quality_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}
    ordinal_cols = ['ExterQual', 'ExterCond', 'KitchenQual', 'HeatingQC']
    for col in ordinal_cols:
        if col in X.columns:
            X[col] = X[col].map(quality_map).fillna(3)
    return X, y

# ===================================================================================
# 3. CONFIGURATION HANDLER (THE FIX)
# ===================================================================================
def get_or_create_config():
    # 1. Check for Params
    if os.path.exists("best_xgb_params.json"):
        with open("best_xgb_params.json", "r") as f:
            best_params = json.load(f)
    else:
        print("⚠️ Params file missing. Creating default XGBoost settings...")
        best_params = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 3,
            "subsample": 0.7,
            "colsample_bytree": 0.7
        }
        with open("best_xgb_params.json", "w") as f:
            json.dump(best_params, f)

    # 2. Check for Features
    if os.path.exists("selected_features.json"):
        with open("selected_features.json", "r") as f:
            selected_features = json.load(f)
    else:
        print("⚠️ Feature list missing. Creating default feature list...")
        # Standard high-impact features for housing data
        selected_features = [
            "OverallQual", "GrLivArea", "GarageCars", "TotalSF", "TotalBath",
            "HouseAge", "YearRemodAdd", "Fireplaces", "BsmtFinSF1", "LotArea",
            "CentralAir", "KitchenQual", "Neighborhood", "MSZoning"
        ]
        with open("selected_features.json", "w") as f:
            json.dump(selected_features, f)

    return best_params, selected_features

# ===================================================================================
# 4. BUILD PIPELINE
# ===================================================================================
def create_pipeline(numeric_features, categorical_features, model_params):
    num_trans = Pipeline(steps=[
        ('skew_fix', SkewnessTransformer(skew_limit=0.75)), 
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler()) 
    ])

    cat_trans = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) 
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_trans, numeric_features),
            ('cat', cat_trans, categorical_features)
        ],
        verbose_feature_names_out=False
    )

    model = XGBRegressor(**model_params, n_jobs=-1, random_state=42)
    return Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])

# ===================================================================================
# 5. MAIN EXECUTION
# ===================================================================================
def main():
    print("🚀 Starting Recovery Build...")
    X, y = load_and_engineer_data()
    if X is None: return

    # Get Config (or create defaults)
    best_params, selected_features = get_or_create_config()
    print(f"✅ Configuration ready: {len(selected_features)} features.")

    # Filter Data
    # Ensure all default features actually exist in the data before using them
    valid_features = [f for f in selected_features if f in X.columns]
    X_final = X[valid_features].copy()
    
    num_feats = X_final.select_dtypes(include=['int64', 'float64']).columns
    cat_feats = X_final.select_dtypes(include=['object', 'category']).columns

    print("🧠 Training Model...")
    student_pipeline = create_pipeline(num_feats, cat_feats, best_params)
    student_pipeline.fit(X_final, y)

    output_filename = 'student_pipeline.pkl'
    joblib.dump(student_pipeline, output_filename)
    print(f"✅ Model saved to: {output_filename}")
    print("👉 Now run: python app.py")

if __name__ == "__main__":
    main()