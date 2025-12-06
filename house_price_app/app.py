from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import json
import os
from scipy.stats import skew
from sklearn.base import BaseEstimator, TransformerMixin

# ===================================================================================
# 1. SETUP & CUSTOM CLASSES
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
# 🚑 THE FIX FOR RENDER DEPLOYMENT
# ===================================================================================
import __main__                              # <--- ADDED THIS
__main__.SkewnessTransformer = SkewnessTransformer  # <--- ADDED THIS
# ===================================================================================

app = Flask(__name__)

# ===================================================================================
# 2. LOAD ASSETS & BASELINE STATS
# ===================================================================================
pipeline = None
selected_features = []
baseline_averages = {} # To compare user input against

try:
    if not os.path.exists('student_pipeline.pkl'): raise FileNotFoundError("PKL file missing")
    
    pipeline = joblib.load('student_pipeline.pkl')
    with open('selected_features.json', 'r') as f:
        selected_features = json.load(f)
        
    # Calculate simple baselines from the trained model's perspective if possible,
    # or use hardcoded averages for the Ames dataset to generate explanations.
    # Here we define rough averages for the key numeric features to create the "AI Logic"
    baseline_averages = {
        'GrLivArea': 1500, 'TotalSF': 2500, 'OverallQual': 6, 'YearBuilt': 1970,
        'GarageCars': 2, 'TotalBath': 2, 'LotArea': 10000, 'HouseAge': 40
    }
    
except Exception as e:
    print(f"❌ LOAD ERROR: {e}")

# ===================================================================================
# 3. METADATA EXTRACTION
# ===================================================================================
def get_feature_info():
    if not pipeline: return [], [], {}
    num_cols, cat_cols,
