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
    num_cols, cat_cols, cat_options = [], [], {}
    try:
        preprocessor = pipeline.named_steps['preprocessor']
        for name, transformer, cols in preprocessor.transformers_:
            if name == 'remainder': continue
            if hasattr(cols, 'tolist'): cols = cols.tolist()
            else: cols = list(cols)
            valid_cols = [c for c in cols if c in selected_features]
            if name == 'num': num_cols.extend(valid_cols)
            elif name == 'cat':
                cat_cols.extend(valid_cols)
                try:
                    ohe = transformer.named_steps['onehot']
                    for col_name, categories in zip(cols, ohe.categories_):
                        if col_name in selected_features:
                            cat_options[col_name] = categories.tolist()
                except: pass
    except: pass
    for c in cat_cols:
        if c not in cat_options: cat_options[c] = ["Unknown"]
    return num_cols, cat_cols, cat_options

if pipeline:
    num_cols, cat_cols, cat_options = get_feature_info()
else:
    num_cols, cat_cols, cat_options = [], [], {}

# ===================================================================================
# 4. EXPLANATION LOGIC
# ===================================================================================
def generate_explanations(input_data):
    """
    Generates text explanations based on how inputs compare to baseline averages.
    """
    insights = []
    
    # 1. Size Logic
    sq_ft = input_data.get('GrLivArea', 0)
    if sq_ft > 2000:
        insights.append(f"Create expansive living space ({int(sq_ft)} sq ft) significantly boosts value.")
    elif sq_ft < 1000:
        insights.append(f"Smaller living area ({int(sq_ft)} sq ft) is a limiting factor on price.")

    # 2. Quality Logic
    quality = input_data.get('OverallQual', 5)
    if quality >= 8:
        insights.append("High build quality rating (8+) is a major value driver.")
    elif quality <= 4:
        insights.append("Below-average build quality reduces the estimate.")

    # 3. Garage Logic
    cars = input_data.get('GarageCars', 0)
    if cars >= 3:
        insights.append("Large garage capacity (3+ cars) is a premium feature.")
    
    # 4. Age Logic
    if 'HouseAge' in input_data:
        age = input_data['HouseAge']
        if age < 5:
            insights.append("Newer construction commands a premium market price.")
        elif age > 50 and input_data.get('OverallQual', 5) >= 7:
            insights.append("Vintage appeal: Older home with high quality retains value well.")

    # 5. Neighborhood Logic (Simplistic check)
    nbhd = input_data.get('Neighborhood', '')
    if nbhd in ['NoRidge', 'NridgHt', 'StoneBr']:
        insights.append(f"Located in high-demand neighborhood ({nbhd}).")
        
    if not insights:
        insights.append("This property aligns with standard market averages for this area.")
        
    return insights

# ===================================================================================
# 5. ROUTES
# ===================================================================================
@app.route('/', methods=['GET', 'POST'])
def index():
    if pipeline is None: return "Error: Model not loaded."
    
    prediction_text = None
    explanations = []

    if request.method == 'POST':
        try:
            input_data = {}
            # Capture numeric
            for col in num_cols:
                val = request.form.get(col)
                input_data[col] = float(val) if val else 0.0
            # Capture categorical
            for col in cat_cols:
                input_data[col] = request.form.get(col)

            input_df = pd.DataFrame([input_data])
            
            # Fill missing
            for col in selected_features:
                if col not in input_df.columns:
                    input_df[col] = 0.0 if col in num_cols else cat_options.get(col, [""])[0]

            input_df = input_df[selected_features]

            # Predict
            log_pred = pipeline.predict(input_df)[0]
            final_price = np.expm1(log_pred)
            prediction_text = f"${final_price:,.2f}"
            
            # Generate AI Explainability
            explanations = generate_explanations(input_data)
            
        except Exception as e:
            prediction_text = f"Error: {str(e)}"

    return render_template('index.html', 
                           num_cols=num_cols,
                           cat_cols=cat_cols,
                           cat_options=cat_options,
                           prediction=prediction_text,
                           explanations=explanations)

if __name__ == "__main__":
    app.run(debug=True)