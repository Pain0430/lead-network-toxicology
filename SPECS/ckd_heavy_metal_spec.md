# CKD + Heavy Metal ML Prediction Module Specification

## Module Name
`ckd_heavy_metal_prediction.py` - Chronic Kidney Disease Prediction with Heavy Metal Exposure

## Objective
Develop a machine learning model to predict CKD progression using heavy metal exposures (Pb, Cd, As, Hg, Mn) combined with clinical biomarkers, aligned with 2025-2026 research trends.

## Research Background (2025-2026)
- Nature Scientific Reports: SURD-enhanced ML for CKD prediction
- npj Digital Medicine: Deep learning for CKD progression in T2DM
- Klinrisk model: ML for CKD progression (AUC 0.80-0.87)
- Heavy metals as modifiable CKD risk factors

## Core Features

### 1. Data Preparation
- Load NHANES heavy metal data (blood Pb, Cd, As, Hg, Mn)
- Kidney function markers: eGFR, creatinine, BUN, albumin
- Combine with existing CKM framework

### 2. Heavy Metal Biomarkers
- Blood lead (BLLDLL)
- Blood cadmium (BDBCDC)
- Blood arsenic (URXUAS)
- Blood mercury (URXUHG)
- Blood manganese (BMXBMN)

### 3. Machine Learning Models
- XGBoost (primary)
- Random Forest
- LightGBM
- Ensemble voting

### 4. Feature Importance
- SHAP value analysis
- Permutation importance
- Metal-specific risk ranking

### 5. CKD Staging Analysis
- Stage 1-2 vs Stage 3-5 classification
- Risk stratification
- Dose-response relationships

## Input
- NHANES heavy metal + kidney function data
- Existing project data (nhanes_data/)

## Output
- `output/ckd_ml_model.pkl` - Trained model
- `output/ckd_shap_summary.png` - SHAP feature importance
- `output/ckd_metal_ranking.png` - Heavy metal risk ranking
- `output/ckd_predictions.csv` - Predictions
- `output/ckd_heavy_metal_results.json` - Full results

## Technical Stack
- pandas, numpy
- xgboost, lightgbm, sklearn
- shap
- matplotlib, seaborn

## Expected Outcomes
- CKD prediction model with heavy metal exposure
- Quantified contribution of each metal
- Integration with existing CKM framework
