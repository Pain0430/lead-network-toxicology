#!/usr/bin/env python3
"""
CKD + Heavy Metal ML Prediction Module
========================================
Predicts Chronic Kidney Disease progression using heavy metal exposures.
Aligns with 2025-2026 research trends (Nature Scientific Reports, npj Digital Medicine)

Author: AI Assistant for Pain
Date: 2026-03-01
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                              VotingClassifier, ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, precision_recall_curve, auc, 
                             classification_report, confusion_matrix,
                             roc_curve)
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

# Configuration
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_nhanes_data():
    """Load NHANES heavy metal and kidney function data."""
    nhanes_dir = "nhanes_data"
    
    if os.path.exists(nhanes_dir):
        csv_files = [f for f in os.listdir(nhanes_dir) if f.endswith('.csv')]
        if csv_files:
            df = pd.read_csv(os.path.join(nhanes_dir, csv_files[0]))
            return df
    
    # Generate demo data for development
    print("Generating demo NHANES-style data...")
    np.random.seed(42)
    n = 3000
    
    data = {
        'age': np.random.randint(20, 80, n),
        'sex': np.random.randint(0, 2, n),
        'race': np.random.randint(0, 5, n),
        'bmi': np.random.normal(28, 5, n),
        'smoking': np.random.randint(0, 2, n),
        'diabetes': np.random.randint(0, 2, n),
        'hypertension': np.random.randint(0, 2, n),
    }
    
    # Heavy metals
    data['BLDLL'] = np.random.lognormal(mean=0.5, sigma=0.8, size=n)
    data['BDCDC'] = np.random.lognormal(mean=-1.5, sigma=0.9, size=n)
    data['URXUAS'] = np.random.lognormal(mean=2, sigma=1, size=n)
    data['URXUHG'] = np.random.lognormal(mean=0.5, sigma=1, size=n)
    data['BMXBMN'] = np.random.normal(10, 3, size=n)
    
    # Kidney function
    data['LBXSCRSI'] = np.random.normal(1.0, 0.3, n)
    data['LBXSAPSI'] = np.random.normal(15, 5, n)
    data['CKDEPID'] = np.random.normal(90, 25, n)
    data['LBXSALSI'] = np.random.normal(4.0, 0.5, n)
    data['LBXSUA'] = np.random.normal(5.5, 1.5, n)
    
    # Create CKD label
    data['CKD'] = (data['CKDEPID'] < 60).astype(int)
    for metal in ['BLDLL', 'BDCDC', 'URXUAS']:
        data['CKD'] = data['CKD'] | (data[metal] > np.percentile(data[metal], 75)).astype(int)
    
    return pd.DataFrame(data)


def engineer_features(df):
    """Create derived features for CKD prediction."""
    df = df.copy()
    
    # Metal exposure index
    df['total_metal_burden'] = (
        (df['BLDLL'] / df['BLDLL'].std()) + 
        (df['BDCDC'] / df['BDCDC'].std()) + 
        (df['URXUAS'] / df['URXUAS'].std())
    )
    
    # Metal quintiles
    for metal in ['BLDLL', 'BDCDC', 'URXUAS']:
        df[f'{metal}_quintile'] = pd.qcut(df[metal], 5, labels=False, duplicates='drop')
    
    # Kidney risk score
    df['kidney_risk'] = (
        (df['LBXSCRSI'] > 1.2).astype(int) +
        (df['LBXSAPSI'] > 20).astype(int) +
        (df['LBXSUA'] > 7).astype(int)
    )
    
    # Metal-kidney interactions
    df['Pb_creatinine'] = df['BLDLL'] * df['LBXSCRSI']
    df['Cd_egfr'] = df['BDCDC'] * df['CKDEPID']
    
    return df


def prepare_data(df):
    """Prepare features and target for modeling."""
    metal_features = ['BLDLL', 'BDCDC', 'URXUAS', 'URXUHG', 'BMXBMN']
    clinical_features = ['age', 'bmi', 'creatinine', 'egfr', 'albumin', 'uric_acid']
    clinical_map = {'creatinine': 'LBXSCRSI', 'egfr': 'CKDEPID', 'albumin': 'LBXSALSI', 'uric_acid': 'LBXSUA'}
    
    feature_cols = []
    for f in clinical_features:
        col = clinical_map.get(f, f.upper())
        if col in df.columns:
            feature_cols.append(col)
    
    for m in metal_features:
        if m in df.columns:
            feature_cols.append(m)
    
    derived = ['total_metal_burden', 'kidney_risk', 'Pb_creatinine', 'Cd_egfr']
    for d in derived:
        if d in df.columns:
            feature_cols.append(d)
    
    target = 'CKD'
    if target not in df.columns:
        target = 'CKD_from_egfr'
        df[target] = (df['CKDEPID'] < 60).astype(int)
    
    df_clean = df[feature_cols + [target]].dropna()
    X = df_clean[feature_cols]
    y = df_clean[target]
    
    return X, y, feature_cols


def train_models(X, y):
    """Train ensemble of ML models."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Models (using sklearn)
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1)
    gb_model = GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42)
    et_model = ExtraTreesClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    
    print("Training Random Forest...")
    rf_model.fit(X_train, y_train)
    
    print("Training Gradient Boosting...")
    gb_model.fit(X_train, y_train)
    
    print("Training Extra Trees...")
    et_model.fit(X_train, y_train)
    
    print("Training Logistic Regression...")
    lr_model.fit(X_train_scaled, y_train)
    
    # Ensemble
    ensemble = VotingClassifier(
        estimators=[('rf', rf_model), ('gb', gb_model), ('et', et_model)],
        voting='soft'
    )
    
    print("Training Ensemble...")
    ensemble.fit(X_train, y_train)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(ensemble, X, y, cv=cv, scoring='roc_auc')
    
    return {
        'rf': rf_model,
        'gb': gb_model,
        'et': et_model,
        'lr': lr_model,
        'ensemble': ensemble,
        'scaler': scaler,
        'X_test': X_test,
        'X_train': X_train,
        'y_test': y_test,
        'y_train': y_train,
        'cv_scores': cv_scores
    }


def evaluate_models(models, X, y, feature_names):
    """Evaluate model performance."""
    results = {}
    X_test = models['X_test']
    y_test = models['y_test']
    
    for name, model in [('RandomForest', models['rf']), 
                        ('GradientBoosting', models['gb']),
                        ('ExtraTrees', models['et']),
                        ('Ensemble', models['ensemble'])]:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        cm = confusion_matrix(y_test, y_pred)
        
        results[name] = {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'accuracy': (cm[0,0] + cm[1,1]) / cm.sum(),
            'sensitivity': cm[1,1] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0,
            'specificity': cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0,
            'confusion_matrix': cm.tolist()
        }
    
    results['cv_auc_mean'] = models['cv_scores'].mean()
    results['cv_auc_std'] = models['cv_scores'].std()
    
    return results


def analyze_feature_importance(models, feature_names):
    """Analyze feature importance."""
    importance_results = {}
    
    rf_importance = models['rf'].feature_importances_
    importance_results['random_forest'] = dict(zip(feature_names, rf_importance.tolist()))
    
    gb_importance = models['gb'].feature_importances_
    importance_results['gradient_boosting'] = dict(zip(feature_names, gb_importance.tolist()))
    
    # Permutation importance
    perm_importance = permutation_importance(models['rf'], models['X_test'], models['y_test'], n_repeats=10, random_state=42)
    importance_results['permutation'] = dict(zip(feature_names, perm_importance.importances_mean.tolist()))
    
    # Metal ranking
    metal_features = ['BLDLL', 'BDCDC', 'URXUAS', 'URXUHG', 'BMXBMN']
    metal_importance = {m: importance_results['random_forest'].get(m, 0) for m in metal_features if m in feature_names}
    importance_results['metal_ranking'] = sorted(metal_importance.items(), key=lambda x: x[1], reverse=True)
    
    return importance_results


def plot_results(results, importance_results, feature_names):
    """Generate visualization plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Model comparison
    ax1 = axes[0, 0]
    model_names = [k for k in results.keys() if isinstance(results[k], dict) and 'roc_auc' in results[k]]
    roc_scores = [results[k]['roc_auc'] for k in model_names]
    ax1.barh(model_names, roc_scores, color=['#3498db', '#2ecc71', '#9b59b6', '#e74c3c'][:len(model_names)])
    ax1.set_xlabel('ROC-AUC Score')
    ax1.set_title('Model Performance Comparison')
    ax1.set_xlim(0.5, 1.0)
    
    # Feature importance
    ax2 = axes[0, 1]
    if 'random_forest' in importance_results:
        sorted_imp = sorted(importance_results['random_forest'].items(), key=lambda x: x[1], reverse=True)[:10]
        features = [f[0] for f in sorted_imp]
        values = [f[1] for f in sorted_imp]
        ax2.barh(features[::-1], values[::-1], color='#3498db')
        ax2.set_xlabel('Importance Score')
        ax2.set_title('Top 10 Feature Importance (Random Forest)')
    
    # Metal ranking
    ax3 = axes[1, 0]
    if 'metal_ranking' in importance_results:
        metals = [m[0] for m in importance_results['metal_ranking']]
        vals = [m[1] for m in importance_results['metal_ranking']]
        labels = {'BLDLL': 'Lead (Pb)', 'BDCDC': 'Cadmium (Cd)', 'URXUAS': 'Arsenic (As)', 'URXUHG': 'Mercury (Hg)', 'BMXBMN': 'Manganese (Mn)'}
        label_list = [labels.get(m, m) for m in metals]
        ax3.barh(label_list, vals, color='#e74c3c')
        ax3.set_xlabel('Risk Contribution')
        ax3.set_title('Heavy Metal Risk Ranking')
    
    # CV results
    ax4 = axes[1, 1]
    if 'cv_auc_mean' in results:
        ax4.bar(['Cross-Validation'], [results['cv_auc_mean']], yerr=[results['cv_auc_std']], color='#2ecc71', capsize=5)
        ax4.set_ylabel('AUC Score')
        ax4.set_title('5-Fold Cross-Validation Performance')
        ax4.set_ylim(0.5, 1.0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ckd_heavy_metal_results.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plots saved to {OUTPUT_DIR}/ckd_heavy_metal_results.png")


def save_results(results, importance_results, models, feature_names):
    """Save all results."""
    with open(os.path.join(OUTPUT_DIR, 'ckd_heavy_metal_model.pkl'), 'wb') as f:
        pickle.dump({'ensemble': models['ensemble'], 'rf': models['rf'], 'scaler': models['scaler'], 'feature_names': feature_names}, f)
    
    output = {
        'model_performance': results,
        'feature_importance': importance_results,
        'metal_risk_ranking': importance_results.get('metal_ranking', []),
        'summary': {
            'best_model': max([(k, v.get('roc_auc', 0)) for k, v in results.items() if isinstance(v, dict)], key=lambda x: x[1])[0] if results else 'N/A',
            'cv_auc': results.get('cv_auc_mean', 0),
            'n_features': len(feature_names)
        }
    }
    
    with open(os.path.join(OUTPUT_DIR, 'ckd_heavy_metal_results.json'), 'w') as f:
        json.dump(output, f, indent=2)
    
    predictions = models['ensemble'].predict_proba(models['X_test'])[:, 1]
    pred_df = pd.DataFrame({'predicted_prob': predictions, 'actual': models['y_test'].values})
    pred_df.to_csv(os.path.join(OUTPUT_DIR, 'ckd_predictions.csv'), index=False)
    
    print(f"Results saved to {OUTPUT_DIR}/")


def main():
    print("=" * 60)
    print("CKD + Heavy Metal ML Prediction Pipeline")
    print("=" * 60)
    
    print("\n[1/5] Loading NHANES data...")
    df = load_nhanes_data()
    print(f"  Loaded {len(df)} samples")
    
    print("\n[2/5] Engineering features...")
    df = engineer_features(df)
    print(f"  Created {len(df.columns)} features")
    
    print("\n[3/5] Preparing modeling data...")
    X, y, feature_names = prepare_data(df)
    print(f"  Training samples: {len(X)}, Features: {len(feature_names)}")
    print(f"  CKD prevalence: {y.mean():.1%}")
    
    print("\n[4/5] Training ML models...")
    models = train_models(X, y)
    print("  Models trained successfully")
    
    print("\n[5/5] Evaluating and saving results...")
    results = evaluate_models(models, X, y, feature_names)
    importance_results = analyze_feature_importance(models, feature_names)
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for name, metrics in results.items():
        if isinstance(metrics, dict) and 'roc_auc' in metrics:
            print(f"\n{name}:")
            print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"  PR-AUC:  {metrics['pr_auc']:.4f}")
            print(f"  Sens:    {metrics['sensitivity']:.4f}")
            print(f"  Spec:    {metrics['specificity']:.4f}")
    
    if 'cv_auc_mean' in results:
        print(f"\nCross-Validation AUC: {results['cv_auc_mean']:.4f} ± {results['cv_auc_std']:.4f}")
    
    print("\nMetal Risk Ranking:")
    if 'metal_ranking' in importance_results:
        metal_labels = {'BLDLL': 'Lead (Pb)', 'BDCDC': 'Cadmium (Cd)', 'URXUAS': 'Arsenic (As)', 'URXUHG': 'Mercury (Hg)', 'BMXBMN': 'Manganese (Mn)'}
        for metal, importance in importance_results['metal_ranking']:
            print(f"  {metal_labels.get(metal, metal)}: {importance:.4f}")
    
    plot_results(results, importance_results, feature_names)
    save_results(results, importance_results, models, feature_names)
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)
    
    return results, importance_results


if __name__ == "__main__":
    results, importance = main()
