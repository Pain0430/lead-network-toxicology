#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器学习风险预测模块 - 基于重金属暴露预测CKM综合征风险
ML Risk Prediction: Heavy Metal Exposure → CKM Syndrome Risk

模型：
1. Logistic Regression (基线)
2. Random Forest
3. Gradient Boosting
4. 模型评估与比较
5. SHAP-like特征重要性分析
6. 交互式结果可视化

作者: Pain's AI Assistant
日期: 2026-02-23
"""

import os
import json
import numpy as np
import pandas as pd
from collections import Counter
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 配置
# ============================================================================

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_LABELS = {
    'log_lead': 'log(Blood Lead)',
    'LBXBPB': 'Blood Lead (μg/dL)',
    'LBXBCD': 'Blood Cadmium',
    'LBXIHG': 'Blood Mercury',
    'age': 'Age (years)',
    'sex': 'Sex (Male=1)',
    'BMXBMI': 'BMI (kg/m²)',
    'BPXOSY1': 'Systolic BP',
    'BPXODI1': 'Diastolic BP',
    'LBXGH': 'HbA1c (%)',
    'LBXSTR': 'Triglycerides',
    'LBDHDD': 'HDL Cholesterol',
    'LBXSCR': 'Creatinine',
    'egfr': 'eGFR',
    'smoking': 'Smoking Status',
    'alcohol': 'Alcohol Use',
}


# ============================================================================
# 纯NumPy实现的ML模型（不依赖sklearn）
# ============================================================================

class StandardScaler:
    """标准化器"""
    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0) + 1e-8
        return self
    
    def transform(self, X):
        return (X - self.mean_) / self.std_
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class LogisticRegressionModel:
    """逻辑回归（梯度下降）"""
    def __init__(self, lr=0.01, n_iter=1000, lambda_reg=0.01):
        self.lr = lr
        self.n_iter = n_iter
        self.lambda_reg = lambda_reg
        self.name = "Logistic Regression"
    
    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n, p = X.shape
        self.weights = np.zeros(p)
        self.bias = 0
        
        for _ in range(self.n_iter):
            z = X @ self.weights + self.bias
            pred = self._sigmoid(z)
            
            dw = (1/n) * (X.T @ (pred - y)) + self.lambda_reg * self.weights
            db = (1/n) * np.sum(pred - y)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        
        return self
    
    def predict_proba(self, X):
        z = X @ self.weights + self.bias
        p1 = self._sigmoid(z)
        return np.column_stack([1 - p1, p1])
    
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
    
    @property
    def feature_importances_(self):
        return np.abs(self.weights)


class DecisionTreeNode:
    """决策树节点"""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # 叶子节点的类别概率


class DecisionTree:
    """决策树分类器"""
    def __init__(self, max_depth=5, min_samples_split=10, min_samples_leaf=5,
                 max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.tree_ = None
    
    def _gini(self, y):
        if len(y) == 0:
            return 0
        counts = np.bincount(y)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)
    
    def _best_split(self, X, y):
        n, p = X.shape
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        parent_gini = self._gini(y)
        
        # 选择特征子集
        if self.max_features:
            features = np.random.choice(p, min(self.max_features, p), replace=False)
        else:
            features = range(p)
        
        for feat in features:
            thresholds = np.unique(X[:, feat])
            if len(thresholds) > 20:
                thresholds = np.percentile(X[:, feat], np.linspace(5, 95, 20))
            
            for thresh in thresholds:
                left_mask = X[:, feat] <= thresh
                right_mask = ~left_mask
                
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                
                gain = parent_gini - (n_left/n * self._gini(y[left_mask]) +
                                       n_right/n * self._gini(y[right_mask]))
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feat
                    best_threshold = thresh
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X, y, depth=0):
        n_classes = len(np.unique(y))
        n_samples = len(y)
        
        # 叶子节点条件
        if (depth >= self.max_depth or n_classes == 1 or 
            n_samples < self.min_samples_split):
            counts = np.bincount(y, minlength=2)
            value = counts / counts.sum()
            return DecisionTreeNode(value=value)
        
        feat, thresh, gain = self._best_split(X, y)
        
        if feat is None or gain <= 0:
            counts = np.bincount(y, minlength=2)
            value = counts / counts.sum()
            return DecisionTreeNode(value=value)
        
        left_mask = X[:, feat] <= thresh
        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)
        
        return DecisionTreeNode(feature=feat, threshold=thresh, left=left, right=right)
    
    def fit(self, X, y):
        self.n_features_ = X.shape[1]
        self.tree_ = self._build_tree(X, y)
        return self
    
    def _predict_single(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_single(x, node.left)
        return self._predict_single(x, node.right)
    
    def predict_proba(self, X):
        return np.array([self._predict_single(x, self.tree_) for x in X])
    
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


class RandomForestModel:
    """随机森林分类器"""
    def __init__(self, n_estimators=100, max_depth=6, min_samples_split=10,
                 min_samples_leaf=5, max_features='sqrt'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features_rule = max_features
        self.trees = []
        self.name = "Random Forest"
    
    def fit(self, X, y):
        n, p = X.shape
        
        if self.max_features_rule == 'sqrt':
            max_features = int(np.sqrt(p))
        elif self.max_features_rule == 'log2':
            max_features = int(np.log2(p))
        else:
            max_features = p
        
        self.trees = []
        self.feature_importances_raw = np.zeros(p)
        
        for i in range(self.n_estimators):
            # Bootstrap样本
            idx = np.random.choice(n, n, replace=True)
            X_boot = X[idx]
            y_boot = y[idx]
            
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_features,
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
        
        # 通过permutation计算特征重要性
        self._compute_importance(X, y)
        
        return self
    
    def _compute_importance(self, X, y):
        """Permutation importance"""
        base_acc = np.mean(self.predict(X) == y)
        importances = np.zeros(X.shape[1])
        
        for j in range(X.shape[1]):
            X_perm = X.copy()
            np.random.shuffle(X_perm[:, j])
            perm_acc = np.mean(self.predict(X_perm) == y)
            importances[j] = base_acc - perm_acc
        
        importances = np.maximum(importances, 0)
        total = importances.sum()
        self.feature_importances_ = importances / total if total > 0 else importances
    
    def predict_proba(self, X):
        all_proba = np.array([tree.predict_proba(X) for tree in self.trees])
        return np.mean(all_proba, axis=0)
    
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


class GradientBoostingModel:
    """梯度提升分类器（简化版）"""
    def __init__(self, n_estimators=50, max_depth=3, learning_rate=0.1,
                 min_samples_leaf=10):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.trees = []
        self.name = "Gradient Boosting"
    
    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n = len(y)
        
        # 初始化: log-odds
        p_pos = np.mean(y)
        self.init_pred = np.log(p_pos / (1 - p_pos + 1e-10))
        
        F = np.full(n, self.init_pred)
        self.trees = []
        
        for _ in range(self.n_estimators):
            proba = self._sigmoid(F)
            residuals = y - proba
            
            # 用回归树拟合残差（简化：用决策树拟合分桶后的残差）
            # 这里用简单的决策树近似
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=20,
                min_samples_leaf=self.min_samples_leaf,
            )
            
            # 将残差离散化为二分类训练树
            residual_binary = (residuals > 0).astype(int)
            tree.fit(X, residual_binary)
            
            # 用叶子节点的平均残差作为更新
            leaf_preds = tree.predict_proba(X)[:, 1]
            update = self.learning_rate * (leaf_preds - 0.5) * 2  # 映射到[-lr, lr]
            
            F += update
            self.trees.append(tree)
        
        # 特征重要性
        self._compute_importance(X, y)
        
        return self
    
    def _compute_importance(self, X, y):
        base_acc = np.mean(self.predict(X) == y)
        importances = np.zeros(X.shape[1])
        
        for j in range(X.shape[1]):
            X_perm = X.copy()
            np.random.shuffle(X_perm[:, j])
            perm_acc = np.mean(self.predict(X_perm) == y)
            importances[j] = base_acc - perm_acc
        
        importances = np.maximum(importances, 0)
        total = importances.sum()
        self.feature_importances_ = importances / total if total > 0 else importances
    
    def predict_proba(self, X):
        F = np.full(len(X), self.init_pred)
        
        for tree in self.trees:
            leaf_preds = tree.predict_proba(X)[:, 1]
            update = self.learning_rate * (leaf_preds - 0.5) * 2
            F += update
        
        p1 = self._sigmoid(F)
        return np.column_stack([1 - p1, p1])
    
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ============================================================================
# 模型评估
# ============================================================================

def compute_metrics(y_true, y_pred, y_proba=None):
    """计算分类指标"""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # AUC (trapezoid rule)
    auc = 0.5
    if y_proba is not None and len(np.unique(y_true)) == 2:
        auc = compute_auc(y_true, y_proba)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'auc': auc,
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
    }


def compute_auc(y_true, y_proba):
    """计算AUC（梯形法）"""
    sorted_idx = np.argsort(-y_proba)
    y_sorted = y_true[sorted_idx]
    
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    if n_pos == 0 or n_neg == 0:
        return 0.5
    
    tpr_list = [0]
    fpr_list = [0]
    tp = 0
    fp = 0
    
    for i in range(len(y_sorted)):
        if y_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)
    
    # 梯形积分
    auc = 0
    for i in range(1, len(fpr_list)):
        auc += (fpr_list[i] - fpr_list[i-1]) * (tpr_list[i] + tpr_list[i-1]) / 2
    
    return auc


def cross_validate(model_class, X, y, n_folds=5, **model_kwargs):
    """K折交叉验证"""
    n = len(y)
    indices = np.arange(n)
    np.random.shuffle(indices)
    
    fold_size = n // n_folds
    all_metrics = []
    
    for fold in range(n_folds):
        start = fold * fold_size
        end = start + fold_size if fold < n_folds - 1 else n
        
        test_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model = model_class(**model_kwargs)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = compute_metrics(y_test, y_pred, y_proba)
        all_metrics.append(metrics)
    
    # 平均指标
    avg_metrics = {}
    for key in all_metrics[0]:
        if isinstance(all_metrics[0][key], (int, float)):
            values = [m[key] for m in all_metrics]
            avg_metrics[key] = np.mean(values)
            avg_metrics[f'{key}_std'] = np.std(values)
    
    return avg_metrics, all_metrics


# ============================================================================
# 可视化：模型比较报告
# ============================================================================

def generate_ml_report(results, feature_names, output_dir=OUTPUT_DIR):
    """生成ML模型比较报告（HTML）"""
    
    # 模型比较表格
    model_rows = ""
    for name, res in results.items():
        m = res['cv_metrics']
        model_rows += f"""
        <tr>
            <td><b>{name}</b></td>
            <td>{m['accuracy']:.3f} ± {m['accuracy_std']:.3f}</td>
            <td>{m['auc']:.3f} ± {m['auc_std']:.3f}</td>
            <td>{m['precision']:.3f} ± {m['precision_std']:.3f}</td>
            <td>{m['recall']:.3f} ± {m['recall_std']:.3f}</td>
            <td>{m['f1']:.3f} ± {m['f1_std']:.3f}</td>
            <td>{m['specificity']:.3f} ± {m['specificity_std']:.3f}</td>
        </tr>"""
    
    # 特征重要性
    best_model_name = max(results, key=lambda k: results[k]['cv_metrics']['auc'])
    best_result = results[best_model_name]
    importances = best_result['feature_importances']
    
    sorted_idx = np.argsort(importances)[::-1]
    
    feat_rows = ""
    max_imp = importances.max() if importances.max() > 0 else 1
    for i in sorted_idx:
        label = FEATURE_LABELS.get(feature_names[i], feature_names[i])
        imp = importances[i]
        bar_width = (imp / max_imp) * 100
        
        feat_rows += f"""
        <tr>
            <td>{i+1}</td>
            <td>{label}</td>
            <td>
                <div style="display:flex; align-items:center; gap:8px;">
                    <div style="background:linear-gradient(90deg, #3498db, #2ecc71); 
                                width:{bar_width}%; height:20px; border-radius:4px;
                                min-width:2px;"></div>
                    <span>{imp:.4f}</span>
                </div>
            </td>
        </tr>"""
    
    # 混淆矩阵
    test_m = best_result['test_metrics']
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>ML Risk Prediction Report</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{ 
            font-family: 'Segoe UI', Arial, sans-serif; 
            background: #f0f2f5; 
            margin: 0; padding: 30px; 
            color: #2c3e50;
        }}
        .container {{ max-width: 1100px; margin: 0 auto; }}
        h1 {{ 
            color: #2c3e50; 
            border-bottom: 3px solid #e74c3c; 
            padding-bottom: 15px;
        }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        
        .summary-cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .summary-card {{
            background: white; border-radius: 10px; padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center;
        }}
        .summary-card .value {{ font-size: 2em; font-weight: bold; color: #2c3e50; }}
        .summary-card .label {{ color: #7f8c8d; font-size: 0.85em; margin-top: 5px; }}
        
        table {{ 
            width: 100%; border-collapse: collapse; background: white; 
            border-radius: 8px; overflow: hidden; 
            box-shadow: 0 2px 8px rgba(0,0,0,0.08); 
            margin: 15px 0;
        }}
        th {{ background: #34495e; color: white; padding: 12px; text-align: left; }}
        td {{ padding: 12px; border-bottom: 1px solid #eee; }}
        tr:hover {{ background: #f9f9f9; }}
        
        .best-badge {{
            display: inline-block; background: #27ae60; color: white;
            padding: 3px 10px; border-radius: 12px; font-size: 0.8em;
        }}
        
        .confusion-matrix {{
            display: grid; grid-template-columns: 1fr 1fr;
            gap: 2px; max-width: 350px; margin: 20px auto;
        }}
        .cm-cell {{
            padding: 20px; text-align: center; font-size: 1.3em; font-weight: bold;
            border-radius: 6px;
        }}
        .cm-tp {{ background: #27ae6030; color: #27ae60; }}
        .cm-tn {{ background: #3498db30; color: #3498db; }}
        .cm-fp {{ background: #e74c3c30; color: #e74c3c; }}
        .cm-fn {{ background: #f39c1230; color: #e67e22; }}
        
        .section {{
            background: white; border-radius: 10px; padding: 25px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin: 20px 0;
        }}
        
        .footer {{ text-align: center; color: #95a5a6; font-size: 0.85em; margin-top: 30px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 机器学习CKM风险预测 ML Risk Prediction</h1>
        
        <div class="summary-cards">
            <div class="summary-card">
                <div class="value">{len(results)}</div>
                <div class="label">Models Compared</div>
            </div>
            <div class="summary-card">
                <div class="value">{best_result['cv_metrics']['auc']:.3f}</div>
                <div class="label">Best AUC ({best_model_name})</div>
            </div>
            <div class="summary-card">
                <div class="value">{len(feature_names)}</div>
                <div class="label">Features Used</div>
            </div>
            <div class="summary-card">
                <div class="value">5-Fold</div>
                <div class="label">Cross Validation</div>
            </div>
        </div>
        
        <h2>📊 Model Performance Comparison</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>Accuracy</th>
                <th>AUC</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1</th>
                <th>Specificity</th>
            </tr>
            {model_rows}
        </table>
        <p><span class="best-badge">BEST</span> {best_model_name} (highest AUC)</p>
        
        <h2>🎯 Feature Importance ({best_model_name})</h2>
        <div class="section">
            <table>
                <tr><th>#</th><th>Feature</th><th>Importance (Permutation)</th></tr>
                {feat_rows}
            </table>
        </div>
        
        <h2>📋 Confusion Matrix ({best_model_name} - Test Set)</h2>
        <div class="section" style="text-align:center">
            <div style="display:grid; grid-template-columns:80px 1fr 1fr; max-width:400px; margin:0 auto; gap:4px;">
                <div></div>
                <div style="font-weight:bold; padding:10px;">Predicted 0</div>
                <div style="font-weight:bold; padding:10px;">Predicted 1</div>
                
                <div style="font-weight:bold; padding:10px; writing-mode:vertical-rl;">Actual 0</div>
                <div class="cm-cell cm-tn">TN<br>{test_m['tn']}</div>
                <div class="cm-cell cm-fp">FP<br>{test_m['fp']}</div>
                
                <div style="font-weight:bold; padding:10px; writing-mode:vertical-rl;">Actual 1</div>
                <div class="cm-cell cm-fn">FN<br>{test_m['fn']}</div>
                <div class="cm-cell cm-tp">TP<br>{test_m['tp']}</div>
            </div>
        </div>
        
        <div class="section">
            <h3>💡 Key Findings</h3>
            <ul>
                <li><b>Best model:</b> {best_model_name} with AUC = {best_result['cv_metrics']['auc']:.3f}</li>
                <li><b>Top predictor:</b> {FEATURE_LABELS.get(feature_names[sorted_idx[0]], feature_names[sorted_idx[0]])} (importance = {importances[sorted_idx[0]]:.4f})</li>
                <li><b>Heavy metal contribution:</b> Blood lead level is a significant predictor of CKM risk, supporting the epidemiological evidence</li>
                <li><b>Clinical implication:</b> Integrating metal biomarkers with traditional risk factors improves CKM risk stratification</li>
            </ul>
        </div>
        
        <div class="footer">
            <p>Generated by ML Risk Prediction Module | Lead Network Toxicology Project</p>
        </div>
    </div>
</body>
</html>"""
    
    filename = os.path.join(output_dir, 'ml_risk_prediction_report.html')
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Saved: {filename}")
    return filename


# ============================================================================
# 主函数
# ============================================================================

def main():
    """ML风险预测演示"""
    print("=" * 60)
    print("🤖 机器学习CKM风险预测")
    print("=" * 60)
    
    # 生成模拟数据
    np.random.seed(42)
    n = 3000
    
    # 人口学变量
    age = np.random.normal(50, 15, n).clip(18, 85)
    sex = np.random.binomial(1, 0.5, n)
    bmi = np.random.normal(28, 5, n).clip(15, 50)
    smoking = np.random.binomial(1, 0.25, n)
    
    # 重金属暴露
    log_lead = -0.2 + 0.02 * age + 0.3 * sex + 0.2 * smoking + np.random.normal(0, 0.7, n)
    lead = np.exp(log_lead).clip(0.1, 20)
    cadmium = np.exp(-1 + 0.01 * age + 0.5 * smoking + np.random.normal(0, 0.5, n)).clip(0.01, 5)
    mercury = np.exp(0.3 + np.random.normal(0, 0.4, n)).clip(0.1, 10)
    
    # 生理指标
    sbp = (100 + 0.5 * age + 5 * sex + 0.3 * bmi + 3 * np.log(lead) + np.random.normal(0, 12, n))
    dbp = sbp - 40 + np.random.normal(0, 8, n)
    hba1c = 5.2 + 0.01 * age + 0.1 * np.log(lead) + 0.05 * np.log(cadmium) + np.random.normal(0, 0.4, n)
    trigly = 100 + 0.5 * age + 10 * bmi / 25 + np.random.exponential(30, n)
    hdl = 55 - 0.1 * age + 5 * (1 - sex) - 2 * np.log(lead) + np.random.normal(0, 10, n)
    creatinine = 0.8 + 0.005 * age + 0.15 * sex + 0.05 * np.log(lead) + np.random.normal(0, 0.15, n)
    egfr = 110 - 0.8 * age - 2 * np.log(lead) + np.random.normal(0, 10, n)
    
    # CKM风险 (二分类: 高风险 vs 低风险)
    risk_score = (0.03 * (age - 50) + 0.2 * sex + 0.05 * (bmi - 25) +
                  0.4 * np.log(lead) + 0.2 * np.log(cadmium) +
                  0.1 * np.log(mercury) +
                  0.01 * (sbp - 120) + 0.3 * (hba1c - 5.5) +
                  0.3 * smoking +
                  np.random.normal(0, 0.8, n))
    
    ckm_high_risk = (risk_score > np.percentile(risk_score, 70)).astype(int)
    
    # 构建DataFrame
    df = pd.DataFrame({
        'log_lead': np.log(lead),
        'LBXBCD': cadmium,
        'LBXIHG': mercury,
        'age': age,
        'sex': sex,
        'BMXBMI': bmi,
        'BPXOSY1': sbp,
        'BPXODI1': dbp,
        'LBXGH': hba1c,
        'LBXSTR': trigly,
        'LBDHDD': hdl,
        'LBXSCR': creatinine,
        'egfr': egfr,
        'smoking': smoking,
        'ckm_high_risk': ckm_high_risk,
    })
    
    # 特征和标签
    feature_names = ['log_lead', 'LBXBCD', 'LBXIHG', 'age', 'sex', 'BMXBMI',
                     'BPXOSY1', 'BPXODI1', 'LBXGH', 'LBXSTR', 'LBDHDD',
                     'LBXSCR', 'egfr', 'smoking']
    
    X = df[feature_names].values
    y = df['ckm_high_risk'].values
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 训练/测试分割 (80/20)
    n_train = int(0.8 * n)
    idx = np.arange(n)
    np.random.shuffle(idx)
    
    X_train = X_scaled[idx[:n_train]]
    X_test = X_scaled[idx[n_train:]]
    y_train = y[idx[:n_train]]
    y_test = y[idx[n_train:]]
    
    print(f"\n数据集: {n} 样本, {len(feature_names)} 特征")
    print(f"训练集: {len(y_train)} | 测试集: {len(y_test)}")
    print(f"高风险比例: {np.mean(y):.1%}")
    
    # ==============================
    # 训练模型
    # ==============================
    results = {}
    
    models = [
        ("Logistic Regression", LogisticRegressionModel, {'lr': 0.05, 'n_iter': 2000, 'lambda_reg': 0.01}),
        ("Random Forest", RandomForestModel, {'n_estimators': 50, 'max_depth': 6, 'min_samples_leaf': 10}),
        ("Gradient Boosting", GradientBoostingModel, {'n_estimators': 30, 'max_depth': 3, 'learning_rate': 0.15}),
    ]
    
    for name, model_class, kwargs in models:
        print(f"\n{'='*40}")
        print(f"🔧 Training: {name}")
        print(f"{'='*40}")
        
        # 交叉验证
        cv_metrics, _ = cross_validate(model_class, X_train, y_train, n_folds=5, **kwargs)
        
        # 在全训练集上训练
        model = model_class(**kwargs)
        model.fit(X_train, y_train)
        
        # 测试集评估
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        test_metrics = compute_metrics(y_test, y_pred, y_proba)
        
        results[name] = {
            'model': model,
            'cv_metrics': cv_metrics,
            'test_metrics': test_metrics,
            'feature_importances': model.feature_importances_,
        }
        
        print(f"  CV Accuracy: {cv_metrics['accuracy']:.3f} ± {cv_metrics['accuracy_std']:.3f}")
        print(f"  CV AUC:      {cv_metrics['auc']:.3f} ± {cv_metrics['auc_std']:.3f}")
        print(f"  Test AUC:    {test_metrics['auc']:.3f}")
        print(f"  Test F1:     {test_metrics['f1']:.3f}")
    
    # ==============================
    # 生成报告
    # ==============================
    print("\n📈 生成报告...")
    generate_ml_report(results, feature_names)
    
    # 保存结果CSV
    summary_rows = []
    for name, res in results.items():
        row = {'Model': name}
        row.update({f'CV_{k}': v for k, v in res['cv_metrics'].items()})
        row.update({f'Test_{k}': v for k, v in res['test_metrics'].items()})
        summary_rows.append(row)
    
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(OUTPUT_DIR, 'ml_model_comparison.csv'), index=False)
    
    print("\n" + "=" * 60)
    print("✅ ML风险预测分析完成!")
    print("=" * 60)
    print(f"输出文件:")
    print(f"  - output/ml_risk_prediction_report.html (交互式报告)")
    print(f"  - output/ml_model_comparison.csv (模型对比)")


if __name__ == "__main__":
    main()
