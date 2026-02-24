#!/usr/bin/env python3
"""
铅网络毒理学 - 交互式可视化模块 (Plotly)
Lead Network Toxicology - Interactive Visualization Module

功能：
1. 交互式ROC/PR曲线
2. 交互式决策曲线
3. 交互式特征重要性
4. 动态风险预测器
5. SHAP交互式可视化

作者: Pain (重庆医科大学)
日期: 2026-02-25
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 工具函数
# ============================================================

def generate_lead_toxicology_data(n_samples=2000, random_state=42):
    """生成铅毒性研究模拟数据集"""
    np.random.seed(random_state)
    
    data = {
        'Age': np.random.normal(45, 15, n_samples).clip(18, 80),
        'Sex': np.random.binomial(1, 0.5, n_samples),
        'BMI': np.random.normal(25, 4, n_samples).clip(15, 45),
        'Blood_Lead_ug_dL': np.random.lognormal(2.5, 0.8, n_samples).clip(1, 80),
        'Urine_Lead_ug_L': np.random.lognormal(3.0, 0.9, n_samples).clip(5, 200),
        'Hair_Lead_ug_g': np.random.lognormal(1.5, 1.0, n_samples).clip(0.5, 50),
        'SOD_U_mL': np.random.normal(120, 25, n_samples),
        'GSH_umol_L': np.random.normal(8, 2, n_samples),
        'MDA_umol_L': np.random.lognormal(1.2, 0.5, n_samples),
        'CRP_mg_L': np.random.lognormal(1.0, 1.2, n_samples).clip(0.1, 50),
        'IL6_pg_mL': np.random.lognormal(2.0, 0.8, n_samples).clip(1, 100),
        'ALT_U_L': np.random.normal(25, 10, n_samples).clip(5, 200),
        'Creatinine_umol_L': np.random.normal(80, 20, n_samples).clip(30, 200),
        'DCA_umol_L': np.random.lognormal(2.0, 0.7, n_samples),
        'LCA_umol_L': np.random.lognormal(1.0, 0.6, n_samples),
        'Calprotectin_ug_g': np.random.lognormal(2.5, 1.0, n_samples).clip(10, 500),
    }
    
    lead_risk = (
        0.4 * (data['Blood_Lead_ug_dL'] > 15).astype(int) +
        0.3 * (data['Urine_Lead_ug_L'] > 50).astype(int) +
        0.3 * (data['MDA_umol_L'] > 4).astype(int) +
        0.2 * (data['CRP_mg_L'] > 5).astype(int) +
        0.2 * (data['Calprotectin_ug_g'] > 100).astype(int)
    )
    lead_risk += np.random.normal(0, 0.3, n_samples)
    data['Toxicity_Risk'] = (lead_risk > 1.5).astype(int)
    
    return pd.DataFrame(data)

def train_models(X, y, random_state=42):
    """训练模型"""
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_curve, auc, precision_recall_curve
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_state, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=random_state, class_weight='balanced'), True),
        'Random Forest': (RandomForestClassifier(n_estimators=100, max_depth=10, random_state=random_state, class_weight='balanced'), False),
        'Gradient Boosting': (GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=random_state), False)
    }
    
    results = {}
    for name, (model, use_scaled) in models.items():
        X_tr = X_train_scaled if use_scaled else X_train
        X_te = X_test_scaled if use_scaled else X_test
        model.fit(X_tr, y_train)
        y_prob = model.predict_proba(X_te)[:, 1]
        y_pred = model.predict(X_te)
        
        # 计算ROC和PR曲线（检查是否有足够的类别）
        try:
            fpr, tpr = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
        except ValueError:
            fpr, tpr = [0, 1], [0, 1]
            roc_auc = 0.5
        
        try:
            precision, recall = precision_recall_curve(y_test, y_prob)
            pr_auc = auc(recall, precision)
        except ValueError:
            precision, recall = [0, 1], [0, 1]
            pr_auc = 0.5
        
        results[name] = {
            'model': model,
            'y_test': y_test,
            'y_prob': y_prob,
            'y_pred': y_pred,
            'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc,
            'precision': precision, 'recall': recall, 'pr_auc': pr_auc,
            'use_scaled': use_scaled
        }
    
    return results, scaler, X_test, X_test_scaled, y_test

# ============================================================
# 交互式ROC曲线
# ============================================================

def plot_interactive_roc(results, output_dir='output'):
    """生成交互式ROC曲线"""
    fig = go.Figure()
    
    colors = {'Logistic Regression': '#2E86AB', 'Random Forest': '#A23B72', 'Gradient Boosting': '#F18F01'}
    dashes = {'Logistic Regression': 'dash', 'Random Forest': 'solid', 'Gradient Boosting': 'dot'}
    
    for name, res in results.items():
        fig.add_trace(go.Scatter(
            x=res['fpr'], y=res['tpr'],
            mode='lines',
            name=f"{name} (AUC = {res['roc_auc']:.3f})",
            line=dict(color=colors[name], width=3, dash=dashes[name]),
            hovertemplate=f'{name}<br>FPR: %{{x:.3f}}<br>TPR: %{{y:.3f}}<extra></extra>'
        ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random (AUC = 0.500)',
        line=dict(color='gray', width=2, dash='dash'),
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title=dict(text='Interactive ROC Curves - Lead Toxicity Prediction', font=dict(size=18)),
        xaxis_title='False Positive Rate (1 - Specificity)',
        yaxis_title='True Positive Rate (Sensitivity)',
        legend=dict(x=0.55, y=0.05, bgcolor='rgba(255,255,255,0.8)'),
        hovermode='closest',
        template='plotly_white',
        width=900, height=700
    )
    
    fig.write_html(f'{output_dir}/interactive_roc.html')
    fig.write_image(f'{output_dir}/interactive_roc.png', scale=2)
    print(f"✓ 交互式ROC曲线: {output_dir}/interactive_roc.html")
    return fig

# ============================================================
# 交互式PR曲线
# ============================================================

def plot_interactive_pr(results, output_dir='output'):
    """生成交互式PR曲线"""
    fig = go.Figure()
    
    colors = {'Logistic Regression': '#2E86AB', 'Random Forest': '#A23B72', 'Gradient Boosting': '#F18F01'}
    baseline = np.mean(results['Logistic Regression']['y_test'])
    
    for name, res in results.items():
        fig.add_trace(go.Scatter(
            x=res['recall'], y=res['precision'],
            mode='lines',
            name=f"{name} (AUC = {res['pr_auc']:.3f})",
            line=dict(color=colors[name], width=3),
            fill='tozeroy' if name == 'Random Forest' else None,
            fillcolor='rgba(162, 59, 114, 0.1)',
            hovertemplate=f'{name}<br>Recall: %{{x:.3f}}<br>Precision: %{{y:.3f}}<extra></extra>'
        ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[baseline, baseline],
        mode='lines',
        name=f'Baseline ({baseline:.3f})',
        line=dict(color='gray', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=dict(text='Interactive PR Curves - Lead Toxicity', font=dict(size=18)),
        xaxis_title='Recall (Sensitivity)',
        yaxis_title='Precision (PPV)',
        legend=dict(x=0.55, y=0.05, bgcolor='rgba(255,255,255,0.8)'),
        hovermode='closest',
        template='plotly_white',
        width=900, height=700
    )
    
    fig.write_html(f'{output_dir}/interactive_pr.html')
    fig.write_image(f'{output_dir}/interactive_pr.png', scale=2)
    print(f"✓ 交互式PR曲线: {output_dir}/interactive_pr.html")
    return fig

# ============================================================
# 交互式决策曲线
# ============================================================

def plot_interactive_dca(results, output_dir='output'):
    """生成交互式DCA曲线"""
    thresholds = np.linspace(0.01, 0.99, 50)
    y_test = results['Logistic Regression']['y_test']
    prevalence = np.mean(y_test)
    
    fig = go.Figure()
    colors = {'Logistic Regression': '#2E86AB', 'Random Forest': '#A23B72', 'Gradient Boosting': '#F18F01'}
    
    for name, res in results.items():
        net_benefits = []
        for t in thresholds:
            tp = np.sum((res['y_prob'] >= t) & (y_test == 1))
            fp = np.sum((res['y_prob'] >= t) & (y_test == 0))
            n = len(y_test)
            if t < 0.99:
                nb = (tp/n) - (fp/n) * (t/(1-t))
            else:
                nb = 0
            net_benefits.append(nb)
        
        fig.add_trace(go.Scatter(
            x=thresholds, y=net_benefits,
            mode='lines',
            name=name,
            line=dict(color=colors[name], width=3),
            hovertemplate=f'{name}<br>Threshold: %{{x:.2f}}<br>Net Benefit: %{{y:.3f}}<extra></extra>'
        ))
    
    # Treat All
    net_all = [prevalence - (1-prevalence)*t/(1-t) if t<0.99 else 0 for t in thresholds]
    fig.add_trace(go.Scatter(
        x=thresholds, y=net_all,
        mode='lines',
        name='Treat All',
        line=dict(color='black', width=2, dash='dash'),
        hoverinfo='skip'
    ))
    
    # Treat None
    fig.add_trace(go.Scatter(
        x=thresholds, y=[0]*len(thresholds),
        mode='lines',
        name='Treat None',
        line=dict(color='black', width=2, dash='solid'),
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title=dict(text='Interactive Decision Curve Analysis (DCA)', font=dict(size=18)),
        xaxis_title='Threshold Probability',
        yaxis_title='Net Benefit',
        legend=dict(x=0.65, y=0.95, bgcolor='rgba(255,255,255,0.8)'),
        hovermode='x unified',
        template='plotly_white',
        width=900, height=600
    )
    
    fig.write_html(f'{output_dir}/interactive_dca.html')
    fig.write_image(f'{output_dir}/interactive_dca.png', scale=2)
    print(f"✓ 交互式DCA曲线: {output_dir}/interactive_dca.html")
    return fig

# ============================================================
# 交互式特征重要性
# ============================================================

def plot_interactive_feature_importance(results, feature_names, output_dir='output'):
    """生成交互式特征重要性图"""
    fig = make_subplots(rows=1, cols=3, subplot_titles=list(results.keys()))
    
    colors = {'Logistic Regression': '#2E86AB', 'Random Forest': '#A23B72', 'Gradient Boosting': '#F18F01'}
    
    for idx, (name, res) in enumerate(results.items()):
        model = res['model']
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            continue
        
        top_n = 15
        indices = np.argsort(importance)[::-1][:top_n]
        
        fig.add_trace(go.Bar(
            y=[feature_names[i] for i in indices][::-1],
            x=importance[indices][::-1],
            orientation='h',
            name=name,
            marker_color=colors[name],
            hovertemplate='%{y}<br>Importance: %{x:.3f}<extra></extra>'
        ), row=1, col=idx+1)
    
    fig.update_layout(
        title=dict(text='Top 15 Feature Importance Comparison', font=dict(size=18)),
        height=600, width=1200,
        showlegend=False,
        template='plotly_white'
    )
    
    fig.write_html(f'{output_dir}/interactive_feature_importance.html')
    fig.write_image(f'{output_dir}/interactive_feature_importance.png', scale=2)
    print(f"✓ 交互式特征重要性: {output_dir}/interactive_feature_importance.html")
    return fig

# ============================================================
# 交互式风险热力图
# ============================================================

def plot_interactive_risk_heatmap(results, X_test, feature_names, output_dir='output'):
    """生成交互式风险预测热力图"""
    best_model = 'Random Forest'
    y_prob = results[best_model]['y_prob']
    
    # 选择重要特征
    important_features = ['Blood_Lead_ug_dL', 'Urine_Lead_ug_L', 'MDA_umol_L', 
                         'CRP_mg_L', 'Calprotectin_ug_g', 'DCA_umol_L']
    available_features = [f for f in important_features if f in X_test.columns]
    
    if len(available_features) < 4:
        available_features = feature_names[:6]
    
    # 标准化
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_plot = pd.DataFrame(scaler.fit_transform(X_test[available_features]), 
                         columns=available_features)
    X_plot['Risk_Score'] = y_prob
    
    # 按风险排序
    X_plot = X_plot.sort_values('Risk_Score')
    
    # 绘制热力图
    fig = go.Figure(data=go.Heatmap(
        z=X_plot[available_features].values,
        x=available_features,
        y=list(range(len(X_plot))),
        colorscale='RdYlGn_r',
        hovertemplate='Feature: %{x}<br>Sample: %{y}<br>Value: %{z:.2f}<extra></extra>',
        colorbar=dict(title='Standardized Value')
    ))
    
    fig.update_layout(
        title=dict(text=f'Risk Factor Heatmap - {best_model}', font=dict(size=18)),
        xaxis_title='Features',
        yaxis_title='Samples (sorted by risk)',
        height=700, width=900,
        template='plotly_white'
    )
    
    fig.write_html(f'{output_dir}/interactive_risk_heatmap.html')
    fig.write_image(f'{output_dir}/interactive_risk_heatmap.png', scale=2)
    print(f"✓ 交互式风险热力图: {output_dir}/interactive_risk_heatmap.html")
    return fig

# ============================================================
# 交互式校准曲线
# ============================================================

def plot_interactive_calibration(results, output_dir='output'):
    """生成交互式校准曲线"""
    fig = go.Figure()
    colors = {'Logistic Regression': '#2E86AB', 'Random Forest': '#A23B72', 'Gradient Boosting': '#F18F01'}
    
    for name, res in results.items():
        y_test = res['y_test']
        y_prob = res['y_prob']
        
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        bin_true = []
        for i in range(len(bins) - 1):
            mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
            if mask.sum() > 0:
                bin_true.append(y_test[mask].mean())
            else:
                bin_true.append(np.nan)
        
        fig.add_trace(go.Scatter(
            x=bin_centers, y=bin_true,
            mode='lines+markers',
            name=name,
            line=dict(color=colors[name], width=3),
            marker=dict(size=10),
            hovertemplate=f'{name}<br>Predicted: %{{x:.2f}}<br>Actual: %{{y:.2f}}<extra></extra>'
        ))
    
    # 完美校准线
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Perfect Calibration',
        line=dict(color='gray', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=dict(text='Interactive Calibration Curves', font=dict(size=18)),
        xaxis_title='Mean Predicted Probability',
        yaxis_title='Fraction of Positives',
        legend=dict(x=0.05, y=0.95, bgcolor='rgba(255,255,255,0.8)'),
        hovermode='closest',
        template='plotly_white',
        width=900, height=700
    )
    
    fig.write_html(f'{output_dir}/interactive_calibration.html')
    fig.write_image(f'{output_dir}/interactive_calibration.png', scale=2)
    print(f"✓ 交互式校准曲线: {output_dir}/interactive_calibration.html")
    return fig

# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print("铅网络毒理学 - 交互式可视化模块 (Plotly)")
    print("=" * 60)
    
    # 生成数据
    print("\n[1/5] 生成数据...")
    df = generate_lead_toxicology_data(n_samples=2000)
    
    # 准备特征
    feature_cols = [col for col in df.columns if col != 'Toxicity_Risk']
    X = df[feature_cols]
    y = df['Toxicity_Risk']
    
    # 训练模型
    print("[2/5] 训练模型...")
    results, scaler, X_test, X_test_scaled, y_test = train_models(X, y)
    
    # 创建输出目录
    import os
    output_dir = '/Users/pengsu/mycode/lead-network-toxicology/output'
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成交互式图表
    print("[3/5] 生成交互式ROC曲线...")
    plot_interactive_roc(results, output_dir)
    
    print("[4/5] 生成其他交互式图表...")
    plot_interactive_pr(results, output_dir)
    plot_interactive_dca(results, output_dir)
    plot_interactive_feature_importance(results, feature_cols, output_dir)
    plot_interactive_risk_heatmap(results, X_test, feature_cols, output_dir)
    plot_interactive_calibration(results, output_dir)
    
    print("\n" + "=" * 60)
    print(f"交互式可视化完成! 文件保存在: {output_dir}")
    print("=" * 60)
    
    return results

if __name__ == '__main__':
    main()
