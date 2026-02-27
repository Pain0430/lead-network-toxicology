#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
铅网络毒理学 - 增强综合分析模块
Lead Network Toxicology - Enhanced Comprehensive Analysis

功能：
1. 多模型对比分析 (Logistic, RF, XGBoost, SVM)
2. 增强校准曲线 (Brier Score)
3. 特征重要性综合视图
4. 决策阈值优化分析
5. 概率校准分析
6. 敏感性-特异性权衡
7. 综合评估报告

作者: Pain AI Assistant
日期: 2026-02-27
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, brier_score_loss,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, balanced_accuracy_score
)
from sklearn.inspection import permutation_importance
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================
# 配置
# ============================================================
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 配色方案
COLORS = {
    'primary': '#2C3E50',
    'secondary': '#E74C3C', 
    'tertiary': '#3498DB',
    'quaternary': '#27AE60',
    'accent': '#F39C12',
    'purple': '#9B59B6',
    'cyan': '#1ABC9C',
    'pink': '#E91E63',
    'orange': '#FF5722',
    'teal': '#009688',
    'light_gray': '#ECF0F1',
    'dark_gray': '#7F8C8D'
}

OUTPUT_DIR = '/Users/pengsu/mycode/lead-network-toxicology/output'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_demo_data(n_samples=3000, random_state=42):
    """生成铅毒性综合模拟数据集"""
    np.random.seed(random_state)
    
    # 铅暴露核心指标
    lead_exposure = np.random.normal(0, 1, n_samples)
    
    # 潜在风险因素
    oxidative_stress = 0.7 * lead_exposure + np.random.normal(0, 0.6, n_samples)
    inflammation = 0.6 * oxidative_stress + np.random.normal(0, 0.5, n_samples)
    gut_dysbiosis = 0.5 * inflammation + np.random.normal(0, 0.5, n_samples)
    metabolic_disorder = 0.4 * inflammation + 0.3 * oxidative_stress + np.random.normal(0, 0.5, n_samples)
    
    data = {
        # 铅暴露指标
        'Blood_Lead': np.exp(2.5 + 0.8 * lead_exposure).clip(1, 80),
        'Urine_Lead': np.exp(1.8 + 0.9 * lead_exposure + np.random.normal(0, 0.3, n_samples)).clip(0.5, 25),
        'Hair_Lead': np.exp(3.0 + 0.85 * lead_exposure + np.random.normal(0, 0.25, n_samples)).clip(1, 50),
        
        # 氧化应激
        'SOD': (120 - 20 * oxidative_stress + np.random.normal(0, 10, n_samples)).clip(50, 250),
        'GSH': (8 - 1.5 * oxidative_stress + np.random.normal(0, 0.8, n_samples)).clip(3, 15),
        'MDA': np.exp(1.2 + 0.6 * oxidative_stress + np.random.normal(0, 0.3, n_samples)).clip(0.5, 10),
        
        # 炎症标志物
        'CRP': np.exp(1.0 + 0.8 * inflammation + np.random.normal(0, 0.5, n_samples)).clip(0.1, 50),
        'IL6': np.exp(1.5 + 0.7 * inflammation + np.random.normal(0, 0.4, n_samples)).clip(0.5, 30),
        'TNF_alpha': np.exp(2.0 + 0.65 * inflammation + np.random.normal(0, 0.35, n_samples)).clip(1, 25),
        
        # 肠道屏障
        'Calprotectin': np.exp(3.5 + 0.6 * gut_dysbiosis + np.random.normal(0, 0.4, n_samples)).clip(10, 500),
        'Zonulin': np.exp(2.8 + 0.55 * gut_dysbiosis + np.random.normal(0, 0.35, n_samples)).clip(20, 200),
        'D_Lactate': np.exp(0.8 + 0.5 * gut_dysbiosis + np.random.normal(0, 0.4, n_samples)).clip(0.5, 8),
        
        # 代谢指标
        'HbA1c': (5.5 + 0.3 * metabolic_disorder + np.random.normal(0, 0.5, n_samples)).clip(4.5, 12),
        'Triglycerides': np.exp(4.5 + 0.4 * metabolic_disorder + np.random.normal(0, 0.4, n_samples)).clip(30, 400),
        'HDL': (55 - 8 * metabolic_disorder + np.random.normal(0, 5, n_samples)).clip(20, 90),
        
        # 协变量
        'Age': np.random.normal(45, 15, n_samples).clip(18, 80),
        'BMI': np.random.normal(24, 4, n_samples).clip(16, 40),
        'Sex': np.random.binomial(1, 0.5, n_samples),
        'Smoking': np.random.binomial(1, 0.25, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # 计算综合风险评分
    risk_score = (
        0.20 * (np.log(df['Blood_Lead']) - np.log(10)) / 2 +
        0.15 * (df['MDA'] - 2) / 8 +
        0.10 * (np.log(df['CRP']) - np.log(1)) / 3 +
        0.10 * (np.log(df['IL6']) - np.log(2)) / 2 +
        0.10 * (np.log(df['Calprotectin']) - np.log(20)) / 3 +
        0.10 * (df['HbA1c'] - 5.5) / 4 +
        0.10 * (np.log(df['Triglycerides']) - np.log(50)) / 2 +
        0.05 * (df['Age'] - 30) / 50 +
        0.05 * (df['BMI'] - 20) / 15 +
        0.05 * df['Smoking']
    )
    
    # 生成二分类结局
    prob = 1 / (1 + np.exp(-2.5 * (risk_score - 0.3 + np.random.normal(0, 0.15, n_samples))))
    df['Outcome'] = (np.random.random(n_samples) < prob).astype(int)
    
    return df


def train_models(X_train, X_test, y_train, y_test):
    """训练多个机器学习模型"""
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
        'SVM': SVC(probability=True, random_state=42, kernel='rbf'),
    }
    
    trained_models = {}
    for name, model in models.items():
        print(f"训练模型: {name}")
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    return trained_models


def evaluate_models(trained_models, X_test, y_test):
    """综合评估所有模型"""
    results = []
    
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # 计算各种指标
        metrics = {
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'AUC': roc_auc_score(y_test, y_prob),
            'Brier Score': brier_score_loss(y_test, y_prob),
            'MCC': matthews_corrcoef(y_test, y_pred),
            'Balanced Accuracy': balanced_accuracy_score(y_test, y_pred),
        }
        
        results.append(metrics)
    
    return pd.DataFrame(results)


def plot_model_comparison(results, save_path=None):
    """绘制模型对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    models = results['Model'].tolist()
    x = np.arange(len(models))
    width = 0.6
    
    # 1. 主要指标对比
    ax1 = axes[0, 0]
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = [COLORS['primary'], COLORS['tertiary'], COLORS['quaternary'], COLORS['accent']]
    
    for i, metric in enumerate(metrics):
        values = results[metric].values
        bars = ax1.bar(x + i * width/4 - 0.3, values, width/4, label=metric, color=colors[i], alpha=0.85)
    
    ax1.set_xlabel('模型', fontsize=12)
    ax1.set_ylabel('分数', fontsize=12)
    ax1.set_title('模型性能指标对比', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace(' ', '\n') for m in models], fontsize=9)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.set_ylim(0, 1.1)
    ax1.axhline(y=0.8, color=COLORS['secondary'], linestyle='--', alpha=0.5, label='0.8阈值')
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. AUC对比
    ax2 = axes[0, 1]
    auc_colors = [COLORS['primary'], COLORS['tertiary'], COLORS['quaternary'], COLORS['accent']]
    bars = ax2.bar(x, results['AUC'], width, color=auc_colors, alpha=0.85, edgecolor='white', linewidth=2)
    
    for bar, val in zip(bars, results['AUC']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.set_xlabel('模型', fontsize=12)
    ax2.set_ylabel('AUC', fontsize=12)
    ax2.set_title('AUC分数对比', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.replace(' ', '\n') for m in models], fontsize=9)
    ax2.set_ylim(0.7, 1.0)
    ax2.axhline(y=0.9, color=COLORS['secondary'], linestyle='--', alpha=0.7, linewidth=2)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Brier Score (越低越好)
    ax3 = axes[1, 0]
    brier_colors = [COLORS['quaternary'] if v < 0.15 else COLORS['secondary'] for v in results['Brier Score']]
    bars = ax3.bar(x, results['Brier Score'], width, color=brier_colors, alpha=0.85, edgecolor='white', linewidth=2)
    
    for bar, val in zip(bars, results['Brier Score']):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax3.set_xlabel('模型', fontsize=12)
    ax3.set_ylabel('Brier Score', fontsize=12)
    ax3.set_title('概率校准质量 (Brier Score, 越低越好)', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([m.replace(' ', '\n') for m in models], fontsize=9)
    ax3.axhline(y=0.15, color=COLORS['secondary'], linestyle='--', alpha=0.7, linewidth=2)
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. 综合雷达图
    ax4 = axes[1, 1]
    
    # 创建简化的对比表格
    table_data = results[['Model', 'AUC', 'F1-Score', 'Brier Score', 'Balanced Accuracy']].copy()
    table_data = table_data.round(3)
    
    ax4.axis('off')
    table = ax4.table(
        cellText=table_data.values,
        colLabels=table_data.columns,
        cellLoc='center',
        loc='center',
        colColours=[COLORS['primary']] * len(table_data.columns)
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # 设置表头颜色
    for i in range(len(table_data.columns)):
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    ax4.set_title('综合评估指标表', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"图表已保存: {save_path}")
    
    plt.close()
    return fig


def plot_roc_curves(trained_models, X_test, y_test, save_path=None):
    """绘制ROC曲线对比"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = [COLORS['primary'], COLORS['tertiary'], COLORS['quaternary'], COLORS['accent']]
    
    for i, (name, model) in enumerate(trained_models.items()):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color=colors[i], lw=2.5, 
                label=f'{name} (AUC = {roc_auc:.3f})')
    
    # 对角线
    ax.plot([0, 1], [0, 1], color=COLORS['light_gray'], lw=2, linestyle='--', label='随机猜测')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('假阳性率 (1 - 特异性)', fontsize=13)
    ax.set_ylabel('真阳性率 (灵敏度)', fontsize=13)
    ax.set_title('ROC曲线对比 - 铅毒性风险预测模型', fontsize=15, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"ROC曲线已保存: {save_path}")
    
    plt.close()
    return fig


def plot_precision_recall_curves(trained_models, X_test, y_test, save_path=None):
    """绘制精确率-召回率曲线"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = [COLORS['primary'], COLORS['tertiary'], COLORS['quaternary'], COLORS['accent']]
    
    for i, (name, model) in enumerate(trained_models.items()):
        y_prob = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        
        ax.plot(recall, precision, color=colors[i], lw=2.5,
                label=f'{name} (AP = {ap:.3f})')
    
    # 基线
    baseline = y_test.mean()
    ax.axhline(y=baseline, color=COLORS['light_gray'], lw=2, linestyle='--',
              label=f'基线 (Prevalence = {baseline:.3f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('召回率 (灵敏度)', fontsize=13)
    ax.set_ylabel('精确率', fontsize=13)
    ax.set_title('精确率-召回率曲线对比', fontsize=15, fontweight='bold')
    ax.legend(loc="lower left", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"PR曲线已保存: {save_path}")
    
    plt.close()
    return fig


def plot_calibration_curves(trained_models, X_test, y_test, save_path=None):
    """绘制校准曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    models_list = list(trained_models.items())
    colors = [COLORS['primary'], COLORS['tertiary'], COLORS['quaternary'], COLORS['accent']]
    
    for idx, (name, model) in enumerate(models_list):
        ax = axes[idx // 2, idx % 2]
        
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # 校准曲线
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy='uniform')
        
        # 计算Brier Score
        brier = brier_score_loss(y_test, y_prob)
        
        # 绘制
        ax.plot(prob_pred, prob_true, 's-', color=colors[idx], lw=2, markersize=8, label='模型')
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='完美校准')
        
        ax.fill_between(prob_pred, prob_true, prob_pred, alpha=0.2, color=colors[idx])
        
        ax.set_xlabel('预测概率', fontsize=12)
        ax.set_ylabel('真实阳性比例', fontsize=12)
        ax.set_title(f'{name}\nBrier Score = {brier:.4f}', fontsize=13, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('模型校准曲线分析', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"校准曲线已保存: {save_path}")
    
    plt.close()
    return fig


def plot_threshold_analysis(trained_models, X_test, y_test, save_path=None):
    """决策阈值优化分析"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    models_list = list(trained_models.items())
    colors = [COLORS['primary'], COLORS['tertiary'], COLORS['quaternary'], COLORS['accent']]
    
    for idx, (name, model) in enumerate(models_list):
        ax = axes[idx // 2, idx % 2]
        
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        
        # 计算敏感度和特异度
        sensitivity = tpr
        specificity = 1 - fpr
        youden = tpr - fpr
        
        # 找到最优阈值 (Youden指数)
        optimal_idx = np.argmax(youden)
        optimal_threshold = thresholds[optimal_idx]
        
        ax.plot(thresholds, sensitivity, color=COLORS['quaternary'], lw=2, label='灵敏度 (TPR)')
        ax.plot(thresholds, specificity, color=COLORS['tertiary'], lw=2, label='特异度 (1-FPR)')
        ax.plot(thresholds, youden, color=COLORS['accent'], lw=2, label='Youden指数')
        
        # 标记最优阈值
        ax.axvline(x=optimal_threshold, color=COLORS['secondary'], linestyle='--', lw=2)
        ax.scatter([optimal_threshold], [youden[optimal_idx]], color=COLORS['secondary'], s=100, zorder=5)
        ax.annotate(f'最优阈值: {optimal_threshold:.3f}', 
                   xy=(optimal_threshold, youden[optimal_idx]),
                   xytext=(optimal_threshold + 0.1, youden[optimal_idx] - 0.1),
                   fontsize=10, fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color=COLORS['secondary']))
        
        ax.set_xlabel('决策阈值', fontsize=12)
        ax.set_ylabel('分数', fontsize=12)
        ax.set_title(f'{name}\n最优阈值: {optimal_threshold:.3f}', fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
    
    plt.suptitle('决策阈值优化分析', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"阈值分析图已保存: {save_path}")
    
    plt.close()
    return fig


def plot_confusion_matrix_heatmap(trained_models, X_test, y_test, save_path=None):
    """绘制混淆矩阵热图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    models_list = list(trained_models.items())
    colors = [COLORS['primary'], COLORS['tertiary'], COLORS['quaternary'], COLORS['accent']]
    
    for idx, (name, model) in enumerate(models_list):
        ax = axes[idx // 2, idx % 2]
        
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        # 归一化
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', ax=ax,
                   xticklabels=['预测阴性', '预测阳性'],
                   yticklabels=['实际阴性', '实际阳性'],
                   cbar_kws={'label': '比例'},
                   annot_kws={'size': 14, 'fontweight': 'bold'})
        
        # 添加数值
        for i in range(2):
            for j in range(2):
                text_color = 'white' if cm_normalized[i, j] > 0.5 else 'black'
                ax.text(j + 0.5, i + 0.7, f'n={cm[i, j]}', 
                       ha='center', va='center', fontsize=10, color=text_color)
        
        accuracy = accuracy_score(y_test, y_pred)
        ax.set_title(f'{name}\n准确率: {accuracy:.2%}', fontsize=13, fontweight='bold')
        ax.set_xlabel('预测类别', fontsize=11)
        ax.set_ylabel('实际类别', fontsize=11)
    
    plt.suptitle('混淆矩阵分析', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"混淆矩阵已保存: {save_path}")
    
    plt.close()
    return fig


def plot_feature_importance(trained_models, feature_names, save_path=None):
    """绘制特征重要性对比"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    models_list = list(trained_models.items())
    colors = [COLORS['primary'], COLORS['tertiary'], COLORS['quaternary'], COLORS['accent']]
    
    for idx, (name, model) in enumerate(models_list):
        ax = axes[idx // 2, idx % 2]
        
        # 获取特征重要性
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            title = '特征重要性 (基于增益)'
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
            title = '特征系数 (绝对值)'
        else:
            continue
        
        # 排序
        sorted_idx = np.argsort(importance)[::-1][:15]  # 前15个
        
        # 绘制水平条形图
        y_pos = np.arange(len(sorted_idx))
        ax.barh(y_pos, importance[sorted_idx], color=colors[idx], alpha=0.85, edgecolor='white')
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel('重要性分数', fontsize=12)
        ax.set_title(f'{name}\n{title}', fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
    
    plt.suptitle('特征重要性对比分析 (Top 15)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"特征重要性图已保存: {save_path}")
    
    plt.close()
    return fig


def generate_report(results, trained_models, X_test, y_test, save_path=None):
    """生成综合分析报告"""
    report = []
    report.append("=" * 70)
    report.append("铅网络毒理学 - 综合模型评估报告")
    report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 70)
    report.append("")
    
    # 1. 数据概览
    report.append("【数据概览】")
    report.append(f"  样本数量: {len(y_test)}")
    report.append(f"  特征数量: {X_test.shape[1]}")
    report.append(f"  阳性比例: {y_test.mean():.2%}")
    report.append("")
    
    # 2. 模型性能对比
    report.append("【模型性能对比】")
    report.append("-" * 70)
    for _, row in results.iterrows():
        report.append(f"\n  {row['Model']}:")
        report.append(f"    AUC: {row['AUC']:.4f}")
        report.append(f"    准确率: {row['Accuracy']:.4f}")
        report.append(f"    精确率: {row['Precision']:.4f}")
        report.append(f"    召回率: {row['Recall']:.4f}")
        report.append(f"    F1分数: {row['F1-Score']:.4f}")
        report.append(f"    Brier Score: {row['Brier Score']:.4f}")
        report.append(f"    平衡准确率: {row['Balanced Accuracy']:.4f}")
    report.append("")
    
    # 3. 最佳模型推荐
    report.append("【最佳模型推荐】")
    best_auc_idx = results['AUC'].idxmax()
    best_f1_idx = results['F1-Score'].idxmax()
    best_brier_idx = results['Brier Score'].idxmin()
    
    report.append(f"  AUC最高: {results.loc[best_auc_idx, 'Model']} ({results.loc[best_auc_idx, 'AUC']:.4f})")
    report.append(f"  F1最高: {results.loc[best_f1_idx, 'Model']} ({results.loc[best_f1_idx, 'F1-Score']:.4f})")
    report.append(f"  校准最佳: {results.loc[best_brier_idx, 'Model']} (Brier={results.loc[best_brier_idx, 'Brier Score']:.4f})")
    report.append("")
    
    # 4. 建议
    report.append("【临床应用建议】")
    report.append("  - 若注重筛查灵敏度: 选择召回率最高的模型")
    report.append("  - 若注重诊断特异性: 选择精确率最高的模型")
    report.append("  - 若注重概率预测: 选择Brier Score最低的模型")
    report.append("  - 综合考虑: 选择AUC和F1分数平衡的模型")
    report.append("")
    report.append("=" * 70)
    
    report_text = "\n".join(report)
    
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"报告已保存: {save_path}")
    
    return report_text


def main():
    """主函数"""
    print("=" * 60)
    print("铅网络毒理学 - 增强综合分析")
    print("=" * 60)
    
    # 1. 生成数据
    print("\n[1/8] 生成模拟数据...")
    df = generate_demo_data(n_samples=3000)
    
    # 准备特征和标签
    feature_cols = [col for col in df.columns if col != 'Outcome']
    X = df[feature_cols]
    y = df['Outcome']
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"  训练集: {len(y_train)} 样本")
    print(f"  测试集: {len(y_test)} 样本")
    print(f"  特征数: {len(feature_cols)}")
    
    # 2. 训练模型
    print("\n[2/8] 训练多个机器学习模型...")
    trained_models = train_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # 3. 评估模型
    print("\n[3/8] 综合评估模型...")
    results = evaluate_models(trained_models, X_test_scaled, y_test)
    print(results.to_string(index=False))
    
    # 4. 保存评估结果
    results_path = os.path.join(OUTPUT_DIR, 'enhanced_model_comparison.csv')
    results.to_csv(results_path, index=False)
    print(f"\n评估结果已保存: {results_path}")
    
    # 5. 绘制可视化图表
    print("\n[4/8] 生成可视化图表...")
    
    # 模型对比
    plot_model_comparison(results, os.path.join(OUTPUT_DIR, 'enhanced_model_comparison.png'))
    
    # ROC曲线
    plot_roc_curves(trained_models, X_test_scaled, y_test, 
                   os.path.join(OUTPUT_DIR, 'enhanced_roc_curves.png'))
    
    # PR曲线
    plot_precision_recall_curves(trained_models, X_test_scaled, y_test,
                                os.path.join(OUTPUT_DIR, 'enhanced_pr_curves.png'))
    
    # 校准曲线
    plot_calibration_curves(trained_models, X_test_scaled, y_test,
                          os.path.join(OUTPUT_DIR, 'enhanced_calibration_curves.png'))
    
    # 阈值分析
    plot_threshold_analysis(trained_models, X_test_scaled, y_test,
                           os.path.join(OUTPUT_DIR, 'enhanced_threshold_analysis.png'))
    
    # 混淆矩阵
    plot_confusion_matrix_heatmap(trained_models, X_test_scaled, y_test,
                                 os.path.join(OUTPUT_DIR, 'enhanced_confusion_matrices.png'))
    
    # 特征重要性
    plot_feature_importance(trained_models, feature_cols,
                          os.path.join(OUTPUT_DIR, 'enhanced_feature_importance.png'))
    
    # 6. 生成报告
    print("\n[5/8] 生成综合分析报告...")
    report = generate_report(results, trained_models, X_test_scaled, y_test,
                            os.path.join(OUTPUT_DIR, 'enhanced_analysis_report.txt'))
    print(report)
    
    # 7. 总结
    print("\n[6/8] 分析完成!")
    print("=" * 60)
    print("生成的文件:")
    print("  - enhanced_model_comparison.csv")
    print("  - enhanced_model_comparison.png")
    print("  - enhanced_roc_curves.png")
    print("  - enhanced_pr_curves.png")
    print("  - enhanced_calibration_curves.png")
    print("  - enhanced_threshold_analysis.png")
    print("  - enhanced_confusion_matrices.png")
    print("  - enhanced_feature_importance.png")
    print("  - enhanced_analysis_report.txt")
    print("=" * 60)
    
    return results, trained_models


if __name__ == "__main__":
    results, models = main()
