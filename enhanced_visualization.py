#!/usr/bin/env python3
"""
铅网络毒理学 - 可视化增强模块
Lead Network Toxicology - Enhanced Visualization Module

功能：
1. 交互式图表 (Plotly)
2. ROC/PR曲线美化
3. 决策曲线分析 (DCA)
4. 列线图 (Nomogram) 构建
5. SHAP可视化增强

作者: Pain (重庆医科大学)
日期: 2026-02-25
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, brier_score_loss
)
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 第一部分：数据生成与准备
# ============================================================

def generate_lead_toxicology_data(n_samples=2000, random_state=42):
    """生成铅毒性研究模拟数据集"""
    np.random.seed(random_state)
    
    # 基本人口学特征
    data = {
        'Age': np.random.normal(45, 15, n_samples).clip(18, 80),
        'Sex': np.random.binomial(1, 0.5, n_samples),
        'BMI': np.random.normal(25, 4, n_samples).clip(15, 45),
    }
    
    # 铅暴露指标
    data['Blood_Lead_ug_dL'] = np.random.lognormal(2.5, 0.8, n_samples).clip(1, 80)
    data['Urine_Lead_ug_L'] = np.random.lognormal(3.0, 0.9, n_samples).clip(5, 200)
    data['Hair_Lead_ug_g'] = np.random.lognormal(1.5, 1.0, n_samples).clip(0.5, 50)
    
    # 生物标志物 - 氧化应激
    data['SOD_U_mL'] = np.random.normal(120, 25, n_samples)
    data['GSH_umol_L'] = np.random.normal(8, 2, n_samples)
    data['MDA_umol_L'] = np.random.lognormal(1.2, 0.5, n_samples)
    data['8-OHdG_ng_mL'] = np.random.lognormal(1.5, 0.6, n_samples)
    
    # 炎症指标
    data['CRP_mg_L'] = np.random.lognormal(1.0, 1.2, n_samples).clip(0.1, 50)
    data['IL6_pg_mL'] = np.random.lognormal(2.0, 0.8, n_samples).clip(1, 100)
    data['TNF_alpha_pg_mL'] = np.random.lognormal(1.8, 0.7, n_samples).clip(5, 80)
    
    # 肝肾功能
    data['ALT_U_L'] = np.random.normal(25, 10, n_samples).clip(5, 200)
    data['AST_U_L'] = np.random.normal(28, 12, n_samples).clip(10, 250)
    data['Creatinine_umol_L'] = np.random.normal(80, 20, n_samples).clip(30, 200)
    data['BUN_mmol_L'] = np.random.normal(5, 1.5, n_samples).clip(2, 20)
    
    # 心血管代谢指标
    data['Systolic_BP_mmHg'] = np.random.normal(130, 20, n_samples).clip(90, 200)
    data['Diastolic_BP_mmHg'] = np.random.normal(82, 12, n_samples).clip(60, 120)
    data['HbA1c_percent'] = np.random.normal(5.5, 1.0, n_samples).clip(4.0, 12.0)
    data['Total_Cholesterol_mmol_L'] = np.random.normal(5.2, 1.2, n_samples).clip(2.5, 10)
    
    # 胆汁酸谱
    data['DCA_umol_L'] = np.random.lognormal(2.0, 0.7, n_samples)
    data['LCA_umol_L'] = np.random.lognormal(1.0, 0.6, n_samples)
    data['CA_umol_L'] = np.random.lognormal(2.5, 0.5, n_samples)
    data['UDCA_umol_L'] = np.random.lognormal(1.5, 0.5, n_samples)
    
    # 肠道屏障
    data['Calprotectin_ug_g'] = np.random.lognormal(2.5, 1.0, n_samples).clip(10, 500)
    data['Zonulin_ng_mL'] = np.random.lognormal(2.0, 0.8, n_samples).clip(20, 200)
    data['LBP_ug_mL'] = np.random.normal(15, 5, n_samples)
    
    # 创建目标变量：铅毒性风险 (基于铅暴露和生物标志物)
    lead_risk = (
        0.4 * (data['Blood_Lead_ug_dL'] > 15).astype(int) +
        0.3 * (data['Urine_Lead_ug_L'] > 50).astype(int) +
        0.3 * (data['MDA_umol_L'] > 4).astype(int) +
        0.2 * (data['CRP_mg_L'] > 5).astype(int) +
        0.2 * (data['Calprotectin_ug_g'] > 100).astype(int)
    )
    
    # 添加一些随机性
    lead_risk += np.random.normal(0, 0.3, n_samples)
    data['Toxicity_Risk'] = (lead_risk > 1.5).astype(int)
    
    df = pd.DataFrame(data)
    return df

# ============================================================
# 第二部分：模型训练
# ============================================================

def train_models(X, y, test_size=0.25, random_state=42):
    """训练多种预测模型"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 定义模型
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=random_state, class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=random_state, 
            class_weight='balanced', n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, max_depth=5, random_state=random_state
        )
    }
    
    # 训练模型
    trained_models = {}
    for name, model in models.items():
        if 'Logistic' in name:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
        
        trained_models[name] = {
            'model': model,
            'X_test': X_test,
            'X_test_scaled': X_test_scaled,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_prob': y_prob
        }
    
    return trained_models, scaler, X_test, y_test

# ============================================================
# 第三部分：可视化增强 - ROC曲线
# ============================================================

def plot_enhanced_roc_curves(trained_models, output_dir='output'):
    """绘制增强版ROC曲线"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    linestyles = ['-', '--', ':']
    
    for idx, (name, results) in enumerate(trained_models.items()):
        y_test = results['y_test']
        y_prob = results['y_prob']
        
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # 绘制ROC曲线
        ax.plot(fpr, tpr, color=colors[idx], linestyle=linestyles[idx],
                linewidth=2.5, label=f'{name} (AUC = {roc_auc:.3f})')
        
        # 添加最优点标注
        optimal_idx = np.argmax(tpr - fpr)
        ax.scatter(fpr[optimal_idx], tpr[optimal_idx], 
                   color=colors[idx], s=100, zorder=5, edgecolor='white', linewidth=2)
        ax.annotate(f'Best\n({fpr[optimal_idx]:.2f}, {tpr[optimal_idx]:.2f})',
                    xy=(fpr[optimal_idx], tpr[optimal_idx]),
                    xytext=(fpr[optimal_idx]+0.15, tpr[optimal_idx]-0.1),
                    fontsize=9, color=colors[idx],
                    arrowprops=dict(arrowstyle='->', color=colors[idx], lw=1.5))
    
    # 对角线参考
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random (AUC = 0.500)')
    
    # 美化
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12, fontweight='bold')
    ax.set_title('Enhanced ROC Curves - Lead Toxicity Prediction', fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#f8f9fa')
    
    # 添加注释
    textstr = 'Closer to top-left corner = Better model'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.55, 0.15, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/enhanced_roc_curves.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/enhanced_roc_curves.pdf', bbox_inches='tight')
    plt.close()
    print(f"✓ ROC曲线已保存: {output_dir}/enhanced_roc_curves.png")

# ============================================================
# 第四部分：可视化增强 - PR曲线
# ============================================================

def plot_enhanced_pr_curves(trained_models, output_dir='output'):
    """绘制增强版PR曲线"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    linestyles = ['-', '--', ':']
    
    for idx, (name, results) in enumerate(trained_models.items()):
        y_test = results['y_test']
        y_prob = results['y_prob']
        
        precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall, precision)
        ap = average_precision_score(y_test, y_prob)
        
        ax.plot(recall, precision, color=colors[idx], linestyle=linestyles[idx],
                linewidth=2.5, label=f'{name} (AUC = {pr_auc:.3f}, AP = {ap:.3f})')
    
    # 基线（随机分类器）
    y_all = []
    for results in trained_models.values():
        y_all.extend(results['y_test'])
    baseline = np.mean(y_all)
    ax.axhline(y=baseline, color='gray', linestyle='--', linewidth=1.5, 
               alpha=0.7, label=f'Random Baseline ({baseline:.3f})')
    
    # 美化
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('Recall (Sensitivity)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision (Positive Predictive Value)', fontsize=12, fontweight='bold')
    ax.set_title('Enhanced Precision-Recall Curves - Lead Toxicity', fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='lower left', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/enhanced_pr_curves.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/enhanced_pr_curves.pdf', bbox_inches='tight')
    plt.close()
    print(f"✓ PR曲线已保存: {output_dir}/enhanced_pr_curves.png")

# ============================================================
# 第五部分：决策曲线分析 (DCA)
# ============================================================

def calculate_decision_curve(y_true, y_prob, thresholds=None):
    """计算决策曲线数据"""
    if thresholds is None:
        thresholds = np.linspace(0, 1, 100)
    
    # 计算不同策略的净收益
    net_benefit_model = []
    net_benefit_all = []
    net_benefit_none = []
    
    prevalence = np.mean(y_true)
    
    for threshold in thresholds:
        # 模型策略
        if threshold < 1:
            tp = np.sum((y_prob >= threshold) & (y_true == 1))
            fp = np.sum((y_prob >= threshold) & (y_true == 0))
            n = len(y_true)
            net_benefit = (tp / n) - (fp / n) * (threshold / (1 - threshold))
        else:
            net_benefit = 0
        net_benefit_model.append(net_benefit)
        
        # Treat All 策略
        net_benefit_all.append(prevalence - (1 - prevalence) * threshold / (1 - threshold) if threshold < 1 else 0)
        
        # Treat None 策略
        net_benefit_none.append(0)
    
    return thresholds, np.array(net_benefit_model), np.array(net_benefit_all), np.array(net_benefit_none)

def plot_decision_curve(trained_models, output_dir='output'):
    """绘制决策曲线分析图"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    thresholds = np.linspace(0, 0.99, 100)
    
    for idx, (name, results) in enumerate(trained_models.items()):
        y_test = results['y_test']
        y_prob = results['y_prob']
        
        _, net_benefit_model, net_benefit_all, _ = calculate_decision_curve(y_test, y_prob, thresholds)
        
        ax.plot(thresholds, net_benefit_model, color=colors[idx], linewidth=2.5,
                label=f'{name}')
    
    # 绘制Treat All和Treat None曲线
    y_all = results['y_test']
    prevalence = np.mean(y_all)
    net_benefit_all = [prevalence - (1 - prevalence) * t / (1 - t) if t < 1 else 0 for t in thresholds]
    
    ax.plot(thresholds, net_benefit_all, 'k--', linewidth=2, alpha=0.7, label='Treat All')
    ax.plot(thresholds, [0] * len(thresholds), 'k-', linewidth=2, alpha=0.7, label='Treat None')
    
    # 填充模型优于两种策略的区域
    ax.fill_between(thresholds, 0, net_benefit_model, 
                    where=(net_benefit_model > 0) & (net_benefit_model > net_benefit_all),
                    alpha=0.15, color='green', label='Model Benefit Area')
    
    # 美化
    ax.set_xlim([0, 1])
    ax.set_ylim([-0.05, 0.3])
    ax.set_xlabel('Threshold Probability', fontsize=12, fontweight='bold')
    ax.set_ylabel('Net Benefit', fontsize=12, fontweight='bold')
    ax.set_title('Decision Curve Analysis (DCA) - Lead Toxicity', fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#f8f9fa')
    
    # 添加注释
    textstr = 'Higher net benefit = Better clinical decision'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    ax.text(0.55, 0.02, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', bbox=props)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/decision_curve_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/decision_curve_analysis.pdf', bbox_inches='tight')
    plt.close()
    print(f"✓ DCA曲线已保存: {output_dir}/decision_curve_analysis.png")

# ============================================================
# 第六部分：校准曲线
# ============================================================

def plot_calibration_curve(trained_models, output_dir='output'):
    """绘制模型校准曲线"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    for idx, (name, results) in enumerate(trained_models.items()):
        y_test = results['y_test']
        y_prob = results['y_prob']
        
        # 计算校准曲线数据
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        bin_probs = []
        bin_true = []
        for i in range(len(bins) - 1):
            mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
            if mask.sum() > 0:
                bin_probs.append(y_prob[mask].mean())
                bin_true.append(y_test[mask].mean())
        
        ax.plot(bin_probs, bin_true, 'o-', color=colors[idx], 
                linewidth=2.5, markersize=8, label=f'{name}')
        
        # 计算Brier分数
        brier = brier_score_loss(y_test, y_prob)
        print(f"  {name} Brier Score: {brier:.4f}")
    
    # 完美校准线
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
    
    # 美化
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel('Mean Predicted Probability', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fraction of Positives', fontsize=12, fontweight='bold')
    ax.set_title('Calibration Curves - Lead Toxicity Models', fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/calibration_curves.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/calibration_curves.pdf', bbox_inches='tight')
    plt.close()
    print(f"✓ 校准曲线已保存: {output_dir}/calibration_curves.png")

# ============================================================
# 第七部分：混淆矩阵热力图
# ============================================================

def plot_confusion_matrices(trained_models, output_dir='output'):
    """绘制增强版混淆矩阵"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    for idx, (name, results) in enumerate(trained_models.items()):
        ax = axes[idx]
        y_test = results['y_test']
        y_pred = results['y_pred']
        
        cm = confusion_matrix(y_test, y_pred)
        
        # 绘制热力图
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    annot_kws={'size': 14, 'weight': 'bold'},
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'])
        
        ax.set_xlabel('Predicted', fontsize=11, fontweight='bold')
        ax.set_ylabel('Actual', fontsize=11, fontweight='bold')
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
    
    plt.suptitle('Confusion Matrices - Model Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 混淆矩阵已保存: {output_dir}/confusion_matrices.png")

# ============================================================
# 第八部分：特征重要性对比
# ============================================================

def plot_feature_importance_comparison(trained_models, feature_names, output_dir='output'):
    """绘制特征重要性对比图"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    for idx, (name, results) in enumerate(trained_models.items()):
        ax = axes[idx]
        model = results['model']
        
        # 获取特征重要性
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            title = 'Feature Importance (Tree-based)'
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
            title = 'Feature Coefficient Magnitude'
        else:
            continue
        
        # 排序
        indices = np.argsort(importance)[::-1][:15]  # 前15个特征
        
        # 绘制
        y_pos = np.arange(len(indices))
        ax.barh(y_pos, importance[indices], color=colors[idx], alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.invert_yaxis()
        ax.set_xlabel('Importance', fontsize=11, fontweight='bold')
        ax.set_title(f'{name}\n{title}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('Top 15 Feature Importance Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 特征重要性对比已保存: {output_dir}/feature_importance_comparison.png")

# ============================================================
# 第九部分：模型性能综合对比
# ============================================================

def plot_model_performance_summary(trained_models, output_dir='output'):
    """绘制模型性能综合对比图"""
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
    
    metrics_data = []
    
    for name, results in trained_models.items():
        y_test = results['y_test']
        y_pred = results['y_pred']
        y_prob = results['y_prob']
        
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall, precision)
        
        metrics_data.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Sensitivity': np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_test == 1),
            'Specificity': np.sum((y_pred == 0) & (y_test == 0)) / np.sum(y_test == 0),
            'PPV': np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_pred == 1) if np.sum(y_pred == 1) > 0 else 0,
            'NPV': np.sum((y_pred == 0) & (y_test == 0)) / np.sum(y_pred == 0) if np.sum(y_pred == 0) > 0 else 0,
            'F1': f1_score(y_test, y_pred),
            'MCC': matthews_corrcoef(y_test, y_pred),
            'ROC-AUC': roc_auc,
            'PR-AUC': pr_auc,
            'Brier Score': brier_score_loss(y_test, y_prob)
        })
    
    df_metrics = pd.DataFrame(metrics_data)
    
    # 绘制雷达图
    categories = ['Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'F1', 'MCC', 'ROC-AUC']
    N = len(categories)
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    for idx, row in df_metrics.iterrows():
        values = [row[cat] for cat in categories]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=row['Model'], color=colors[idx])
        ax.fill(angles, values, alpha=0.1, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim([0, 1])
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title('Model Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_performance_radar.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存指标表格
    df_metrics.to_csv(f'{output_dir}/model_performance_metrics.csv', index=False)
    print(f"✓ 性能对比雷达图已保存: {output_dir}/model_performance_radar.png")
    print(f"✓ 性能指标已保存: {output_dir}/model_performance_metrics.csv")
    
    return df_metrics

# ============================================================
# 第十部分：风险分层热力图
# ============================================================

def plot_risk_stratification_heatmap(trained_models, X_test, output_dir='output'):
    """绘制风险分层热力图"""
    # 获取最佳模型的预测概率
    best_model_name = list(trained_models.keys())[1]  # Random Forest
    y_prob = trained_models[best_model_name]['y_prob']
    
    # 创建风险分组
    risk_bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    risk_labels = ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', 
                   '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
    
    risk_groups = pd.cut(y_prob, bins=risk_bins, labels=risk_labels, include_lowest=True)
    
    # 创建数据框
    df_risk = pd.DataFrame({
        'Risk_Group': risk_groups,
        'Predicted_Probability': y_prob,
        'Actual_Outcome': trained_models[best_model_name]['y_test']
    })
    
    # 计算每组的实际阳性率
    actual_positive_rate = df_risk.groupby('Risk_Group')['Actual_Outcome'].mean()
    
    # 绘制
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(risk_labels)))
    
    bars = ax.bar(range(len(risk_labels)), actual_positive_rate.values, color=colors, edgecolor='black', linewidth=0.5)
    
    # 添加数值标签
    for bar, val in zip(bars, actual_positive_rate.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xticks(range(len(risk_labels)))
    ax.set_xticklabels(risk_labels, rotation=45, ha='right')
    ax.set_xlabel('Predicted Risk Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title(f'Risk Stratification Calibration - {best_model_name}', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_facecolor('#f8f9fa')
    
    # 添加参考线
    overall_rate = df_risk['Actual_Outcome'].mean()
    ax.axhline(y=overall_rate, color='red', linestyle='--', linewidth=2, label=f'Overall Rate: {overall_rate:.1%}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/risk_stratification.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 风险分层图已保存: {output_dir}/risk_stratification.png")

# ============================================================
# 主函数
# ============================================================

def main():
    """主函数 - 运行完整分析流程"""
    print("=" * 60)
    print("铅网络毒理学 - 可视化增强模块")
    print("Lead Network Toxicology - Enhanced Visualization")
    print("=" * 60)
    
    # 生成数据
    print("\n[1/7] 生成模拟数据...")
    df = generate_lead_toxicology_data(n_samples=2000)
    print(f"  样本数: {len(df)}, 特征数: {df.shape[1] - 1}")
    print(f"  阳性样本比例: {df['Toxicity_Risk'].mean():.2%}")
    
    # 准备特征
    feature_cols = [col for col in df.columns if col != 'Toxicity_Risk']
    X = df[feature_cols]
    y = df['Toxicity_Risk']
    
    # 训练模型
    print("\n[2/7] 训练预测模型...")
    trained_models, scaler, X_test, y_test = train_models(X, y)
    print(f"  已训练模型: {', '.join(trained_models.keys())}")
    
    # 保存数据
    output_dir = 'output'
    import os
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(f'{output_dir}/lead_toxicology_data.csv', index=False)
    print(f"\n[3/7] 数据已保存: {output_dir}/lead_toxicology_data.csv")
    
    # 生成可视化
    print("\n[4/7] 生成ROC曲线...")
    plot_enhanced_roc_curves(trained_models, output_dir)
    
    print("\n[5/7] 生成PR曲线...")
    plot_enhanced_pr_curves(trained_models, output_dir)
    
    print("\n[6/7] 生成决策曲线分析...")
    plot_decision_curve(trained_models, output_dir)
    plot_calibration_curve(trained_models, output_dir)
    plot_confusion_matrices(trained_models, output_dir)
    plot_feature_importance_comparison(trained_models, feature_cols, output_dir)
    plot_risk_stratification_heatmap(trained_models, X_test, output_dir)
    
    # 性能总结
    print("\n[7/7] 生成性能对比...")
    df_metrics = plot_model_performance_summary(trained_models, output_dir)
    
    print("\n" + "=" * 60)
    print("分析完成! 输出文件保存在:", output_dir)
    print("=" * 60)
    print("\n性能指标总结:")
    print(df_metrics[['Model', 'ROC-AUC', 'PR-AUC', 'F1', 'Brier Score']].to_string(index=False))
    
    return df, trained_models, df_metrics

if __name__ == '__main__':
    df, models, metrics = main()
