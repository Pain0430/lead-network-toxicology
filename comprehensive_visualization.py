#!/usr/bin/env python3
"""
铅网络毒理学 - 多模型综合可视化模块
Lead Network Toxicology - Multi-Model Comprehensive Visualization

功能：
1. 多模型ROC/PR对比图
2. 特征分布小提琴图
3. 特征相关性网络图
4. 模型性能综合仪表板

作者: Pain (重庆医科大学)
日期: 2026-02-25
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, brier_score_loss, f1_score, accuracy_score
)
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

# 专业配色方案
COLORS = {
    'RF': '#2ecc71',      # 绿色
    'GB': '#3498db',      # 蓝色  
    'LR': '#e74c3c',      # 红色
    'SVM': '#9b59b6',     # 紫色
    'XGB': '#f39c12'      # 橙色
}

# ============================================================
# 数据生成
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
        '8-OHdG_ng_mL': np.random.lognormal(1.5, 0.6, n_samples),
        'CRP_mg_L': np.random.lognormal(1.0, 1.2, n_samples).clip(0.1, 50),
        'IL6_pg_mL': np.random.lognormal(2.0, 0.8, n_samples).clip(1, 100),
        'TNF_alpha_pg_mL': np.random.lognormal(1.8, 0.7, n_samples).clip(5, 80),
        'ALT_U_L': np.random.normal(25, 10, n_samples).clip(5, 200),
        'AST_U_L': np.random.normal(28, 12, n_samples).clip(10, 250),
        'Creatinine_umol_L': np.random.normal(80, 20, n_samples).clip(30, 200),
        'BUN_mmol_L': np.random.normal(5, 1.5, n_samples).clip(2, 20),
        'Systolic_BP_mmHg': np.random.normal(130, 20, n_samples).clip(90, 200),
        'Diastolic_BP_mmHg': np.random.normal(82, 12, n_samples).clip(60, 120),
        'HbA1c_percent': np.random.normal(5.5, 1.0, n_samples).clip(4.0, 12.0),
        'Total_Cholesterol_mmol_L': np.random.normal(5.2, 1.2, n_samples).clip(2.5, 10),
        'DCA_umol_L': np.random.lognormal(2.0, 0.7, n_samples).clip(0.5, 50),
        'LCA_umol_L': np.random.lognormal(1.0, 0.8, n_samples).clip(0.1, 30),
        'CA_umol_L': np.random.lognormal(2.5, 0.6, n_samples).clip(1, 60),
        'UDCA_umol_L': np.random.lognormal(1.5, 0.7, n_samples).clip(0.5, 40),
        'Calprotectin_ug_g': np.random.lognormal(2.0, 1.0, n_samples).clip(10, 500),
        'Zonulin_ng_mL': np.random.lognormal(1.0, 0.5, n_samples).clip(5, 100),
        'LBP_ug_mL': np.random.normal(15, 5, n_samples).clip(3, 50),
    }
    
    df = pd.DataFrame(data)
    
    # 生成目标变量 (铅毒性风险)
    risk_score = (
        0.4 * (df['Blood_Lead_ug_dL'] / 20) +
        0.3 * (df['Urine_Lead_ug_L'] / 100) +
        0.3 * (df['MDA_umol_L'] / 5) +
        0.2 * (df['CRP_mg_L'] / 10) -
        0.2 * (df['SOD_U_mL'] / 150) +
        0.15 * (df['DCA_umol_L'] / 20) +
        0.1 * (df['Age'] / 50)
    )
    
    df['Lead_Toxicity'] = (risk_score > np.percentile(risk_score, 75)).astype(int)
    
    return df

# ============================================================
# 第一部分：多模型综合ROC/PR对比图
# ============================================================

def plot_multi_model_roc_pr_comparison(models, X_test, y_test, output_path='output/'):
    """创建多模型ROC和PR曲线对比图"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # ROC曲线
    ax1 = axes[0]
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1.5, label='Random')
    
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, color=COLORS.get(name, '#666666'), 
                linewidth=2.5, label=f'{name} (AUC = {roc_auc:.3f})')
    
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # PR曲线
    ax2 = axes[1]
    
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        ax2.plot(recall, precision, color=COLORS.get(name, '#666666'),
                linewidth=2.5, label=f'{name} (AUC = {pr_auc:.3f})')
    
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('PR Curves Comparison', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_path}multi_model_roc_pr_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_path}multi_model_roc_pr_comparison.pdf', bbox_inches='tight')
    plt.close()
    print(f"✅ 多模型ROC/PR对比图已保存")

# ============================================================
# 第二部分：特征分布小提琴图
# ============================================================

def plot_feature_distributions(df, target_col='Lead_Toxicity', output_path='output/'):
    """创建关键特征分布小提琴图"""
    
    # 选择重要特征
    key_features = [
        'Blood_Lead_ug_dL', 'Urine_Lead_ug_L', 
        'MDA_umol_L', 'CRP_mg_L', 
        'DCA_umol_L', 'Calprotectin_ug_g'
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, feature in enumerate(key_features):
        ax = axes[idx]
        
        # 标准化数据用于可视化
        data_0 = df[df[target_col] == 0][feature]
        data_1 = df[df[target_col] == 1][feature]
        
        # 合并数据
        plot_data = pd.DataFrame({
            'Value': pd.concat([data_0, data_1]),
            'Group': ['Non-Toxicity'] * len(data_0) + ['Toxicity'] * len(data_1)
        })
        
        # 小提琴图
        parts = ax.violinplot([data_0, data_1], positions=[0, 1], 
                             showmeans=True, showmedians=True)
        
        # 设置颜色
        colors = ['#3498db', '#e74c3c']
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Non-Toxicity', 'Toxicity'])
        ax.set_ylabel(feature.replace('_', ' '), fontsize=11)
        ax.set_title(f'{feature} Distribution', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Feature Distributions by Lead Toxicity Status', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_path}feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_path}feature_distributions.pdf', bbox_inches='tight')
    plt.close()
    print(f"✅ 特征分布小提琴图已保存")

# ============================================================
# 第三部分：模型性能综合仪表板
# ============================================================

def plot_model_performance_dashboard(models, X_test, y_test, output_path='output/'):
    """创建模型性能综合仪表板"""
    
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)
    
    # 计算各模型指标
    metrics_data = []
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        
        metrics_data.append({
            'Model': name,
            'ROC-AUC': roc_auc,
            'PR-AUC': pr_auc,
            'Accuracy': accuracy_score(y_test, y_pred),
            'F1': f1_score(y_test, y_pred),
            'Brier': brier_score_loss(y_test, y_prob)
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # 1. ROC曲线 (左上)
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    for _, row in metrics_df.iterrows():
        fpr, tpr, _ = roc_curve(y_test, models[row['Model']].predict_proba(X_test)[:, 1])
        ax1.plot(fpr, tpr, color=COLORS.get(row['Model'], '#666'), 
                linewidth=2, label=f"{row['Model']} ({row['ROC-AUC']:.3f})")
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curves', fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # 2. PR曲线 (右上)
    ax2 = fig.add_subplot(gs[0, 2:4])
    for _, row in metrics_df.iterrows():
        precision, recall, _ = precision_recall_curve(y_test, models[row['Model']].predict_proba(X_test)[:, 1])
        ax2.plot(recall, precision, color=COLORS.get(row['Model'], '#666'),
                linewidth=2, label=f"{row['Model']} ({row['PR-AUC']:.3f})")
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('PR Curves', fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # 3. 模型性能条形图
    ax3 = fig.add_subplot(gs[1, 0:2])
    x = np.arange(len(metrics_df))
    width = 0.35
    ax3.bar(x - width/2, metrics_df['ROC-AUC'], width, label='ROC-AUC', color='#3498db')
    ax3.bar(x + width/2, metrics_df['PR-AUC'], width, label='PR-AUC', color='#e74c3c')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics_df['Model'])
    ax3.set_ylim([0, 1.1])
    ax3.set_ylabel('Score')
    ax3.set_title('Model Performance Comparison', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Radar图
    ax4 = fig.add_subplot(gs[1, 2:4], polar=True)
    categories = ['ROC-AUC', 'PR-AUC', 'Accuracy', 'F1']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    for _, row in metrics_df.iterrows():
        values = [row['ROC-AUC'], row['PR-AUC'], row['Accuracy'], row['F1']]
        values += values[:1]
        ax4.plot(angles, values, 'o-', linewidth=2, 
                label=row['Model'], color=COLORS.get(row['Model'], '#666'))
        ax4.fill(angles, values, alpha=0.1, color=COLORS.get(row['Model'], '#666'))
    
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(categories)
    ax4.set_title('Model Radar Chart', fontweight='bold', pad=20)
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    # 5. 热力图
    ax5 = fig.add_subplot(gs[2, 0:2])
    heatmap_data = metrics_df.set_index('Model')[['ROC-AUC', 'PR-AUC', 'Accuracy', 'F1', 'Brier']]
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                ax=ax5, vmin=0, vmax=1, center=0.5)
    ax5.set_title('Performance Metrics Heatmap', fontweight='bold')
    
    # 6. Brier Score对比
    ax6 = fig.add_subplot(gs[2, 2:4])
    bars = ax6.barh(metrics_df['Model'], metrics_df['Brier'], 
                   color=[COLORS.get(m, '#666') for m in metrics_df['Model']])
    ax6.set_xlabel('Brier Score (Lower is Better)')
    ax6.set_title('Model Calibration (Brier Score)', fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='x')
    for bar, val in zip(bars, metrics_df['Brier']):
        ax6.text(val + 0.002, bar.get_y() + bar.get_height()/2, 
                f'{val:.4f}', va='center', fontsize=10)
    
    plt.suptitle('Lead Toxicology Model Performance Dashboard', 
                 fontsize=18, fontweight='bold', y=0.98)
    plt.savefig(f'{output_path}model_performance_dashboard.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_path}model_performance_dashboard.pdf', bbox_inches='tight')
    plt.close()
    print(f"✅ 模型性能综合仪表板已保存")
    
    # 保存指标数据
    metrics_df.to_csv(f'{output_path}comprehensive_metrics.csv', index=False)
    print(f"✅ 综合指标数据已保存")

# ============================================================
# 第四部分：特征相关性网络热力图
# ============================================================

def plot_correlation_heatmap(df, output_path='output/'):
    """创建特征相关性热力图"""
    
    # 选择数值型特征
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Lead_Toxicity' in numeric_cols:
        numeric_cols.remove('Lead_Toxicity')
    if 'Sex' in numeric_cols:
        numeric_cols.remove('Sex')
    
    corr_matrix = df[numeric_cols].corr()
    
    # 创建大型热力图
    fig, ax = plt.subplots(figsize=(20, 16))
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    cmap = sns.diverging_palette(250, 10, as_cmap=True)
    
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, center=0,
                annot=True, fmt='.2f', annot_kws={'size': 7},
                square=True, linewidths=0.5,
                cbar_kws={'shrink': 0.8, 'label': 'Correlation'})
    
    plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    
    plt.savefig(f'{output_path}correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_path}correlation_heatmap.pdf', bbox_inches='tight')
    plt.close()
    print(f"✅ 相关性热力图已保存")

# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print("铅网络毒理学 - 多模型综合可视化模块")
    print("=" * 60)
    
    # 创建输出目录
    import os
    output_path = 'output/'
    os.makedirs(output_path, exist_ok=True)
    
    # 生成数据
    print("\n📊 正在生成数据...")
    df = generate_lead_toxicology_data(n_samples=2000)
    df.to_csv(f'{output_path}lead_toxicology_data.csv', index=False)
    print(f"✅ 数据已生成: {len(df)} 样本")
    
    # 准备特征和目标
    feature_cols = [col for col in df.columns if col != 'Lead_Toxicity']
    X = df[feature_cols]
    y = df['Lead_Toxicity']
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练模型
    print("\n🔬 正在训练模型...")
    models = {
        'RF': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'GB': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
        'LR': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    for name, model in models.items():
        if name == 'LR':
            model.fit(X_train_scaled, y_train)
        else:
            model.fit(X_train, y_train)
        print(f"✅ {name} 模型训练完成")
    
    # 生成可视化
    print("\n📈 正在生成可视化...")
    
    # 1. 多模型ROC/PR对比
    plot_multi_model_roc_pr_comparison(models, X_test_scaled if hasattr(X_test_scaled, 'shape') else X_test, y_test, output_path)
    
    # 2. 特征分布
    plot_feature_distributions(df, output_path=output_path)
    
    # 3. 模型性能仪表板
    plot_model_performance_dashboard(models, X_test_scaled if hasattr(X_test_scaled, 'shape') else X_test, y_test, output_path)
    
    # 4. 相关性热力图
    plot_correlation_heatmap(df, output_path)
    
    print("\n" + "=" * 60)
    print("🎉 所有可视化已完成!")
    print("=" * 60)
    
    # 输出文件列表
    print("\n📁 生成的文件:")
    import glob
    files = glob.glob(f'{output_path}*')
    for f in sorted(files):
        size = os.path.getsize(f) / 1024
        print(f"  - {os.path.basename(f)} ({size:.1f} KB)")

if __name__ == '__main__':
    main()
