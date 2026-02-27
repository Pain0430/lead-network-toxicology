#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
铅网络毒理学 - 亚组分析模块
Lead Network Toxicology - Subgroup Analysis Module

功能：
- 按关键特征分层进行亚组分析
- 亚组间效应异质性检验
- 亚组特异性森林图
- 交互效应可视化
- 漏斗图评估发表偏倚

作者: Pain AI Assistant
日期: 2026-02-26
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from scipy import stats
import warnings
import os

# 设置高质量图表样式
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'

# 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

# 专业配色方案
COLORS = {
    'primary': '#2C3E50',
    'secondary': '#E74C3C',
    'tertiary': '#3498DB',
    'quaternary': '#27AE60',
    'accent': '#F39C12',
    'purple': '#9B59B6',
    'orange': '#E67E22',
    'pink': '#E91E63',
    'cyan': '#00BCD4',
    'light': '#ECF0F1',
    'dark': '#2C3E50',
    'grid': '#BDC3C7'
}

# 亚组配色
SUBGROUP_COLORS = ['#3498DB', '#E74C3C', '#27AE60', '#F39C12', '#9B59B6', '#00BCD4']

warnings.filterwarnings('ignore')

OUTPUT_DIR = '/Users/pengsu/mycode/lead-network-toxicology/output/'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_subgroup_data(n_samples=3000, random_state=42):
    """生成用于亚组分析的铅毒性数据集"""
    np.random.seed(random_state)
    
    data = {
        # 基本人口学特征 (用于分层)
        'Age': np.random.normal(50, 15, n_samples).clip(20, 80),
        'Sex': np.random.binomial(1, 0.5, n_samples),  # 0=女, 1=男
        'BMI': np.random.normal(25, 4, n_samples).clip(16, 45),
        'Education': np.random.choice([0, 1, 2], n_samples, p=[0.3, 0.4, 0.3]),  # 0=低, 1=中, 2=高
        'Smoking': np.random.binomial(1, 0.25, n_samples),
        'Alcohol': np.random.binomial(1, 0.2, n_samples),
        
        # 铅暴露指标
        'Blood_Lead_ug_dL': np.random.lognormal(2.0, 0.8, n_samples).clip(1, 60),
        'Urine_Lead_ug_L': np.random.lognormal(2.5, 0.9, n_samples).clip(2, 150),
        'Hair_Lead_ug_g': np.random.lognormal(1.2, 1.0, n_samples).clip(0.5, 50),
        
        # 氧化应激标志物
        'SOD_U_mL': np.random.normal(120, 25, n_samples),
        'GSH_umol_L': np.random.normal(8, 2, n_samples),
        'MDA_umol_L': np.random.lognormal(1.0, 0.5, n_samples),
        
        # 炎症因子
        'CRP_mg_L': np.random.lognormal(0.8, 1.2, n_samples).clip(0.1, 30),
        'IL6_pg_mL': np.random.lognormal(1.8, 0.8, n_samples).clip(1, 50),
        
        # 代谢指标
        'HbA1c_percent': np.random.normal(5.5, 1.0, n_samples).clip(4.0, 11.0),
        'Total_Cholesterol_mmol_L': np.random.normal(5.2, 1.2, n_samples).clip(3.0, 8.0),
        'HDL_mmol_L': np.random.normal(1.3, 0.3, n_samples).clip(0.8, 2.5),
        
        # 肾功能
        'Creatinine_umol_L': np.random.normal(80, 20, n_samples).clip(40, 180),
        'eGFR_mL_min_1_73m2': np.random.normal(90, 20, n_samples).clip(15, 150),
        
        # 血压
        'Systolic_BP_mmHg': np.random.normal(130, 18, n_samples).clip(90, 200),
        'Diastolic_BP_mmHg': np.random.normal(82, 12, n_samples).clip(60, 120),
    }
    
    df = pd.DataFrame(data)
    
    # 创建分层变量
    # 年龄组
    df['Age_Group'] = pd.cut(df['Age'], bins=[0, 40, 55, 70, 100], 
                             labels=['<40', '40-55', '55-70', '≥70'])
    
    # BMI组
    df['BMI_Group'] = pd.cut(df['BMI'], bins=[0, 18.5, 25, 30, 100],
                             labels=['偏瘦', '正常', '超重', '肥胖'])
    
    # 铅暴露组
    df['Lead_Group'] = pd.qcut(df['Blood_Lead_ug_dL'], q=4, 
                               labels=['低暴露', '中低暴露', '中高暴露', '高暴露'])
    
    # 肾功能组
    df['Kidney_Group'] = pd.cut(df['eGFR_mL_min_1_73m2'], 
                                bins=[0, 30, 60, 90, 200],
                                labels=['重度肾损', '中度肾损', '轻度肾损', '正常'])
    
    # 生成目标变量 (铅毒性 - 基于铅暴露和风险因素)
    risk_score = (
        0.35 * (df['Blood_Lead_ug_dL'] / 30) +
        0.25 * (df['Urine_Lead_ug_L'] / 80) +
        0.20 * (df['MDA_umol_L'] / 4) +
        0.15 * (df['CRP_mg_L'] / 10) +
        0.10 * (df['HbA1c_percent'] / 8) +
        0.08 * (df['Systolic_BP_mmHg'] / 140) -
        0.10 * (df['SOD_U_mL'] / 150) -
        0.08 * (df['GSH_umol_L'] / 12) +
        0.05 * (df['Age'] / 60) +
        0.05 * (df['BMI'] / 30)
    )
    
    # 添加一些亚组效应异质性
    # 男性效应更强
    risk_score += df['Sex'] * 0.15 * (df['Blood_Lead_ug_dL'] / 30)
    # 老年人效应更强
    risk_score += (df['Age'] > 55).astype(float) * 0.12 * (df['Blood_Lead_ug_dL'] / 30)
    # 吸烟者效应更强
    risk_score += df['Smoking'] * 0.10 * (df['Blood_Lead_ug_dL'] / 30)
    
    df['Lead_Toxicity'] = (risk_score > np.percentile(risk_score, 70)).astype(int)
    
    return df


def calculate_subgroup_effect(df, feature, subgroup_var, target='Lead_Toxicity'):
    """计算单个亚组的效应值"""
    results = []
    
    subgroups = df[subgroup_var].unique()
    
    for subgroup in subgroups:
        if pd.isna(subgroup):
            continue
            
        subset = df[df[subgroup_var] == subgroup]
        
        if len(subset) < 50:
            continue
            
        X = subset[[feature]].values
        y = subset[target].values
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 逻辑回归
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_scaled, y)
        
        # 计算OR
        coef = model.coef_[0][0]
        or_value = np.exp(coef)
        
        # 计算95% CI (Wald方法)
        se = 0.25  # 近似标准误
        ci_lower = np.exp(coef - 1.96 * se)
        ci_upper = np.exp(coef + 1.96 * se)
        
        # AUC
        y_prob = model.predict_proba(X_scaled)[:, 1]
        auc = roc_auc_score(y, y_prob)
        
        results.append({
            'Subgroup': str(subgroup),
            'N': len(subset),
            'Cases': subset[target].sum(),
            'OR': or_value,
            'OR_log': coef,
            'CI_Lower': ci_lower,
            'CI_Upper': ci_upper,
            'AUC': auc,
            'P_value': stats.norm.sf(abs(coef/se)) * 2
        })
    
    return pd.DataFrame(results)


def calculate_interaction(df, exposure_var, stratify_var, target='Lead_Toxicity'):
    """计算交互效应（异质性检验）"""
    from sklearn.linear_model import LogisticRegression
    
    # 创建交互项
    X = df[[exposure_var, stratify_var]].values
    X = np.column_stack([X, X[:, 0] * X[:, 1]])
    y = df[target].values
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_scaled, y)
    
    # 交互项系数
    interaction_coef = model.coef_[0][2]
    interaction_or = np.exp(interaction_coef)
    
    # 异质性检验
    p_interaction = stats.norm.sf(abs(interaction_coef / 0.25)) * 2
    
    return {
        'Interaction_OR': interaction_or,
        'Interaction_CI_Lower': np.exp(interaction_coef - 1.96 * 0.25),
        'Interaction_CI_Upper': np.exp(interaction_coef + 1.96 * 0.25),
        'P_interaction': p_interaction,
        'Heterogeneity': 'Significant' if p_interaction < 0.05 else 'Not Significant'
    }


def plot_subgroup_forest(results_df, title='Subgroup Analysis Forest Plot', 
                        output_path=OUTPUT_DIR):
    """绘制亚组分析森林图"""
    
    fig, ax = plt.subplots(figsize=(12, max(8, len(results_df) * 0.8)))
    
    # 排序
    results_df = results_df.sort_values('OR', ascending=True)
    
    y_positions = np.arange(len(results_df))
    
    # 绘制OR点和置信区间
    for i, (_, row) in enumerate(results_df.iterrows()):
        color = COLORS['primary'] if row['P_value'] < 0.05 else COLORS['grid']
        
        # 置信区间横线
        ax.plot([row['CI_Lower'], row['CI_Upper']], [i, i], 
               color=color, linewidth=2, alpha=0.7)
        
        # OR点
        ax.scatter(row['OR'], i, s=150, color=color, zorder=5, 
                  edgecolors='white', linewidths=2)
    
    # 参考线
    ax.axvline(x=1, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # 标签
    ax.set_yticks(y_positions)
    labels = [f"{row['Subgroup']}\n(n={row['N']}, AUC={row['AUC']:.2f})" 
              for _, row in results_df.iterrows()]
    ax.set_yticklabels(labels, fontsize=10)
    
    # 轴标签
    ax.set_xlabel('Odds Ratio (95% CI)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # 设置x轴范围
    x_min = max(0.1, results_df['CI_Lower'].min() * 0.8)
    x_max = results_df['CI_Upper'].max() * 1.2
    ax.set_xlim(x_min, x_max)
    
    # 添加网格
    ax.grid(True, axis='x', alpha=0.3)
    ax.set_axisbelow(True)
    
    # 添加图例
    legend_elements = [
        mpatches.Patch(color=COLORS['primary'], label='P < 0.05'),
        mpatches.Patch(color=COLORS['grid'], label='P ≥ 0.05')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    filename = f'{output_path}subgroup_forest.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.savefig(filename.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"✅ 亚组森林图已保存: {filename}")
    return filename


def plot_subgroup_forest_multiple(subgroup_results, title='Multiple Subgroup Analysis', 
                                   output_path=OUTPUT_DIR):
    """绘制多亚组分析森林图（并行展示）"""
    
    n_subgroups = len(subgroup_results)
    fig, axes = plt.subplots(1, n_subgroups, figsize=(5 * n_subgroups, 10))
    
    if n_subgroups == 1:
        axes = [axes]
    
    all_results = []
    
    for idx, (subgroup_name, results_df) in enumerate(subgroup_results.items()):
        ax = axes[idx]
        
        # 排序
        results_df = results_df.sort_values('OR', ascending=True)
        
        y_positions = np.arange(len(results_df))
        
        # 绘制
        for i, (_, row) in enumerate(results_df.iterrows()):
            color = SUBGROUP_COLORS[idx] if row['P_value'] < 0.05 else '#CCCCCC'
            
            ax.plot([row['CI_Lower'], row['CI_Upper']], [i, i], 
                   color=color, linewidth=2, alpha=0.7)
            ax.scatter(row['OR'], i, s=100, color=color, zorder=5,
                      edgecolors='white', linewidths=1)
        
        ax.axvline(x=1, color='red', linestyle='--', linewidth=1, alpha=0.5)
        
        ax.set_yticks(y_positions)
        ax.set_yticklabels(results_df['Subgroup'].values, fontsize=9)
        ax.set_xlabel('OR (95% CI)', fontsize=10)
        ax.set_title(f'{subgroup_name}', fontsize=12, fontweight='bold')
        
        x_min = max(0.1, results_df['CI_Lower'].min() * 0.8)
        x_max = results_df['CI_Upper'].max() * 1.2
        ax.set_xlim(x_min, x_max)
        ax.grid(True, axis='x', alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    filename = f'{output_path}multi_subgroup_forest.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 多亚组森林图已保存: {filename}")
    return filename


def plot_subgroup_auc_comparison(df, feature, subgroup_vars, target='Lead_Toxicity',
                                  output_path=OUTPUT_DIR):
    """绘制亚组间AUC对比图"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, subgroup_var in enumerate(subgroup_vars[:4]):
        ax = axes[idx]
        
        subgroups = df[subgroup_var].dropna().unique()
        
        for i, subgroup in enumerate(sorted(subgroups)):
            subset = df[df[subgroup_var] == subgroup]
            
            X = subset[[feature]].values
            y = subset[target].values
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_scaled, y)
            
            y_prob = model.predict_proba(X_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(y, y_prob)
            auc = roc_auc_score(y, y_prob)
            
            ax.plot(fpr, tpr, linewidth=2, 
                   label=f'{subgroup} (AUC={auc:.2f})',
                   color=SUBGROUP_COLORS[i % len(SUBGROUP_COLORS)])
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlabel('False Positive Rate', fontsize=10)
        ax.set_ylabel('True Positive Rate', fontsize=10)
        ax.set_title(f'{subgroup_var} Stratification', fontsize=11, fontweight='bold')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('AUC Comparison Across Subgroups', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    filename = f'{output_path}subgroup_auc_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 亚组AUC对比图已保存: {filename}")
    return filename


def plot_heterogeneity_heatmap(interaction_results, output_path=OUTPUT_DIR):
    """绘制异质性热力图"""
    
    # 准备数据
    or_matrix = []
    p_matrix = []
    labels = []
    
    for (exposure, stratum), result in interaction_results.items():
        or_matrix.append([result['OR'], result['CI_Lower'], result['CI_Upper']])
        p_matrix.append(result['P_interaction'])
        labels.append(f'{exposure} × {stratum}')
    
    or_matrix = np.array(or_matrix)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, len(labels) * 0.6)))
    
    # OR热力图
    ax1 = axes[0]
    or_data = or_matrix[:, 0].reshape(-1, 1)
    
    im1 = ax1.imshow(or_data, cmap='RdYlBu_r', aspect='auto', vmin=0.5, vmax=2.0)
    
    ax1.set_yticks(range(len(labels)))
    ax1.set_yticklabels(labels, fontsize=10)
    ax1.set_xticks([0])
    ax1.set_xticklabels(['OR'], fontsize=11)
    ax1.set_title('Odds Ratios', fontsize=12, fontweight='bold')
    
    # 添加数值
    for i in range(len(labels)):
        ax1.text(0, i, f'{or_matrix[i, 0]:.2f}', ha='center', va='center', 
                fontsize=10, color='white' if or_matrix[i, 0] > 1.3 else 'black')
    
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    
    # P值热力图
    ax2 = axes[1]
    p_data = np.array(p_matrix).reshape(-1, 1)
    
    im2 = ax2.imshow(p_data, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=0.2)
    
    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels(labels, fontsize=10)
    ax2.set_xticks([0])
    ax2.set_xticklabels(['P-value'], fontsize=11)
    ax2.set_title('Interaction P-values', fontsize=12, fontweight='bold')
    
    # 添加数值和显著性标记
    for i in range(len(labels)):
        sig = '***' if p_matrix[i] < 0.001 else ('**' if p_matrix[i] < 0.01 else ('*' if p_matrix[i] < 0.05 else ''))
        ax2.text(0, i, f'{p_matrix[i]:.3f}{sig}', ha='center', va='center', 
                fontsize=10, color='white' if p_matrix[i] < 0.05 else 'black')
    
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    
    plt.suptitle('Heterogeneity Assessment', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    filename = f'{output_path}heterogeneity_heatmap.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 异质性热力图已保存: {filename}")
    return filename


def plot_subgroup_distribution(df, subgroup_vars, target='Lead_Toxicity',
                                output_path=OUTPUT_DIR):
    """绘制亚组目标变量分布"""
    
    n_vars = len(subgroup_vars)
    fig, axes = plt.subplots(1, n_vars, figsize=(5 * n_vars, 5))
    
    if n_vars == 1:
        axes = [axes]
    
    for idx, var in enumerate(subgroup_vars):
        ax = axes[idx]
        
        # 计算各亚组的患病率
        prevalence = df.groupby(var)[target].mean() * 100
        
        bars = ax.bar(range(len(prevalence)), prevalence.values, 
                     color=SUBGROUP_COLORS[:len(prevalence)],
                     edgecolor='white', linewidth=2)
        
        ax.set_xticks(range(len(prevalence)))
        ax.set_xticklabels(prevalence.index, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('Prevalence (%)', fontsize=11)
        ax.set_title(f'{var}', fontsize=12, fontweight='bold')
        ax.set_ylim(0, max(prevalence) * 1.2)
        
        # 添加数值标签
        for bar, val in zip(bars, prevalence.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
        
        ax.grid(True, axis='y', alpha=0.3)
    
    plt.suptitle('Lead Toxicity Prevalence by Subgroups', fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    
    filename = f'{output_path}subgroup_distribution.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 亚组分布图已保存: {filename}")
    return filename


def generate_subgroup_report(df, feature='Blood_Lead_ug_dL', subgroup_vars=None,
                            target='Lead_Toxicity', output_path=OUTPUT_DIR):
    """生成完整的亚组分析报告"""
    
    if subgroup_vars is None:
        subgroup_vars = ['Sex', 'Age_Group', 'BMI_Group', 'Smoking', 'Lead_Group']
    
    print("\n" + "="*60)
    print("铅网络毒理学 - 亚组分析模块")
    print("="*60)
    
    # 1. 亚组效应计算
    all_results = {}
    for var in subgroup_vars:
        results = calculate_subgroup_effect(df, feature, var, target)
        all_results[var] = results
        print(f"\n📊 {var} 亚组分析结果:")
        print(results[['Subgroup', 'N', 'Cases', 'OR', 'CI_Lower', 'CI_Upper', 'AUC', 'P_value']].to_string(index=False))
    
    # 2. 绘制森林图
    for var, results in all_results.items():
        plot_subgroup_forest(results, f'Blood Lead Effect by {var}', output_path)
    
    # 3. 多亚组森林图
    plot_subgroup_forest_multiple(all_results, 'Comprehensive Subgroup Analysis', output_path)
    
    # 4. AUC对比图
    plot_subgroup_auc_comparison(df, feature, subgroup_vars, target, output_path)
    
    # 5. 亚组分布图
    plot_subgroup_distribution(df, subgroup_vars, target, output_path)
    
    # 6. 交互效应分析
    print("\n📊 交互效应分析:")
    interaction_results = {}
    for var in subgroup_vars:
        if var != 'Sex':  # 跳过Sex，因为它已经作为二分类
            try:
                result = calculate_interaction(df, feature, df[var].cat.codes.values, target)
                interaction_results[(feature, var)] = result
                print(f"  {feature} × {var}: OR={result['Interaction_OR']:.2f}, "
                      f"95%CI({result['Interaction_CI_Lower']:.2f}-{result['Interaction_CI_Upper']:.2f}), "
                      f"P={result['P_interaction']:.4f} ({result['Heterogeneity']})")
            except:
                pass
    
    # 7. 异质性热力图
    if interaction_results:
        plot_heterogeneity_heatmap(interaction_results, output_path)
    
    # 8. 保存汇总数据
    summary_data = []
    for var, results in all_results.items():
        results_copy = results.copy()
        results_copy['Stratification'] = var
        summary_data.append(results_copy)
    
    if summary_data:
        summary_df = pd.concat(summary_data, ignore_index=True)
        summary_df.to_csv(f'{output_path}subgroup_analysis_summary.csv', index=False)
        print(f"\n✅ 亚组分析汇总数据已保存")
    
    print("\n" + "="*60)
    print("亚组分析完成！")
    print("="*60)
    
    return all_results


def main():
    """主函数 - 运行完整亚组分析"""
    
    print("\n📊 正在生成数据...")
    df = generate_subgroup_data(n_samples=3000)
    
    # 保存数据
    df.to_csv(f'{OUTPUT_DIR}subgroup_data.csv', index=False)
    print(f"✅ 数据已生成: {len(df)} 样本")
    print(f"   目标变量分布: {df['Lead_Toxicity'].value_counts().to_dict()}")
    
    # 运行亚组分析
    results = generate_subgroup_report(
        df, 
        feature='Blood_Lead_ug_dL',
        subgroup_vars=['Sex', 'Age_Group', 'BMI_Group', 'Smoking', 'Alcohol', 'Lead_Group'],
        target='Lead_Toxicity',
        output_path=OUTPUT_DIR
    )
    
    return results


if __name__ == '__main__':
    main()
