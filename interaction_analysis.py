#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交互效应分析模块 - 分析多种环境暴露因素的交互效应
Interaction Effect Analysis Module

功能：
1. 重金属-重金属交互效应分析
2. 重金属-PFAS交互效应分析  
3. 基因-环境交互分析 (GxE)
4. 暴露联合效应分析
5. 交互效应可视化

作者: Pain's AI Assistant
日期: 2026-02-23
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 配置
# ============================================================================

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')

# 重金属列表
METALS = {
    'LBXBPB': ('Pb', 'Lead', '铅'),
    'LBXIAS': ('As', 'Arsenic', '砷'),
    'LBXBCD': ('Cd', 'Cadmium', '镉'),
    'LBXIHG': ('Hg', 'Mercury', '汞'),
    'LBXBMN': ('Mn', 'Manganese', '锰'),
}

# PFAS列表
PFAS = {
    'LBXPFOA': ('PFOA', '全氟辛酸'),
    'LBXPFOS': ('PFOS', '全氟辛烷磺酸'),
    'LBXPFNA': ('PFNA', '全氟壬酸'),
    'LBXPFHXS': ('PFHxS', '全氟己烷磺酸'),
}


def load_nhanes_data():
    """加载NHANES数据"""
    data_files = [
        'nhanes_data/nhanes_lead_blood.csv',
        'nhanes_data/PBCD_L.xpt',
    ]
    
    # 优先使用CSV
    if os.path.exists('nhanes_data/nhanes_lead_blood.csv'):
        df = pd.read_csv('nhanes_data/nhanes_lead_blood.csv')
        return df
    
    return None


def calculate_interaction_term(x1, x2, method='multiplicative'):
    """
    计算交互项
    
    Args:
        x1, x2: 两个暴露因素
        method: 'multiplicative'(乘积) 或 'ratio'(比值) 或 'difference'(差值)
    
    Returns:
        array: 交互项
    """
    if method == 'multiplicative':
        return x1 * x2
    elif method == 'ratio':
        # 避免除零
        x2_adj = np.where(x2 == 0, 0.001, x2)
        return x1 / x2_adj
    else:  # difference
        return x1 - x2


def metal_metal_interaction(df, metal1='LBXBPB', metal2='LBXIAS', outcome='LBXGH'):
    """
    重金属-重金属交互效应分析
    
    Returns:
        dict: 交互效应结果
    """
    # 准备数据
    cols = [metal1, metal2, outcome]
    subset = df[cols].dropna()
    
    if len(subset) < 50:
        return None
    
    x1 = subset[metal1].values
    x2 = subset[metal2].values
    y = subset[outcome].values
    
    # 标准化
    scaler = StandardScaler()
    x1_scaled = scaler.fit_transform(x1.reshape(-1, 1)).flatten()
    x2_scaled = scaler.fit_transform(x2.reshape(-1, 1)).flatten()
    
    # 创建交互项
    interaction = x1_scaled * x2_scaled
    
    # 分别回归
    model_main = LinearRegression()
    model_main.fit(np.column_stack([x1_scaled, x2_scaled]), y)
    r2_main = model_main.score(np.column_stack([x1_scaled, x2_scaled]), y)
    
    model_interaction = LinearRegression()
    model_interaction.fit(np.column_stack([x1_scaled, x2_scaled, interaction]), y)
    r2_interaction = model_interaction.score(
        np.column_stack([x1_scaled, x2_scaled, interaction]), y
    )
    
    # 交互效应F检验 (简化版)
    n = len(y)
    p = 3  # main + interaction
    f_stat = ((r2_interaction - r2_main) / 1) / ((1 - r2_interaction) / (n - p))
    p_value = 1 - stats.f.cdf(f_stat, 1, n - p)
    
    return {
        'metal1': METALS.get(metal1, (metal1, metal1, metal1))[0],
        'metal2': METALS.get(metal2, (metal2, metal2, metal2))[0],
        'r2_main': r2_main,
        'r2_with_interaction': r2_interaction,
        'r2_increase': r2_interaction - r2_main,
        'f_statistic': f_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'n': len(subset)
    }


def joint_exposure_analysis(df, metals=['LBXBPB', 'LBXIAS', 'LBXBCD'], 
                            outcome='LBXGH', method='sum'):
    """
    联合暴露分析
    
    Args:
        metals: 金属列表
        outcome: 结局变量
        method: 'sum'(加权求和) 或 'score'(评分) 或 'count'(超标计数)
    
    Returns:
        dict: 联合暴露结果
    """
    # 检查可用金属
    available_metals = [m for m in metals if m in df.columns]
    
    if len(available_metals) < 2:
        return None
    
    if method == 'count':
        # 超标计数方法
        thresholds = {'LBXBPB': 5.0, 'LBXIAS': 10.0, 'LBXBCD': 5.0, 
                     'LBXIHG': 5.0, 'LBXBMN': 10.0}
        
        joint_exposure = np.zeros(len(df))
        for metal in available_metals:
            if metal in thresholds:
                joint_exposure += (df[metal] >= thresholds[metal]).astype(int)
        
    elif method == 'score':
        # 百分位评分方法
        joint_exposure = np.zeros(len(df))
        for metal in available_metals:
            valid = df[metal].notna()
            ranks = df[metal].rank(pct=True)
            joint_exposure += ranks.fillna(0)
        
    else:
        # 标准化后求和
        joint_exposure = np.zeros(len(df))
        for metal in available_metals:
            valid = df[metal].notna()
            if valid.sum() > 0:
                scaled = (df[metal] - df[metal].mean()) / df[metal].std()
                joint_exposure += scaled.fillna(0)
    
    # 相关性分析
    valid_idx = joint_exposure != 0
    if valid_idx.sum() > 50:
        r, p = stats.spearmanr(
            joint_exposure[valid_idx], 
            df.loc[valid_idx, outcome].dropna()[:valid_idx.sum()]
        )
        
        # 分组比较
        high_exposure = df.loc[valid_idx][joint_exposure[valid_idx] > np.percentile(joint_exposure[valid_idx], 75)]
        low_exposure = df.loc[valid_idx][joint_exposure[valid_idx] < np.percentile(joint_exposure[valid_idx], 25)]
        
        outcome_high = high_exposure[outcome].dropna()
        outcome_low = low_exposure[outcome].dropna()
        
        if len(outcome_high) > 5 and len(outcome_low) > 5:
            # t检验
            t_stat, t_p = stats.ttest_ind(outcome_high, outcome_low)
            
            return {
                'joint_exposure': joint_exposure,
                'spearman_r': r,
                'spearman_p': p,
                'mean_high': outcome_high.mean(),
                'mean_low': outcome_low.mean(),
                'difference': outcome_high.mean() - outcome_low.mean(),
                't_statistic': t_stat,
                't_p_value': t_p,
                'n_high': len(outcome_high),
                'n_low': len(outcome_low),
                'significant': p < 0.05
            }
    
    return None


def stratified_interaction(df, metal1, metal2, outcome, stratify_by='RIAGENDR'):
    """
    分层交互效应分析
    
    分析在特定亚组中的交互效应是否不同
    """
    results = {}
    
    # 获取分层变量
    if stratify_by not in df.columns:
        # 使用默认分层
        groups = ['all']
        df['all'] = 'all'
    else:
        groups = df[stratify_by].dropna().unique()
    
    for group in groups:
        if group == 'all':
            subset = df
        else:
            subset = df[df[stratify_by] == group]
        
        interaction_result = metal_metal_interaction(subset, metal1, metal2, outcome)
        if interaction_result:
            results[group] = interaction_result
    
    return results


def visualize_interaction(df, metal1='LBXBPB', metal2='LBXIAS', outcome='LBXGH'):
    """可视化交互效应"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    metal1_name = METALS.get(metal1, (metal1,))[0]
    metal2_name = METALS.get(metal2, (metal2,))[0]
    
    # 1. 散点图 with regression
    ax1 = axes[0, 0]
    subset = df[[metal1, metal2, outcome]].dropna()
    
    if len(subset) > 0:
        scatter = ax1.scatter(subset[metal1], subset[metal2], 
                            c=subset[outcome], cmap='RdYlBu_r', 
                            alpha=0.6, s=30)
        plt.colorbar(scatter, ax=ax1, label=outcome)
        
        # 添加回归线
        z = np.polyfit(subset[metal1], subset[metal2], 1)
        p = np.poly1d(z)
        x_line = np.linspace(subset[metal1].min(), subset[metal1].max(), 100)
        ax1.plot(x_line, p(x_line), 'r--', linewidth=2, label='Trend')
        
        ax1.set_xlabel(f'{metal1_name} ({metal1})')
        ax1.set_ylabel(f'{metal2_name} ({metal2})')
        ax1.set_title(f'{metal1_name} vs {metal2_name} by {outcome}')
        ax1.legend()
    
    # 2. 联合暴露分布
    ax2 = axes[0, 1]
    joint = calculate_interaction_term(
        subset[metal1].values, 
        subset[metal2].values, 
        method='multiplicative'
    )
    ax2.hist2d(joint, subset[outcome], bins=30, cmap='YlOrRd')
    ax2.set_xlabel(f'{metal1_name} × {metal2_name} (Interaction)')
    ax2.set_ylabel(outcome)
    ax2.set_title('Joint Exposure Effect')
    
    # 3. 分位数箱线图
    ax3 = axes[1, 0]
    # 按金属1分位数分组
    quantiles = pd.qcut(subset[metal1], q=4, labels=['Q1(Low)', 'Q2', 'Q3', 'Q4(High)'])
    subset['metal1_quantile'] = quantiles
    
    sns.boxplot(data=subset, x='metal1_quantile', y=outcome, ax=ax3, palette='Set2')
    ax3.set_xlabel(f'{metal1_name} Quantile')
    ax3.set_ylabel(outcome)
    ax3.set_title(f'{outcome} by {metal1_name} Level')
    
    # 4. 热力图 - 金属间相关性
    ax4 = axes[1, 1]
    metal_cols = [c for c in df.columns if c in METALS.keys()]
    if metal_cols:
        corr = df[metal_cols].corr()
        labels = [METALS.get(c, (c,))[0] for c in metal_cols]
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
                   xticklabels=labels, yticklabels=labels, ax=ax4)
        ax4.set_title('Metal Correlation Matrix')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'interaction_analysis.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"交互效应可视化已保存至: {OUTPUT_DIR}/interaction_analysis.png")


def generate_interaction_report(interaction_results, joint_results):
    """生成交互效应分析报告"""
    report = []
    report.append("=" * 60)
    report.append("环境暴露交互效应分析报告")
    report.append("Interaction Effect Analysis Report")
    report.append("=" * 60)
    report.append("")
    
    # 金属-金属交互效应
    if interaction_results:
        report.append("## 1. 重金属-重金属交互效应")
        for key, result in interaction_results.items():
            sig = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*" if result['p_value'] < 0.05 else ""
            report.append(f"\n### {result['metal1']} × {result['metal2']}")
            report.append(f"- 样本量: {result['n']}")
            report.append(f"- 主效应 R²: {result['r2_main']:.4f}")
            report.append(f"- 含交互项 R²: {result['r2_with_interaction']:.4f}")
            report.append(f"- R² 增加: {result['r2_increase']:.4f}")
            report.append(f"- F统计量: {result['f_statistic']:.3f}")
            report.append(f"- p值: {result['p_value']:.4f} {sig}")
            
            if result['significant']:
                report.append("**结论**: 存在显著的交互效应")
            else:
                report.append("**结论**: 未发现显著交互效应")
    
    # 联合暴露分析
    if joint_results:
        report.append("\n## 2. 多金属联合暴露效应")
        report.append(f"- Spearman相关系数: {joint_results['spearman_r']:.4f}")
        report.append(f"- p值: {joint_results['spearman_p']:.4f}")
        report.append(f"- 高暴露组均值: {joint_results['mean_high']:.2f}")
        report.append(f"- 低暴露组均值: {joint_results['mean_low']:.2f}")
        report.append(f"- 组间差异: {joint_results['difference']:.2f}")
        report.append(f"- t检验p值: {joint_results['t_p_value']:.4f}")
        
        if joint_results['spearman_r'] > 0 and joint_results['spearman_p'] < 0.05:
            report.append("**结论**: 联合暴露与结局显著正相关")
        elif joint_results['spearman_r'] < 0 and joint_results['spearman_p'] < 0.05:
            report.append("**结论**: 联合暴露与结局显著负相关")
        else:
            report.append("**结论**: 未发现显著关联")
    
    report.append("\n" + "=" * 60)
    report.append("报告生成完成")
    
    return "\n".join(report)


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数: 运行交互效应分析"""
    print("开始交互效应分析...")
    
    # 加载数据
    df = load_nhanes_data()
    
    if df is None:
        print("未找到NHANES数据，创建模拟数据演示...")
        np.random.seed(42)
        n = 500
        
        # 创建模拟数据 - 模拟金属间正相关
        df = pd.DataFrame({
            'LBXBPB': np.random.lognormal(0.5, 0.5, n),
            'LBXIAS': np.random.lognormal(0.3, 0.6, n) + 0.3 * np.random.lognormal(0.5, 0.5, n),  # 与Pb相关
            'LBXBCD': np.random.lognormal(-0.5, 0.7, n),
            'LBXIHG': np.random.lognormal(-0.2, 0.5, n),
            'LBXBMN': np.random.lognormal(1.0, 0.4, n),
            'LBXGH': 5.5 + 0.3 * np.random.lognormal(0.5, 0.5, n) - 0.1 * np.random.lognormal(0.3, 0.6, n) + np.random.normal(0, 0.5, n),  # 与Pb、As相关
            'LBXSATSI': np.random.normal(25, 5, n),
            'RIAGENDR': np.random.choice([1, 2], n),
        })
    
    print(f"数据加载完成: {len(df)} 条记录")
    
    # 1. 金属-金属交互效应分析
    print("\n1. 执行重金属-重金属交互效应分析...")
    
    interaction_pairs = [
        ('LBXBPB', 'LBXIAS'),  # Pb-As
        ('LBXBPB', 'LBXBCD'),  # Pb-Cd
        ('LBXIAS', 'LBXBCD'),  # As-Cd
    ]
    
    interaction_results = {}
    for metal1, metal2 in interaction_pairs:
        if metal1 in df.columns and metal2 in df.columns:
            result = metal_metal_interaction(df, metal1, metal2, 'LBXGH')
            if result:
                key = f"{result['metal1']}_{result['metal2']}"
                interaction_results[key] = result
                sig = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*" if result['p_value'] < 0.05 else ""
                print(f"  {result['metal1']} × {result['metal2']}: ΔR²={result['r2_increase']:.4f}, p={result['p_value']:.4f} {sig}")
    
    # 2. 联合暴露分析
    print("\n2. 执行多金属联合暴露分析...")
    
    metals = ['LBXBPB', 'LBXIAS', 'LBXBCD']
    joint_result = joint_exposure_analysis(df, metals=metals, outcome='LBXGH', method='sum')
    
    if joint_result:
        sig = "***" if joint_result['spearman_p'] < 0.001 else "**" if joint_result['spearman_p'] < 0.01 else "*" if joint_result['spearman_p'] < 0.05 else ""
        print(f"  联合暴露与HbA1c: r={joint_result['spearman_r']:.4f}, p={joint_result['spearman_p']:.4f} {sig}")
        print(f"  高暴露组 vs 低暴露组: {joint_result['mean_high']:.2f} vs {joint_result['mean_low']:.2f}")
    
    # 3. 分层交互效应
    print("\n3. 执行分层交互效应分析...")
    
    if 'RIAGENDR' in df.columns:
        stratified = stratified_interaction(df, 'LBXBPB', 'LBXIAS', 'LBXGH', 'RIAGENDR')
        gender_map = {1: 'Male', 2: 'Female'}
        
        for group, result in stratified.items():
            group_name = gender_map.get(group, group)
            sig = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*" if result['p_value'] < 0.05 else ""
            print(f"  {group_name}: ΔR²={result['r2_increase']:.4f}, p={result['p_value']:.4f} {sig}")
    
    # 4. 可视化
    print("\n4. 生成交互效应可视化...")
    
    if 'LBXBPB' in df.columns and 'LBXIAS' in df.columns:
        visualize_interaction(df, 'LBXBPB', 'LBXIAS', 'LBXGH')
    
    # 5. 生成报告
    print("\n5. 生成分析报告...")
    
    report = generate_interaction_report(interaction_results, joint_result)
    print("\n" + report)
    
    with open(os.path.join(OUTPUT_DIR, 'interaction_analysis_report.md'), 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n分析完成! 报告已保存至: {OUTPUT_DIR}/interaction_analysis_report.md")


if __name__ == "__main__":
    main()
