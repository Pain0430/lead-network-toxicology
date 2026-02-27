#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分层分析模块 - 按人群特征进行铅暴露风险分层分析
Stratification Analysis Module for Lead Exposure Risk

功能：
1. 按年龄分层分析
2. 按性别分层分析  
3. 按种族/民族分层分析
4. 按社会经济地位(SES)分层分析
5. 分层后的剂量-效应关系分析
6. 分层间效应异质性检验

作者: Pain's AI Assistant
日期: 2026-02-23
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, kruskal
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 配置
# ============================================================================

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 11

# 年龄分组定义 (岁)
AGE_GROUPS = {
    'child': (0, 12, '儿童 (0-12岁)'),
    'adolescent': (13, 19, '青少年 (13-19岁)'),
    'young_adult': (20, 39, '青年 (20-39岁)'),
    'middle_adult': (40, 59, '中年 (40-59岁)'),
    'elderly': (60, 100, '老年 (≥60岁)'),
}

# 铅血阈值 (μg/dL)
LEAD_THRESHOLDS = {
    'low': 3.5,
    'medium': 5.0,
    'high': 10.0,
    'very_high': 20.0,
}


def load_nhanes_data(data_dir="nhanes_data"):
    """加载NHANES数据"""
    files = {
        'blood': 'nhanes_lead_blood.csv',
        'demo': os.path.join(data_dir, 'DEMO_L.xpt'),
        'bmx': os.path.join(data_dir, 'BMX_L.xpt'),
    }
    
    # 加载血液铅数据
    blood_file = os.path.join(data_dir, 'nhanes_lead_blood.csv')
    if os.path.exists(blood_file):
        df = pd.read_csv(blood_file)
        return df
    
    return None


def categorize_age(age):
    """将年龄转换为分类变量"""
    if age < 12:
        return 'child'
    elif age < 20:
        return 'adolescent'
    elif age < 40:
        return 'young_adult'
    elif age < 60:
        return 'middle_adult'
    else:
        return 'elderly'


def categorize_lead(level):
    """将血铅水平转换为分类变量"""
    if pd.isna(level):
        return np.nan
    elif level < LEAD_THRESHOLDS['low']:
        return 'low'
    elif level < LEAD_THRESHOLDS['medium']:
        return 'medium'
    elif level < LEAD_THRESHOLDS['high']:
        return 'high'
    else:
        return 'very_high'


def stratify_by_age(df, age_col='RIDAGEYR'):
    """按年龄分层"""
    df = df.copy()
    df['age_group'] = df[age_col].apply(categorize_age)
    return df


def stratify_by_gender(df, gender_col='RIAGENDR'):
    """按性别分层"""
    df = df.copy()
    gender_map = {1: 'male', 2: 'female'}
    df['gender'] = df[gender_col].map(gender_map)
    return df


def stratify_by_race(df, race_col='RIDRETH3'):
    """按种族/民族分层"""
    df = df.copy()
    race_map = {
        1: 'Mexican American',
        2: 'Other Hispanic',
        3: 'Non-Hispanic White',
        4: 'Non-Hispanic Black',
        5: 'Non-Hispanic Asian',
        6: 'Other/Multiracial',
    }
    df['race_ethnicity'] = df[race_col].map(race_map)
    return df


def stratify_by_ses(df, income_col='INDHHIN2', education_col='DMDEDUC2'):
    """按社会经济地位分层 (收入+教育)"""
    df = df.copy()
    
    # 收入分层
    def categorize_income(x):
        if pd.isna(x):
            return np.nan
        elif x in [1, 2, 3, 4]:  # <$25,000
            return 'low'
        elif x in [5, 6, 7]:  # $25,000-$75,000
            return 'middle'
        else:  # >$75,000
            return 'high'
    
    # 教育分层
    def categorize_education(x):
        if pd.isna(x):
            return np.nan
        elif x in [1, 2]:  # 高中及以下
            return 'low'
        elif x == 3:  # 大学
            return 'middle'
        else:  # 研究生
            return 'high'
    
    df['income_level'] = df[income_col].apply(categorize_income)
    df['education_level'] = df[education_col].apply(categorize_education)
    
    # 综合SES
    def combined_ses(row):
        if pd.isna(row['income_level']) or pd.isna(row['education_level']):
            return np.nan
        score = 0
        if row['income_level'] == 'low':
            score += 1
        elif row['income_level'] == 'high':
            score += 3
        else:
            score += 2
        
        if row['education_level'] == 'low':
            score += 1
        elif row['education_level'] == 'high':
            score += 3
        else:
            score += 2
        
        if score <= 3:
            return 'low'
        elif score <= 5:
            return 'middle'
        else:
            return 'high'
    
    df['ses_level'] = df.apply(combined_ses, axis=1)
    return df


def stratified_correlation_analysis(df, stratify_col, target_cols, method='spearman'):
    """
    分层相关性分析
    
    Args:
        df: 数据框
        stratify_col: 分层列名
        target_cols: 目标列列表
        method: 相关方法
    
    Returns:
        dict: 分层相关性结果
    """
    results = {}
    groups = df[stratify_col].dropna().unique()
    
    for group in sorted(groups):
        subset = df[df[stratify_col] == group]
        group_results = {}
        
        for col in target_cols:
            if col in subset.columns:
                # 计算相关性
                mask = subset[['LBXBPB', col]].notna().all(axis=1)
                if mask.sum() >= 10:
                    x = subset.loc[mask, 'LBXBPB'].values
                    y = subset.loc[mask, col].values
                    
                    if method == 'spearman':
                        r, p = stats.spearmanr(x, y)
                    else:
                        r, p = stats.pearsonr(x, y)
                    
                    group_results[col] = {
                        'r': r,
                        'p': p,
                        'n': mask.sum(),
                        'significant': p < 0.05
                    }
        
        results[group] = group_results
    
    return results


def heterogeneity_test(df, stratify_col, outcome_col):
    """
    分层间效应异质性检验
    
    使用Cochran's Q检验或Kruskal-Wallis检验
    """
    groups = df[stratify_col].dropna().unique()
    group_effects = []
    group_ns = []
    
    for group in groups:
        subset = df[df[stratify_col] == group]
        mask = subset[['LBXBPB', outcome_col]].notna().all(axis=1)
        
        if mask.sum() >= 10:
            r, _ = stats.spearmanr(
                subset.loc[mask, 'LBXBPB'],
                subset.loc[mask, outcome_col]
            )
            # Fisher's z transformation
            if abs(r) < 1:
                z = 0.5 * np.log((1 + r) / (1 - r))
                group_effects.append(z)
                group_ns.append(mask.sum())
    
    if len(group_effects) < 2:
        return None
    
    # 异质性检验 (简化版: Kruskal-Wallis)
    stat, p = kruskal(*[df[df[stratify_col] == g][['LBXBPB', outcome_col]].dropna()[outcome_col].values 
                        for g in groups if df[df[stratify_col] == g][['LBXBPB', outcome_col]].dropna().shape[0] >= 10])
    
    return {
        'statistic': stat,
        'p_value': p,
        'heterogeneous': p < 0.05
    }


def calculate_stratified_risk(df, stratify_col, lead_col='LBXBPB', outcome_col='high_risk'):
    """
    计算分层后的风险比
    
    Args:
        df: 数据框
        stratify_col: 分层变量
        lead_col: 铅暴露列
        outcome_col: 结局列
    
    Returns:
        DataFrame: 分层风险比
    """
    results = []
    groups = df[stratify_col].dropna().unique()
    
    for group in sorted(groups):
        subset = df[df[stratify_col] == group]
        
        # 高铅 vs 低铅
        high_lead = subset[subset[lead_col] >= 5.0][outcome_col]
        low_lead = subset[subset[lead_col] < 3.5][outcome_col]
        
        if len(high_lead) >= 5 and len(low_lead) >= 5:
            high_risk = high_lead.mean()
            low_risk = low_lead.mean()
            
            # 风险比
            if low_risk > 0:
                rr = high_risk / low_risk
            else:
                rr = np.nan
            
            results.append({
                'stratum': group,
                'n_total': len(subset),
                'n_high_lead': len(high_lead),
                'n_low_lead': len(low_lead),
                'risk_high_lead': high_risk,
                'risk_low_lead': low_risk,
                'risk_ratio': rr
            })
    
    return pd.DataFrame(results)


def dose_response_stratified(df, stratify_col, lead_col='LBXBPB', outcome_col='LBXGH'):
    """
    分层剂量-反应分析
    
    Returns:
        dict: 每层的剂量-反应斜率
    """
    results = {}
    groups = df[stratify_col].dropna().unique()
    
    for group in sorted(groups):
        subset = df[df[stratify_col] == group]
        mask = subset[[lead_col, outcome_col]].notna().all(axis=1)
        
        if mask.sum() >= 20:
            x = subset.loc[mask, lead_col].values
            y = subset.loc[mask, outcome_col].values
            
            # Spearman相关作为剂量-反应指标
            r, p = stats.spearmanr(x, y)
            
            # 线性回归
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            results[group] = {
                'spearman_r': r,
                'spearman_p': p,
                'slope': slope,
                'r_squared': r_value ** 2,
                'n': mask.sum()
            }
    
    return results


def plot_stratified_analysis(df, stratify_col, lead_col='LBXBPB', outcome_col='LBXGH'):
    """绘制分层分析图表"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. 各层铅水平分布
    ax1 = axes[0, 0]
    groups = sorted(df[stratify_col].dropna().unique())
    box_data = [df[df[stratify_col] == g][lead_col].dropna() for g in groups]
    bp = ax1.boxplot(box_data, labels=groups, patch_artist=True)
    colors = plt.cm.Set2(np.linspace(0, 1, len(groups)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax1.set_xlabel(stratify_col)
    ax1.set_ylabel('Blood Lead (μg/dL)')
    ax1.set_title('Lead Levels by Stratum')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. 各层剂量-反应关系
    ax2 = axes[0, 1]
    for i, group in enumerate(groups):
        subset = df[df[stratify_col] == group]
        mask = subset[[lead_col, outcome_col]].notna().all(axis=1)
        if mask.sum() > 10:
            ax2.scatter(subset.loc[mask, lead_col], 
                       subset.loc[mask, outcome_col], 
                       alpha=0.3, label=group, s=20)
    
    ax2.set_xlabel('Blood Lead (μg/dL)')
    ax2.set_ylabel(outcome_col)
    ax2.set_title('Dose-Response by Stratum')
    ax2.legend()
    
    # 3. 各层效应量比较
    ax3 = axes[1, 0]
    effect_data = []
    labels = []
    for group in groups:
        subset = df[df[stratify_col] == group]
        mask = subset[[lead_col, outcome_col]].notna().all(axis=1)
        if mask.sum() > 10:
            r, _ = stats.spearmanr(subset.loc[mask, lead_col], 
                                   subset.loc[mask, outcome_col])
            effect_data.append(r)
            labels.append(group)
    
    bars = ax3.bar(labels, effect_data, color=plt.cm.Set2(np.linspace(0, 1, len(effect_data))))
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax3.set_xlabel(stratify_col)
    ax3.set_ylabel("Spearman's r")
    ax3.set_title('Effect Size by Stratum')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. 各层样本量
    ax4 = axes[1, 1]
    ns = [len(df[df[stratify_col] == g]) for g in groups]
    ax4.bar(labels, ns, color=plt.cm.Set3(np.linspace(0, 1, len(ns))))
    ax4.set_xlabel(stratify_col)
    ax4.set_ylabel('Sample Size')
    ax4.set_title('Sample Size by Stratum')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'stratified_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"分层分析图表已保存至: {OUTPUT_DIR}/stratified_analysis.png")


def generate_stratification_report(df):
    """生成分层分析报告"""
    report = []
    report.append("=" * 60)
    report.append("铅暴露分层分析报告")
    report.append("Stratified Lead Exposure Analysis Report")
    report.append("=" * 60)
    report.append("")
    
    # 数据概览
    report.append("## 1. 数据概览")
    report.append(f"总样本量: {len(df)}")
    if 'LBXBPB' in df.columns:
        valid_lead = df['LBXBPB'].dropna()
        report.append(f"有效血铅数据: {len(valid_lead)}")
        report.append(f"血铅均值: {valid_lead.mean():.2f} μg/dL")
        report.append(f"血铅中位数: {valid_lead.median():.2f} μg/dL")
    report.append("")
    
    # 年龄分层分析
    if 'age_group' in df.columns:
        report.append("## 2. 年龄分层分析")
        groups = sorted(df['age_group'].dropna().unique())
        for group in groups:
            subset = df[df['age_group'] == group]
            n = len(subset)
            lead_mean = subset['LBXBPB'].mean()
            report.append(f"- {group}: n={n}, 平均血铅={lead_mean:.2f} μg/dL")
        report.append("")
    
    # 性别分层分析
    if 'gender' in df.columns:
        report.append("## 3. 性别分层分析")
        groups = sorted(df['gender'].dropna().unique())
        for group in groups:
            subset = df[df['gender'] == group]
            n = len(subset)
            lead_mean = subset['LBXBPB'].mean()
            report.append(f"- {group}: n={n}, 平均血铅={lead_mean:.2f} μg/dL")
        report.append("")
    
    # 种族分层分析
    if 'race_ethnicity' in df.columns:
        report.append("## 4. 种族/民族分层分析")
        groups = sorted(df['race_ethnicity'].dropna().unique())
        for group in groups:
            subset = df[df['race_ethnicity'] == group]
            n = len(subset)
            lead_mean = subset['LBXBPB'].mean()
            report.append(f"- {group}: n={n}, 平均血铅={lead_mean:.2f} μg/dL")
        report.append("")
    
    # SES分层分析
    if 'ses_level' in df.columns:
        report.append("## 5. 社会经济地位分层分析")
        groups = sorted(df['ses_level'].dropna().unique())
        for group in groups:
            subset = df[df['ses_level'] == group]
            n = len(subset)
            lead_mean = subset['LBXBPB'].mean()
            report.append(f"- {group}: n={n}, 平均血铅={lead_mean:.2f} μg/dL")
        report.append("")
    
    report.append("=" * 60)
    report.append("报告生成完成")
    
    return "\n".join(report)


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数: 运行分层分析"""
    print("开始分层分析...")
    
    # 加载数据
    df = load_nhanes_data()
    
    if df is None:
        print("未找到NHANES数据，创建模拟数据演示...")
        np.random.seed(42)
        n = 500
        
        # 创建模拟数据
        df = pd.DataFrame({
            'LBXBPB': np.random.lognormal(0.5, 0.5, n),  # 血铅
            'LBXGH': np.random.normal(5.5, 1.2, n),  # 糖化血红蛋白
            'LBXSATSI': np.random.normal(25, 5, n),  # ALT
            'RIDAGEYR': np.random.randint(5, 80, n),  # 年龄
            'RIAGENDR': np.random.choice([1, 2], n),  # 性别
            'RIDRETH3': np.random.choice([1, 2, 3, 4, 5], n),  # 种族
            'INDHHIN2': np.random.choice([1, 2, 3, 4, 5, 6, 7, 14, 15], n),  # 收入
            'DMDEDUC2': np.random.choice([1, 2, 3, 4, 5], n),  # 教育
        })
    
    # 分层
    df = stratify_by_age(df)
    df = stratify_by_gender(df)
    df = stratify_by_race(df)
    df = stratify_by_ses(df)
    
    print(f"数据加载完成: {len(df)} 条记录")
    print(f"年龄分组: {df['age_group'].value_counts().to_dict()}")
    print(f"性别分组: {df['gender'].value_counts().to_dict()}")
    
    # 执行分层相关性分析
    if 'LBXGH' in df.columns:
        outcome_cols = ['LBXGH', 'LBXSATSI']
        print("\n执行按年龄分层的相关性分析...")
        
        age_results = stratified_correlation_analysis(df, 'age_group', outcome_cols)
        
        print("\n年龄分层相关性结果:")
        for age_group, results in age_results.items():
            print(f"\n{age_group}:")
            for outcome, stats_dict in results.items():
                sig = "***" if stats_dict['significant'] else ""
                print(f"  {outcome}: r={stats_dict['r']:.3f}, p={stats_dict['p']:.4f}, n={stats_dict['n']} {sig}")
    
    # 剂量-反应分层分析
    if 'LBXBPB' in df.columns and 'LBXGH' in df.columns:
        print("\n执行剂量-反应分层分析...")
        dose_results = dose_response_stratified(df, 'age_group')
        
        print("\n年龄分层剂量-反应结果:")
        for group, results in dose_results.items():
            sig = "***" if results['spearman_p'] < 0.001 else "**" if results['spearman_p'] < 0.01 else "*" if results['spearman_p'] < 0.05 else ""
            print(f"  {group}: r={results['spearman_r']:.3f}, slope={results['slope']:.4f}, n={results['n']} {sig}")
    
    # 绘制分层分析图表
    print("\n生成分层分析图表...")
    plot_stratified_analysis(df, 'age_group')
    
    # 生成报告
    report = generate_stratification_report(df)
    print("\n" + report)
    
    # 保存报告
    with open(os.path.join(OUTPUT_DIR, 'stratification_report.md'), 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n分析完成! 报告已保存至: {OUTPUT_DIR}/stratification_report.md")


if __name__ == "__main__":
    main()
