#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计分析模块 - 为网络毒理学研究提供统计检验功能
Statistical Analysis Module for Network Toxicology

功能：
1. 相关性分析（含显著性检验）
2. 组间比较（t检验、Mann-Whitney U检验、Kruskal-Wallis检验）
3. 效应量计算（Cohen's d、Cramer's V）
4. 多重比较校正（Bonferroni、FDR）
5. 回归分析显著性
6. 剂量-效应关系分析

作者: Pain's AI Assistant
日期: 2026-02-23
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr, pearsonr, ttest_ind, mannwhitneyu, kruskal
from itertools import combinations
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 配置
# ============================================================================

# 重金属中英文对照
METAL_NAMES = {
    'LBXBPB': ('Pb', '铅'),
    'LBXIAS': ('As', '砷'),
    'LBXBCD': ('Cd', '镉'),
    'LBXIHG': ('Hg', '汞'),
    'LBXBMN': ('Mn', '锰'),
}

# 统计分析结果输出目录
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# 相关性分析
# ============================================================================

def correlation_with_significance(df, col1, col2, method='pearson'):
    """
    计算相关性及显著性
    
    Args:
        df: 数据框
        col1: 第一列
        col2: 第二列
        method: 'pearson' 或 'spearman'
    
    Returns:
        dict: {r, p_value, n, method, significant}
    """
    # 去除缺失值
    mask = df[[col1, col2]].notna().all(axis=1)
    x = df.loc[mask, col1].values
    y = df.loc[mask, col2].values
    
    if len(x) < 3:
        return {'r': np.nan, 'p_value': np.nan, 'n': 0, 'method': method, 'significant': False}
    
    if method == 'pearson':
        r, p = pearsonr(x, y)
    else:
        r, p = spearmanr(x, y)
    
    return {
        'r': r,
        'p_value': p,
        'n': len(x),
        'method': method,
        'significant': p < 0.05,
        'significant_bonf': p < 0.05 / len(x),  # 简化Bonferroni
    }


def correlation_matrix_with_stats(df, columns, method='pearson', alpha=0.05):
    """
    构建相关性矩阵（含统计显著性）
    
    Args:
        df: 数据框
        columns: 列名列表
        method: 'pearson' 或 'spearman'
        alpha: 显著性水平
    
    Returns:
        tuple: (r_matrix, p_matrix, n_matrix)
    """
    n_cols = len(columns)
    r_matrix = np.zeros((n_cols, n_cols))
    p_matrix = np.zeros((n_cols, n_cols))
    n_matrix = np.zeros((n_cols, n_cols))
    
    for i, col1 in enumerate(columns):
        for j, col2 in enumerate(columns):
            if i == j:
                r_matrix[i, j] = 1.0
                p_matrix[i, j] = 0.0
                n_matrix[i, j] = df[col1].notna().sum()
            else:
                result = correlation_with_significance(df, col1, col2, method)
                r_matrix[i, j] = result['r']
                p_matrix[i, j] = result['p_value']
                n_matrix[i, j] = result['n']
    
    r_df = pd.DataFrame(r_matrix, index=columns, columns=columns)
    p_df = pd.DataFrame(p_matrix, index=columns, columns=columns)
    n_df = pd.DataFrame(n_matrix, index=columns, columns=columns)
    
    return r_df, p_df, n_df


def multiple_correlation_correction(p_values, method='bonferroni'):
    """
    多重比较校正
    
    Args:
        p_values: p值列表
        method: 'bonferroni' 或 'fdr' (Benjamini-Hochberg)
    
    Returns:
        dict: 校正后的p值和显著性判断
    """
    n = len(p_values)
    adjusted = []
    
    if method == 'bonferroni':
        # Bonferroni校正
        adjusted = [min(p * n, 1.0) for p in p_values]
    elif method == 'fdr':
        # Benjamini-Hochberg FDR校正
        sorted_p = sorted(enumerate(p_values), key=lambda x: x[1])
        adjusted = [0] * n
        for rank, (idx, p) in enumerate(sorted_p):
            bh_value = p * n / (rank + 1)
            adjusted[idx] = min(bh_value, 1.0)
    
    return {
        'original': p_values,
        'adjusted': adjusted,
        'significant': [p < 0.05 for p in adjusted]
    }


# ============================================================================
# 组间比较
# ============================================================================

def compare_groups(df, group_col, value_col, test='mann-whitney'):
    """
    两组间比较检验
    
    Args:
        df: 数据框
        group_col: 分组列名
        value_col: 数值列名
        test: 't-test' 或 'mann-whitney'
    
    Returns:
        dict: 检验结果
    """
    groups = df[group_col].unique()
    if len(groups) != 2:
        return {'error': '需要恰好2个组'}
    
    g1 = df[df[group_col] == groups[0]][value_col].dropna()
    g2 = df[df[group_col] == groups[1]][value_col].dropna()
    
    if len(g1) < 3 or len(g2) < 3:
        return {'error': '每组至少需要3个样本'}
    
    if test == 't-test':
        stat, p = ttest_ind(g1, g2, equal_var=False)  # Welch's t-test
        test_name = "Welch's t-test"
    else:
        stat, p = mannwhitneyu(g1, g2, alternative='two-sided')
        test_name = "Mann-Whitney U"
    
    # 计算效应量
    effect_size = (g1.mean() - g2.mean()) / np.sqrt((g1.std()**2 + g2.std()**2) / 2)
    
    return {
        'test': test_name,
        'statistic': stat,
        'p_value': p,
        'n1': len(g1),
        'n2': len(g2),
        'mean1': g1.mean(),
        'mean2': g2.mean(),
        'std1': g1.std(),
        'std2': g2.std(),
        'effect_size': effect_size,
        'significant': p < 0.05,
        'effect_interpretation': interpret_effect_size(effect_size)
    }


def kruskal_wallis_test(df, group_col, value_col):
    """
    多组比较（Kruskal-Wallis检验）
    
    Args:
        df: 数据框
        group_col: 分组列名
        value_col: 数值列名
    
    Returns:
        dict: 检验结果
    """
    groups = df[group_col].dropna().unique()
    group_data = [df[df[group_col] == g][value_col].dropna().values for g in groups]
    
    # 过滤空组
    group_data = [g for g in group_data if len(g) >= 3]
    
    if len(group_data) < 2:
        return {'error': '需要至少2个有效组'}
    
    stat, p = kruskal(*group_data)
    
    # 计算效应量 (Epsilon-squared)
    n = sum(len(g) for g in group_data)
    k = len(group_data)
    epsilon_sq = (stat - k + 1) / (n - k)
    
    return {
        'test': 'Kruskal-Wallis H',
        'statistic': stat,
        'p_value': p,
        'n_groups': k,
        'n_total': n,
        'epsilon_squared': epsilon_sq,
        'significant': p < 0.05,
        'group_means': {groups[i]: np.mean(group_data[i]) for i in range(len(groups))}
    }


def interpret_effect_size(d):
    """解释效应量大小 (Cohen's d)"""
    abs_d = abs(d)
    if abs_d < 0.2:
        return 'negligible'
    elif abs_d < 0.5:
        return 'small'
    elif abs_d < 0.8:
        return 'medium'
    else:
        return 'large'


# ============================================================================
# 回归分析
# ============================================================================

def simple_regression_with_stats(x, y):
    """
    简单线性回归（含统计检验）
    
    Args:
        x: 自变量
        y: 因变量
    
    Returns:
        dict: 回归结果
    """
    # 去除NaN
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    
    if len(x) < 3:
        return {'error': '样本量不足'}
    
    # 线性回归
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    # R²
    r_squared = r_value ** 2
    
    # 置信区间 (95%)
    n = len(x)
    t_val = stats.t.ppf(0.975, n - 2)
    ci_lower = slope - t_val * std_err
    ci_upper = slope + t_val * std_err
    
    return {
        'slope': slope,
        'intercept': intercept,
        'r': r_value,
        'r_squared': r_squared,
        'p_value': p_value,
        'std_err': std_err,
        'ci_95': (ci_lower, ci_upper),
        'n': n,
        'significant': p_value < 0.05
    }


def dose_response_analysis(df, exposure_col, outcome_col, n_groups=4):
    """
    剂量-效应关系分析
    
    Args:
        df: 数据框
        exposure_col: 暴露列
        outcome_col: 结果列
        n_groups: 分组数量
    
    Returns:
        dict: 分析结果
    """
    # 去除缺失值
    data = df[[exposure_col, outcome_col]].dropna()
    
    if len(data) < 10:
        return {'error': '样本量不足'}
    
    # 按暴露量分组
    data['exposure_group'] = pd.qcut(data[exposure_col], q=n_groups, labels=False, duplicates='drop')
    
    # 计算每组的中位数暴露和平均结果
    group_stats = data.groupby('exposure_group').agg({
        exposure_col: 'median',
        outcome_col: ['mean', 'std', 'count']
    }).round(3)
    
    # 趋势检验（Spearman相关）
    median_exposure = group_stats[(exposure_col, 'median')].values
    mean_outcome = group_stats[(outcome_col, 'mean')].values
    
    if len(median_exposure) >= 3:
        trend_r, trend_p = spearmanr(median_exposure, mean_outcome)
    else:
        trend_r, trend_p = np.nan, np.nan
    
    return {
        'group_stats': group_stats,
        'trend_r': trend_r,
        'trend_p': trend_p,
        'trend_significant': trend_p < 0.05 if not np.isnan(trend_p) else False,
        'n_groups': len(group_stats),
        'interpretation': interpret_dose_response(trend_r, trend_p)
    }


def interpret_dose_response(r, p):
    """解释剂量-效应关系"""
    if np.isnan(p):
        return "数据不足以判断趋势"
    
    if p >= 0.05:
        return "无显著剂量-效应关系"
    
    if r > 0:
        return f"存在显著正相关 (r={r:.3f}, p={p:.3f})"
    else:
        return f"存在显著负相关 (r={r:.3f}, p={p:.3f})"


# ============================================================================
# 效应量计算
# ============================================================================

def cohens_d(group1, group2):
    """计算Cohen's d效应量"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    
    #  pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0
    
    return (group1.mean() - group2.mean()) / pooled_std


def cramers_v(contingency_table):
    """计算Cramer's V（分类变量关联强度）"""
    chi2 = stats.chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape[0], contingency_table.shape[1]) - 1
    
    if min_dim == 0 or n == 0:
        return 0
    
    return np.sqrt(chi2 / (n * min_dim))


# ============================================================================
# 综合统计分析
# ============================================================================

def comprehensive_correlation_analysis(df, metals, health_outcomes):
    """
    综合相关性分析（重金属 vs 健康指标）
    
    Args:
        df: 数据框
        metals: 重金属列列表
        health_outcomes: 健康指标列列表
    
    Returns:
        DataFrame: 相关性分析结果表
    """
    results = []
    
    for metal in metals:
        metal_info = METAL_NAMES.get(metal, (metal, metal))
        
        for outcome in health_outcomes:
            # Pearson相关
            pearson_result = correlation_with_significance(df, metal, outcome, 'pearson')
            # Spearman相关
            spearman_result = correlation_with_significance(df, metal, outcome, 'spearman')
            
            results.append({
                'Metal': metal_info[0],
                'Metal_CN': metal_info[1],
                'Outcome': outcome,
                'Pearson_r': pearson_result['r'],
                'Pearson_p': pearson_result['p_value'],
                'Pearson_n': pearson_result['n'],
                'Pearson_sig': pearson_result['significant'],
                'Spearman_r': spearman_result['r'],
                'Spearman_p': spearman_result['p_value'],
                'Spearman_n': spearman_result['n'],
                'Spearman_sig': spearman_result['significant'],
            })
    
    results_df = pd.DataFrame(results)
    
    # FDR校正
    if len(results_df) > 0:
        p_cols = ['Pearson_p', 'Spearman_p']
        for col in p_cols:
            p_values = results_df[col].values
            corrected = multiple_correlation_correction(p_values, 'fdr')
            results_df[col.replace('_p', '_p_fdr')] = corrected['adjusted']
            results_df[col.replace('_p', '_sig_fdr')] = corrected['significant']
    
    return results_df


def generate_statistical_report(df, metals, outcomes, output_path=None):
    """
    生成完整统计分析报告
    
    Args:
        df: 数据框
        metals: 重金属列列表
        outcomes: 健康指标列列表
        output_path: 输出路径
    
    Returns:
        str: 报告文本
    """
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("网络毒理学统计分析报告")
    report_lines.append("Network Toxicology Statistical Analysis Report")
    report_lines.append("=" * 70)
    report_lines.append("")
    
    # 1. 数据概览
    report_lines.append("1. 数据概览")
    report_lines.append("-" * 40)
    report_lines.append(f"样本量: {len(df)}")
    report_lines.append(f"重金属变量: {len(metals)}")
    report_lines.append(f"健康指标: {len(outcomes)}")
    report_lines.append("")
    
    # 2. 相关性分析
    report_lines.append("2. 重金属-健康指标相关性分析")
    report_lines.append("-" * 40)
    
    corr_results = comprehensive_correlation_analysis(df, metals, outcomes)
    
    # 显著结果汇总
    sig_results = corr_results[corr_results['Spearman_sig'] == True].sort_values('Spearman_p')
    
    report_lines.append(f"显著相关数: {len(sig_results)}/{len(corr_results)}")
    report_lines.append("")
    
    if len(sig_results) > 0:
        report_lines.append("显著相关结果 (Spearman, p < 0.05):")
        for _, row in sig_results.head(20).iterrows():
            direction = "↑" if row['Spearman_r'] > 0 else "↓"
            report_lines.append(
                f"  {row['Metal_CN']}({row['Metal']}) - {row['Outcome']}: "
                f"r={row['Spearman_r']:.3f}, p={row['Spearman_p']:.2e} {direction}"
            )
    
    report_lines.append("")
    
    # 3. 多重比较校正
    report_lines.append("3. 多重比较校正 (FDR)")
    report_lines.append("-" * 40)
    
    fdr_sig = corr_results[corr_results['Spearman_p_fdr'] < 0.05]
    report_lines.append(f"FDR校正后显著相关数: {len(fdr_sig)}/{len(corr_results)}")
    report_lines.append("")
    
    # 4. 效应量解释
    report_lines.append("4. 效应量解释")
    report_lines.append("-" * 40)
    report_lines.append("Cohen's d解释标准:")
    report_lines.append("  |d| < 0.2: 微效应 (negligible)")
    report_lines.append("  0.2 ≤ |d| < 0.5: 小效应 (small)")
    report_lines.append("  0.5 ≤ |d| < 0.8: 中效应 (medium)")
    report_lines.append("  |d| ≥ 0.8: 大效应 (large)")
    report_lines.append("")
    
    # 保存结果
    if output_path:
        corr_results.to_csv(output_path, index=False, encoding='utf-8-sig')
        report_lines.append(f"详细结果已保存至: {output_path}")
    
    report_text = "\n".join(report_lines)
    return report_text, corr_results


# ============================================================================
# 主函数 - 测试
# ============================================================================

if __name__ == "__main__":
    # 创建模拟数据测试
    np.random.seed(42)
    n = 500
    
    # 模拟NHANES数据
    lead_vals = np.random.lognormal(1.5, 0.8, n)  # 铅
    data = {
        'LBXBPB': lead_vals,
        'LBXIAS': np.random.lognormal(0.5, 0.6, n),   # 砷
        'LBXBCD': np.random.lognormal(-1, 0.5, n),   # 镉
        'LBXBMN': np.random.lognormal(2, 0.5, n),    # 锰
        'LBXIHG': np.random.lognormal(0.8, 0.4, n),  # 汞
        # 健康指标
        'systolic_bp': 120 + 10 * np.random.randn(n) + 5 * np.log(lead_vals),
        'HbA1c': 5.5 + 0.3 * np.random.randn(n) + 0.2 * np.log(lead_vals),
        'eGFR': 90 - 3 * np.log(np.random.lognormal(-1, 0.5, n)) + 5 * np.random.randn(n),
    }
    
    df = pd.DataFrame(data)
    
    metals = ['LBXBPB', 'LBXIAS', 'LBXBCD', 'LBXBMN', 'LBXIHG']
    outcomes = ['systolic_bp', 'HbA1c', 'eGFR']
    
    # 生成报告
    report, results_df = generate_statistical_report(df, metals, outcomes, 'output/statistical_analysis_results.csv')
    print(report)
    
    print("\n" + "=" * 70)
    print("统计分析模块测试完成!")
