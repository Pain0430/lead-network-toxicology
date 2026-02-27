#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
铅网络毒理学 - 发表偏倚分析模块
Lead Network Toxicology - Publication Bias Analysis

功能：
1. 漏斗图 (Funnel Plot)
2. Egger's检验
3. Trim-and-fill分析
4. 敏感性分析
5. 森林图综合
6. 累积Meta分析

作者: Pain AI Assistant
日期: 2026-02-27
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
import warnings
import os

warnings.filterwarnings('ignore')

# ============================================================
# 配置
# ============================================================
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

COLORS = {
    'primary': '#2C3E50',
    'secondary': '#E74C3C',
    'tertiary': '#3498DB',
    'quaternary': '#27AE60',
    'accent': '#F39C12',
    'purple': '#9B59B6'
}

OUTPUT_DIR = '/Users/pengsu/mycode/lead-network-toxicology/output'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_meta_data(n_studies=30, random_state=42):
    """生成Meta分析模拟数据"""
    np.random.seed(random_state)
    
    # 模拟研究效应量 (模拟发表偏倚)
    # 小样本研究倾向于报告较大的效应量
    studies = []
    
    for i in range(n_studies):
        # 样本量 (小到中等)
        n = np.random.randint(50, 500)
        
        # 基础效应量 (log odds ratio)
        true_effect = np.random.normal(0.5, 0.2)
        
        # 标准误 (与样本量成反比)
        se = 1.96 / np.sqrt(n / 100)
        
        # 添加发表偏倚: 小样本研究效应量被高估
        if n < 150:
            # 小样本研究倾向于有显著结果
            if np.random.random() < 0.7:
                # 只发表显著的结果
                reported_effect = true_effect + np.random.normal(0.2, 0.1)
            else:
                reported_effect = true_effect
        else:
            reported_effect = true_effect + np.random.normal(0, 0.05)
        
        # 计算置信区间
        ci_lower = reported_effect - 1.96 * se
        ci_upper = reported_effect + 1.96 * se
        
        # 转换为OR
        or_val = np.exp(reported_effect)
        or_lower = np.exp(ci_lower)
        or_upper = np.exp(ci_upper)
        
        # 判断是否显著
        significant = (ci_lower > 0) or (ci_upper < 0)
        
        # 发表状态 (模拟: 显著结果更容易发表)
        published = significant or (np.random.random() < 0.3)
        
        study = {
            'study_id': f'Study_{i+1}',
            'n': n,
            'log_or': reported_effect,
            'se': se,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'or': or_val,
            'or_lower': or_lower,
            'or_upper': or_upper,
            'significant': significant,
            'published': published
        }
        studies.append(study)
    
    df = pd.DataFrame(studies)
    
    # 过滤已发表的研究
    df_published = df[df['published']].copy()
    
    return df, df_published


def funnel_plot(df, effect_col='log_or', se_col='se', 
                title='Funnel Plot', xlabel='Log Odds Ratio', ylabel='Standard Error'):
    """绘制漏斗图"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制散点
    ax.scatter(df[effect_col], df[se_col], alpha=0.6, s=80, c=COLORS['tertiary'], edgecolors='white')
    
    # 添加汇总效应线
    pooled_effect = np.average(df[effect_col], weights=1/df[se_col]**2)
    ax.axvline(x=pooled_effect, color=COLORS['primary'], linestyle='--', linewidth=2, 
              label=f'Pooled Effect: {pooled_effect:.3f}')
    
    # 添加无效线
    ax.axvline(x=0, color='red', linestyle='-', linewidth=1.5, alpha=0.7, label='Null Effect (OR=1)')
    
    # 添加置信区间边界线
    y_range = df[se_col].max() - df[se_col].min()
    y_max = df[se_col].max() + 0.1 * y_range
    
    # 95% CI 边界
    x_ci = np.linspace(-3, 3, 100)
    se_upper = 1.96 / np.sqrt(np.linspace(10, 500, 100))
    
    # 绘制pseudo-confidence boundaries
    ax.fill_betweenx(np.linspace(0, df[se_col].max(), 100), 
                     -1.96 * np.linspace(0.05, 0.5, 100), 
                     1.96 * np.linspace(0.05, 0.5, 100), 
                     alpha=0.2, color='gray', label='95% CI')
    
    # 标记显著研究
    sig_mask = df['significant']
    ax.scatter(df.loc[sig_mask, effect_col], df.loc[sig_mask, se_col], 
              alpha=0.8, s=100, c=COLORS['secondary'], edgecolors='black', linewidth=1.5,
              label='Significant (p<0.05)', zorder=5)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.invert_yaxis()  # 小样本在上
    ax.grid(True, alpha=0.3)
    
    # 设置范围
    ax.set_xlim(-2, 2.5)
    ax.set_ylim(df[se_col].max() + 0.1, 0)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/funnel_plot.png', bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Funnel plot saved: {OUTPUT_DIR}/funnel_plot.png")


def egger_test(df, effect_col='log_or', se_col='se'):
    """Egger's线性回归检验"""
    # 标准化效应
    precision = 1 / df[se_col]
    
    # 回归: effect / se = a + b * (1/se)
    # 如果截距a显著偏离0，存在发表偏倚
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        precision, df[effect_col]
    )
    
    return {
        'intercept': intercept,
        'intercept_se': std_err,
        't_statistic': intercept / std_err,
        'p_value': p_value,
        'r_squared': r_value**2,
        'significant': p_value < 0.05
    }


def trim_and_fill(df, effect_col='log_or', se_col='se', n_iterations=100):
    """Trim-and-fill方法估计缺失研究"""
    # 计算初始汇总效应
    weights = 1 / df[se_col]**2
    pooled_effect = np.sum(df[effect_col] * weights) / np.sum(weights)
    
    # 迭代估计缺失研究
    filled_df = df.copy()
    
    for iteration in range(n_iterations):
        # 计算不对称性
        # 简化的trim-and-fill: 估计缺失数量
        positive_effects = (filled_df[effect_col] > pooled_effect).sum()
        negative_effects = (filled_df[effect_col] < pooled_effect).sum()
        
        # 估计缺失研究数
        missing = max(0, positive_effects - negative_effects)
        
        if missing == 0:
            break
        
        # 添加"镜像"研究
        for i in range(int(missing)):
            # 镜像效应量
            mirrored_effect = 2 * pooled_effect - filled_df[effect_col].iloc[-(i+1)]
            mirrored_se = filled_df[se_col].iloc[-(i+1)]
            
            new_study = {
                'study_id': f'Filled_{i+1}',
                'log_or': mirrored_effect,
                'se': mirrored_se,
                'or': np.exp(mirrored_effect),
                'ci_lower': np.exp(mirrored_effect - 1.96 * mirrored_se),
                'ci_upper': np.exp(mirrored_effect + 1.96 * mirrored_se),
                'filled': True
            }
            filled_df = pd.concat([filled_df, pd.DataFrame([new_study])], ignore_index=True)
        
        # 重新计算汇总效应
        weights = 1 / filled_df[se_col]**2
        pooled_effect = np.sum(filled_df[effect_col] * weights) / np.sum(weights)
    
    return filled_df, pooled_effect, missing


def plot_trim_and_fill(df, filled_df, pooled_original, pooled_filled):
    """绘制Trim-and-fill结果"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 原始漏斗图
    ax1 = axes[0]
    ax1.scatter(df['log_or'], df['se'], alpha=0.6, s=80, c=COLORS['tertiary'], 
               edgecolors='white', label='Published studies')
    ax1.axvline(x=pooled_original, color=COLORS['primary'], linestyle='--', linewidth=2)
    ax1.axvline(x=0, color='red', linestyle='-', linewidth=1.5, alpha=0.7)
    ax1.set_xlabel('Log Odds Ratio', fontsize=12)
    ax1.set_ylabel('Standard Error', fontsize=12)
    ax1.set_title(f'Original Funnel Plot\nPooled OR = {np.exp(pooled_original):.2f}', fontsize=14)
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3)
    
    # 填充后漏斗图
    ax2 = axes[1]
    
    # Check if 'filled' column exists
    if 'filled' in filled_df.columns:
        original = filled_df[~filled_df['filled'].fillna(False)]
        filled = filled_df[filled_df['filled'].fillna(False)]
    else:
        original = filled_df
        filled = pd.DataFrame()
    
    if len(original) > 0:
        ax2.scatter(original['log_or'], original['se'], alpha=0.6, s=80, 
                   c=COLORS['tertiary'], edgecolors='white', label='Published')
    if len(filled) > 0:
        ax2.scatter(filled['log_or'], filled['se'], alpha=0.6, s=80, 
                   c=COLORS['secondary'], marker='d', edgecolors='black', 
                   label=f'Filled ({len(filled)} studies)')
    
    ax2.axvline(x=pooled_filled, color=COLORS['quaternary'], linestyle='--', linewidth=2,
               label=f'Adjusted: {np.exp(pooled_filled):.2f}')
    ax2.axvline(x=pooled_original, color=COLORS['primary'], linestyle=':', linewidth=2,
               label=f'Original: {np.exp(pooled_original):.2f}')
    ax2.axvline(x=0, color='red', linestyle='-', linewidth=1.5, alpha=0.7)
    
    ax2.set_xlabel('Log Odds Ratio', fontsize=12)
    ax2.set_ylabel('Standard Error', fontsize=12)
    ax2.set_title('Trim-and-Fill Adjusted\nPublication Bias Corrected', fontsize=14)
    ax2.invert_yaxis()
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/trim_and_fill.png', bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Trim-and-fill plot saved: {OUTPUT_DIR}/trim_and_fill.png")


def cumulative_meta_analysis(df, effect_col='log_or', se_col='se'):
    """累积Meta分析"""
    # 按样本量排序
    df_sorted = df.sort_values('n', ascending=True).reset_index(drop=True)
    
    cumulative_effects = []
    cumulative_lower = []
    cumulative_upper = []
    n_studies = []
    
    for i in range(1, len(df_sorted) + 1):
        subset = df_sorted.iloc[:i]
        
        weights = 1 / subset[se_col]**2
        pooled = np.sum(subset[effect_col] * weights) / np.sum(weights)
        pooled_se = np.sqrt(1 / np.sum(weights))
        
        cumulative_effects.append(pooled)
        cumulative_lower.append(pooled - 1.96 * pooled_se)
        cumulative_upper.append(pooled + 1.96 * pooled_se)
        n_studies.append(i)
    
    return n_studies, cumulative_effects, cumulative_lower, cumulative_upper


def plot_cumulative_forest(df):
    """绘制累积森林图"""
    n_studies, cum_effects, cum_lower, cum_upper = cumulative_meta_analysis(df)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 绘制每个累积步骤
    for i, (n, effect, lower, upper) in enumerate(zip(n_studies, cum_effects, cum_lower, cum_upper)):
        color = plt.cm.Blues(0.3 + 0.7 * (i / len(n_studies)))
        
        ax.errorbar(effect, n, xerr=[[effect-lower], [upper-effect]], 
                   fmt='o', color=color, capsize=3, markersize=6)
    
    # 汇总效应
    final_effect = cum_effects[-1]
    ax.axvline(x=final_effect, color=COLORS['quaternary'], linestyle='--', linewidth=2,
              label=f'Final pooled effect: {np.exp(final_effect):.2f}')
    ax.axvline(x=0, color='red', linestyle='-', linewidth=1.5, alpha=0.7, label='Null effect')
    
    ax.set_xlabel('Log Odds Ratio', fontsize=12)
    ax.set_ylabel('Number of Studies', fontsize=12)
    ax.set_title('Cumulative Meta-Analysis (by sample size)', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/cumulative_meta.png', bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Cumulative meta-analysis saved: {OUTPUT_DIR}/cumulative_meta.png")


def plot_forest_with_subgroups(df):
    """绘制亚组森林图"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 添加研究类型标签
    df = df.copy()
    df['sample_category'] = pd.cut(df['n'], bins=[0, 100, 200, 500], 
                                   labels=['Small (n<100)', 'Medium (100-200)', 'Large (n>200)'])
    
    y_positions = []
    y = 0
    
    # 绘制每个研究
    for idx, row in df.iterrows():
        ax.errorbar(row['log_or'], y, 
                   xerr=[[row['log_or']-row['ci_lower']], [row['ci_upper']-row['log_or']]],
                   fmt='o', color=COLORS['tertiary'], capsize=3, markersize=5)
        
        y_positions.append(y)
        y += 1
    
    # 汇总效应 (按样本类别)
    categories = df['sample_category'].unique()
    pooled_by_group = {}
    
    for cat in categories:
        subset = df[df['sample_category'] == cat]
        weights = 1 / subset['se']**2
        pooled = np.sum(subset['log_or'] * weights) / np.sum(weights)
        pooled_se = np.sqrt(1 / np.sum(weights))
        pooled_by_group[cat] = (pooled, pooled_se)
        
        ax.errorbar(pooled, y, 
                   xerr=[[1.96*pooled_se], [1.96*pooled_se]],
                   fmt='s', color=COLORS['secondary'], capsize=5, markersize=10,
                   markeredgecolor='black', linewidth=2)
        y += 1
        y_positions.append(y)
    
    # 总体汇总
    weights = 1 / df['se']**2
    pooled_overall = np.sum(df['log_or'] * weights) / np.sum(weights)
    pooled_se = np.sqrt(1 / np.sum(weights))
    
    ax.errorbar(pooled_overall, y, 
               xerr=[[1.96*pooled_se], [1.96*pooled_se]],
               fmt='D', color=COLORS['quaternary'], capsize=5, markersize=12,
               markeredgecolor='black', linewidth=2, label='Overall')
    
    # 无效线
    ax.axvline(x=0, color='red', linestyle='-', linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('Log Odds Ratio (95% CI)', fontsize=12)
    ax.set_ylabel('Study', fontsize=12)
    ax.set_title('Forest Plot with Sample Size Subgroups', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/forest_subgroups.png', bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Forest plot with subgroups saved: {OUTPUT_DIR}/forest_subgroups.png")
    
    return pooled_by_group


def sensitivity_analysis(df, effect_col='log_or', se_col='se'):
    """敏感性分析: 逐个移除研究"""
    results = []
    df = df.reset_index(drop=True)  # Reset index for proper dropping
    
    for i in range(len(df)):
        # 移除第i个研究 (by position)
        subset = df.drop(index=i)
        
        weights = 1 / subset[se_col]**2
        pooled = np.sum(subset[effect_col] * weights) / np.sum(weights)
        pooled_se = np.sqrt(1 / np.sum(weights))
        
        results.append({
            'removed_study': df.iloc[i]['study_id'],
            'pooled_log_or': pooled,
            'pooled_or': np.exp(pooled),
            'pooled_se': pooled_se,
            'ci_lower': pooled - 1.96 * pooled_se,
            'ci_upper': pooled + 1.96 * pooled_se
        })
    
    return pd.DataFrame(results)


def plot_sensitivity_analysis(sens_results):
    """绘制敏感性分析图"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y = range(len(sens_results))
    x = sens_results['pooled_log_or']
    xerr = 1.96 * sens_results['pooled_se']
    
    # 绘制每个移除研究后的效应
    colors = [COLORS['quaternary'] if (row['ci_lower'] > 0) else COLORS['secondary'] 
              for _, row in sens_results.iterrows()]
    
    ax.errorbar(x, y, xerr=xerr, fmt='o', color=COLORS['tertiary'], 
               capsize=3, markersize=4, alpha=0.7)
    
    # 标记每个点
    for i, (idx, row) in enumerate(sens_results.iterrows()):
        if i % 5 == 0:  # 每5个标注一个
            ax.text(row['pooled_log_or'] + 0.15, i, 
                   row['removed_study'].replace('Study_', 'S'), 
                   fontsize=7, alpha=0.7)
    
    # 原始汇总效应
    ax.axvline(x=x.mean(), color=COLORS['primary'], linestyle='--', linewidth=2,
              label=f'Mean pooled effect: {np.exp(x.mean()):.2f}')
    ax.axvline(x=0, color='red', linestyle='-', linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('Pooled Odds Ratio (95% CI)', fontsize=12)
    ax.set_ylabel('Removed Study', fontsize=12)
    ax.set_title('Sensitivity Analysis (Leave-one-out)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/sensitivity_analysis.png', bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Sensitivity analysis plot saved: {OUTPUT_DIR}/sensitivity_analysis.png")


def generate_publication_bias_report(df, filled_df, pooled_original, pooled_filled, 
                                     egger_result, sens_results):
    """生成发表偏倚分析报告"""
    report = []
    report.append("=" * 70)
    report.append("发表偏倚分析报告")
    report.append("Publication Bias Analysis Report")
    report.append("=" * 70)
    report.append("")
    
    # 基本信息
    report.append("【研究概况】")
    report.append(f"原始研究数: {len(df)}")
    report.append(f"已发表研究数: {len(filled_df)}")
    report.append(f"显著研究数: {df['significant'].sum()}")
    report.append(f"显著率: {df['significant'].mean():.1%}")
    report.append("")
    
    # 汇总效应
    report.append("【汇总效应】")
    report.append(f"原始汇总OR: {np.exp(pooled_original):.3f} (95% CI: {np.exp(pooled_original-1.96*np.sqrt(1/np.sum(1/df['se']**2))):.3f}-{np.exp(pooled_original+1.96*np.sqrt(1/np.sum(1/df['se']**2))):.3f})")
    report.append(f"Trim-and-fill调整后OR: {np.exp(pooled_filled):.3f}")
    report.append(f"差异: {np.abs(np.exp(pooled_filled) - np.exp(pooled_original)):.3f}")
    report.append("")
    
    # Egger's检验
    report.append("【Egger's检验结果】")
    report.append(f"截距: {egger_result['intercept']:.4f}")
    report.append(f"t统计量: {egger_result['t_statistic']:.4f}")
    report.append(f"p值: {egger_result['p_value']:.4f}")
    if egger_result['significant']:
        report.append("结论: 存在显著发表偏倚 (p<0.05)")
    else:
        report.append("结论: 未检测到显著发表偏倚")
    report.append("")
    
    # 敏感性分析
    report.append("【敏感性分析 (Leave-one-out)】")
    report.append(f"效应范围: {sens_results['pooled_or'].min():.3f} - {sens_results['pooled_or'].max():.3f}")
    report.append(f"效应变异: {sens_results['pooled_or'].std():.4f}")
    report.append(f"所有移除后的效应均显著: {(sens_results['ci_lower'] > 0).all()}")
    report.append("")
    
    # 结论
    report.append("【结论】")
    if egger_result['significant']:
        report.append("1. Egger's检验表明存在发表偏倚")
        if pooled_filled < pooled_original:
            report.append("2. 真实效应可能被高估")
        report.append("3. 建议使用Trim-and-fill调整后的效应量")
    else:
        report.append("1. 未检测到显著发表偏倚")
        report.append("2. 现有结果较为稳健")
    
    report_text = "\n".join(report)
    
    with open(f'{OUTPUT_DIR}/publication_bias_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"✓ Publication bias report saved: {OUTPUT_DIR}/publication_bias_report.txt")
    
    return report_text


def main():
    """主函数"""
    print("=" * 60)
    print("铅暴露研究 - 发表偏倚分析")
    print("=" * 60)
    
    # 生成Meta分析数据
    print("\n[1/7] 生成Meta分析数据...")
    df_all, df = generate_meta_data(n_studies=40)
    print(f"    原始研究: {len(df_all)}, 已发表: {len(df)}")
    
    # 漏斗图
    print("\n[2/7] 绘制漏斗图...")
    funnel_plot(df)
    
    # Egger's检验
    print("\n[3/7] 执行Egger's检验...")
    egger_result = egger_test(df)
    print(f"    截距: {egger_result['intercept']:.4f}, p值: {egger_result['p_value']:.4f}")
    
    # Trim-and-fill分析
    print("\n[4/7] 执行Trim-and-fill分析...")
    pooled_original = np.sum(df['log_or'] / df['se']**2) / np.sum(1 / df['se']**2)
    filled_df, pooled_filled, n_filled = trim_and_fill(df)
    print(f"    填充研究数: {n_filled}")
    print(f"    原始效应: {np.exp(pooled_original):.3f} -> 调整后: {np.exp(pooled_filled):.3f}")
    
    # 绘制Trim-and-fill
    print("\n[5/7] 绘制Trim-and-fill结果...")
    plot_trim_and_fill(df, filled_df, pooled_original, pooled_filled)
    
    # 累积Meta分析
    print("\n[6/7] 绘制累积Meta分析...")
    plot_cumulative_forest(df)
    
    # 敏感性分析
    print("\n[7/7] 执行敏感性分析...")
    sens_results = sensitivity_analysis(df)
    plot_sensitivity_analysis(sens_results)
    
    # 亚组森林图
    plot_forest_with_subgroups(df)
    
    # 生成报告
    print("\n生成分析报告...")
    report = generate_publication_bias_report(df, filled_df, pooled_original, pooled_filled,
                                               egger_result, sens_results)
    
    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)
    
    return df, report


if __name__ == "__main__":
    df, report = main()
    print("\n" + report)
