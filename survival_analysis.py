"""
铅毒性生存分析模块
Survival Analysis Module for Lead Toxicity Research

功能:
1. Kaplan-Meier 生存曲线
2. Log-rank 检验
3. Cox 比例风险模型
4. 生存森林图
5. 生存列线图

作者: AI Assistant
日期: 2026-02-26
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def generate_survival_data(n=500, seed=42):
    """
    生成铅毒性模拟生存数据
    
    参数:
        n: 样本数量
        seed: 随机种子
    
    返回:
        DataFrame: 包含生存时间和事件的DataFrame
    """
    np.random.seed(seed)
    
    # 基本特征
    age = np.random.normal(55, 15, n)
    bmi = np.random.normal(24, 4, n)
    
    # 铅暴露指标 (确保正值)
    blood_lead = np.random.exponential(15, n) + 1  # 血铅 (加1确保>0)
    urine_lead = np.random.exponential(8, n) + 1    # 尿铅
    hair_lead = np.random.exponential(20, n) + 1   # 发铅
    
    # 生活习惯
    smoking = np.random.binomial(1, 0.3, n)
    alcohol = np.random.binomial(1, 0.25, n)
    occupation = np.random.binomial(1, 0.35, n)  # 职业暴露
    
    # 创建风险评分（铅暴露越高，风险越大）
    risk_score = (
        0.03 * blood_lead + 
        0.05 * urine_lead + 
        0.02 * hair_lead +
        0.5 * smoking +
        0.4 * alcohol +
        0.6 * occupation +
        0.01 * age +
        0.02 * (bmi - 24)
    )
    
    # 生成生存时间（使用指数分布）
    base_hazard = 0.02
    hazard = base_hazard * np.exp(risk_score - risk_score.mean())
    time = np.random.exponential(1/hazard, n)
    
    # 删失（模拟随访结束）
    censor_time = np.random.uniform(12, 60, n)  # 12-60个月的随访
    event = (time <= censor_time).astype(int)   # 1=事件发生, 0=删失
    observed_time = np.minimum(time, censor_time)
    
    # 创建DataFrame
    df = pd.DataFrame({
        'ID': range(1, n+1),
        'Time_Months': observed_time,
        'Event': event,  # 1=死亡/事件, 0=删失
        'Age': age,
        'BMI': bmi,
        'Blood_Lead': blood_lead,
        'Urine_Lead': urine_lead,
        'Hair_Lead': hair_lead,
        'Smoking': smoking,
        'Alcohol': alcohol,
        'Occupational_Exposure': occupation
    })
    
    # 添加分组变量 (确保所有值都能分组)
    df['Lead_Group'] = pd.cut(df['Blood_Lead'], 
                              bins=[0, 5, 15, 100],
                              labels=['低暴露', '中暴露', '高暴露'])
    df['Age_Group'] = pd.cut(df['Age'],
                             bins=[0, 40, 60, 100],
                             labels=['年轻', '中年', '老年'])
    
    # 删除无法分类的行
    df = df.dropna(subset=['Lead_Group', 'Age_Group'])
    
    return df


def plot_kaplan_meier(df, time_col='Time_Months', event_col='Event', 
                      group_col=None, save_path=None):
    """
    绘制 Kaplan-Meier 生存曲线
    
    参数:
        df: DataFrame
        time_col: 时间列名
        event_col: 事件列名
        group_col: 分组列名（可选）
        save_path: 保存路径
    
    返回:
        图对象
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    kmf = KaplanMeierFitter()
    
    if group_col is None:
        # 单条曲线
        kmf.fit(df[time_col], df[event_col], label='整体生存')
        kmf.plot_survival_function(ax=ax, ci_show=True, color='#2E86AB')
    else:
        # 多条曲线（按组）
        groups = df[group_col].unique()
        colors = ['#2E86AB', '#E94F37', '#F39C12', '#27AE60', '#8E44AD']
        
        for i, group in enumerate(groups):
            group_data = df[df[group_col] == group]
            kmf.fit(group_data[time_col], group_data[event_col], 
                   label=f'{group}')
            kmf.plot_survival_function(ax=ax, ci_show=True, 
                                       color=colors[i % len(colors)])
    
    ax.set_xlabel('时间 (月)', fontsize=12)
    ax.set_ylabel('生存概率', fontsize=12)
    ax.set_title('Kaplan-Meier 生存曲线', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    # 添加风险表
    textstr = '风险表 (n at risk)'
    ax.text(0.98, 0.98, textstr, transform=ax.transAxes, 
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"生存曲线已保存: {save_path}")
    
    return fig


def logrank_test_groups(df, group_col, time_col='Time_Months', event_col='Event'):
    """
    执行 Log-rank 检验
    
    参数:
        df: DataFrame
        group_col: 分组列名
        time_col: 时间列名
        event_col: 事件列名
    
    返回:
        检验结果字典
    """
    groups = df[group_col].unique()
    
    if len(groups) < 2:
        raise ValueError("需要至少两组进行比较")
    
    # 比较第一组和最后一组（通常是低暴露 vs 高暴露）
    group1 = df[df[group_col] == groups[0]]
    group2 = df[df[group_col] == groups[-1]]
    
    results = logrank_test(
        group1[time_col], group2[time_col],
        group1[event_col], group2[event_col]
    )
    
    return {
        'test_statistic': results.test_statistic,
        'p_value': results.p_value,
        'group1': groups[0],
        'group2': groups[-1]
    }


def fit_cox_model(df, time_col='Time_Months', event_col='Event', 
                  covariates=None):
    """
    拟合 Cox 比例风险模型
    
    参数:
        df: DataFrame
        time_col: 时间列名
        event_col: 事件列名
        covariates: 协变量列表
    
    返回:
        CoxPHFitter 模型对象
    """
    if covariates is None:
        covariates = ['Age', 'BMI', 'Blood_Lead', 'Smoking', 'Alcohol', 
                      'Occupational_Exposure']
    
    # 准备数据 - 确保所有列都是数值类型
    cox_data = df[[time_col, event_col] + covariates].copy()
    cox_data[event_col] = cox_data[event_col].astype(bool)
    
    # 转换为数值类型
    for col in cox_data.columns:
        if cox_data[col].dtype.name == 'category':
            cox_data[col] = cox_data[col].astype(float)
        elif cox_data[col].dtype == 'object':
            cox_data[col] = pd.to_numeric(cox_data[col], errors='coerce')
    
    # 删除任何包含NaN的行
    cox_data = cox_data.dropna()
    
    # 拟合模型
    cph = CoxPHFitter()
    cph.fit(cox_data, duration_col=time_col, event_col=event_col)
    
    return cph


def plot_cox_summary(cph, save_path=None):
    """
    绘制 Cox 模型摘要图
    
    参数:
        cph: CoxPHFitter 对象
        save_path: 保存路径
    
    返回:
        图对象
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 获取HR和置信区间
    summary = cph.summary.copy()
    hr = summary['exp(coef)']
    ci_lower = np.exp(summary['coef'] - 1.96 * summary['se(coef)'])
    ci_upper = np.exp(summary['coef'] + 1.96 * summary['se(coef)'])
    p_values = summary['p']
    
    # 排序
    idx = hr.sort_values().index
    hr = hr[idx]
    ci_lower = ci_lower[idx]
    ci_upper = ci_upper[idx]
    p_values = p_values[idx]
    
    y_pos = np.arange(len(hr))
    
    # 绘制森林图
    ax.errorbar(hr, y_pos, xerr=[hr - ci_lower, ci_upper - hr],
                fmt='o', color='#2E86AB', capsize=5, markersize=8)
    
    # 添加参考线
    ax.axvline(x=1, color='red', linestyle='--', alpha=0.7)
    
    # 设置标签
    ax.set_yticks(y_pos)
    ax.set_yticklabels(hr.index)
    ax.set_xlabel('风险比 (HR)', fontsize=12)
    ax.set_title('Cox 比例风险模型 - 森林图', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # 添加P值标注
    for i, (h, p) in enumerate(zip(hr, p_values)):
        sig = ''
        if p < 0.001:
            sig = '***'
        elif p < 0.01:
            sig = '**'
        elif p < 0.05:
            sig = '*'
        ax.text(ci_upper[i] + 0.1, i, sig, va='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Cox森林图已保存: {save_path}")
    
    return fig


def create_survival_nomogram(cph, save_path=None):
    """
    创建生存分析列线图
    
    参数:
        cph: CoxPHFitter 对象
        save_path: 保存路径
    
    返回:
        图对象
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 简化版列线图
    summary = cph.summary.copy()
    features = summary.index.tolist()
    coefficients = summary['coef'].values
    hr = np.exp(coefficients)
    
    # 归一化分数
    max_points = 100
    point_range = np.abs(coefficients)
    point_range = point_range / point_range.max() * max_points
    
    # 绘制
    y_positions = np.linspace(0.9, 0.1, len(features))
    
    for i, (feat, pts, h) in enumerate(zip(features, point_range, hr)):
        ax.barh(y_positions[i], pts, height=0.05, color='#3498DB', alpha=0.7)
        ax.text(-5, y_positions[i], feat, ha='right', va='center', fontsize=10)
        
        # HR标注
        ax.text(pts + 2, y_positions[i], f'HR={h:.2f}', ha='left', va='center', fontsize=9)
    
    ax.set_xlim(-15, 120)
    ax.set_ylim(0, 1)
    ax.set_xlabel('分数', fontsize=12)
    ax.set_title('Cox 模型生存预测列线图', fontsize=14, fontweight='bold')
    ax.set_yticks([])
    ax.grid(True, alpha=0.3, axis='x')
    
    # 添加使用说明
    instructions = """
使用说明:
1. 根据患者特征值找到对应分数
2. 将所有分数相加得到总分
3. 从总分向下读取预测生存概率
    """
    ax.text(0.02, 0.02, instructions, transform=ax.transAxes, 
            fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"生存列线图已保存: {save_path}")
    
    return fig


def create_comprehensive_survival_dashboard(df, output_dir='output'):
    """
    创建综合生存分析仪表板
    
    参数:
        df: DataFrame
        output_dir: 输出目录
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("生存分析综合仪表板")
    print("=" * 60)
    
    # 1. 整体Kaplan-Meier曲线
    print("\n1. 绘制整体生存曲线...")
    fig1 = plot_kaplan_meier(df, save_path=f'{output_dir}/km_overall.png')
    plt.close(fig1)
    
    # 2. 按血铅分组的KM曲线
    print("2. 绘制血铅分组生存曲线...")
    fig2 = plot_kaplan_meier(df, group_col='Lead_Group', 
                            save_path=f'{output_dir}/km_by_lead.png')
    plt.close(fig2)
    
    # 3. Log-rank检验
    print("3. 执行Log-rank检验...")
    logrank_result = logrank_test_groups(df, 'Lead_Group')
    print(f"   Log-rank检验: χ² = {logrank_result['test_statistic']:.3f}")
    print(f"   P值: {logrank_result['p_value']:.4e}")
    
    # 4. Cox模型
    print("4. 拟合Cox比例风险模型...")
    cph = fit_cox_model(df)
    print("\n   Cox模型摘要:")
    print(cph.print_summary(decimals=3))
    
    # 5. Cox森林图
    print("\n5. 绘制Cox森林图...")
    fig5 = plot_cox_summary(cph, save_path=f'{output_dir}/cox_forest.png')
    plt.close(fig5)
    
    # 6. 生存列线图
    print("6. 创建生存列线图...")
    fig6 = create_survival_nomogram(cph, save_path=f'{output_dir}/survival_nomogram.png')
    plt.close(fig6)
    
    # 保存Cox模型结果
    cph.summary.to_csv(f'{output_dir}/cox_results.csv')
    print(f"\n模型结果已保存: {output_dir}/cox_results.csv")
    
    print("\n" + "=" * 60)
    print("生存分析完成!")
    print("=" * 60)
    
    return {
        'logrank_test': logrank_result,
        'cph_model': cph
    }


# 主程序
if __name__ == '__main__':
    # 生成模拟数据
    print("生成模拟生存数据...")
    df = generate_survival_data(n=500)
    
    print(f"\n数据概览:")
    print(f"  样本量: {len(df)}")
    print(f"  事件发生率: {df['Event'].mean()*100:.1f}%")
    print(f"  随访时间范围: {df['Time_Months'].min():.1f} - {df['Time_Months'].max():.1f} 月")
    
    # 执行完整分析
    results = create_comprehensive_survival_dashboard(df, output_dir='output')
    
    print("\n✅ 生存分析模块创建成功!")
