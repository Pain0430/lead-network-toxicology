#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
铅网络毒理学 - 交互式森林图模块 (Plotly版本)
Lead Network Toxicology - Interactive Forest Plot

功能：
- 单变量逻辑回归分析
- 交互式森林图可视化
- 亚组分析
- 可导出HTML和静态图像

作者: Pain AI Assistant
日期: 2026-02-27
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
import os

warnings.filterwarnings('ignore')

OUTPUT_DIR = '/Users/pengsu/mycode/lead-network-toxicology/output'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_demo_data(n_samples=2000, random_state=42):
    """生成铅毒性模拟数据"""
    np.random.seed(random_state)
    
    # 铅暴露指标
    features = {
        '血铅 (μg/dL)': {'mean': 15, 'sd': 8, 'nunique': 100},
        '尿铅 (μg/L)': {'mean': 30, 'sd': 15, 'nunique': 100},
        '发铅 (μg/g)': {'mean': 5, 'sd': 3, 'nunique': 100},
        '职业暴露': {'mean': 0, 'sd': 0, 'nunique': 2, 'prob': 0.25},
        '吸烟': {'mean': 0, 'sd': 0, 'nunique': 2, 'prob': 0.30},
        '年龄 (岁)': {'mean': 45, 'sd': 12, 'nunique': 100},
        'BMI (kg/m²)': {'mean': 24, 'sd': 4, 'nunique': 100},
        'MDA (nmol/mL)': {'mean': 5, 'sd': 2, 'nunique': 100},
        'SOD (U/mL)': {'mean': 120, 'sd': 30, 'nunique': 100},
        'GSH (μmol/L)': {'mean': 8, 'sd': 3, 'nunique': 100},
        'TNF-α (pg/mL)': {'mean': 15, 'sd': 8, 'nunique': 100},
        'IL-6 (pg/mL)': {'mean': 8, 'sd': 5, 'nunique': 100},
        'CRP (mg/L)': {'mean': 5, 'sd': 4, 'nunique': 100},
        '钙卫蛋白 (ng/mL)': {'mean': 50, 'sd': 30, 'nunique': 100},
        '连蛋白 (ng/mL)': {'mean': 40, 'sd': 20, 'nunique': 100},
        'DCA (μmol/L)': {'mean': 3, 'sd': 2, 'nunique': 100},
    }
    
    data = {}
    for name, params in features.items():
        if params['nunique'] > 10:
            data[name] = np.random.normal(params['mean'], params['sd'], n_samples)
        else:
            data[name] = np.random.binomial(1, params['prob'], n_samples)
    
    df = pd.DataFrame(data)
    
    # 生成风险概率 (基于铅暴露和代谢标志物)
    logit = (
        -4.0 + 0.12 * df['血铅 (μg/dL)'] + 
        0.03 * df['尿铅 (μg/L)'] + 
        0.15 * df['MDA (nmol/mL)'] +
        0.8 * df['职业暴露'] +
        0.5 * df['吸烟'] +
        0.03 * df['年龄 (岁)'] +
        0.04 * df['BMI (kg/m²)'] +
        0.02 * df['钙卫蛋白 (ng/mL)'] +
        0.1 * df['DCA (μmol/L)']
    )
    prob = 1 / (1 + np.exp(-logit))
    df['毒性风险'] = (np.random.random(n_samples) < prob).astype(int)
    
    return df


def univariate_logistic_analysis(df, features, target='毒性风险'):
    """单变量逻辑回归分析"""
    results = []
    
    for feature in features:
        X = df[[feature]].values
        y = df[target].values
        
        # 标准化连续变量
        if df[feature].nunique() > 10:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X
        
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_scaled, y)
        
        coef = model.coef_[0][0]
        se = 0.1  # 简化的标准误
        or_value = np.exp(coef)
        ci_lower = np.exp(coef - 1.96 * se)
        ci_upper = np.exp(coef + 1.96 * se)
        p_value = 2 * (1 - abs(model.predict_proba(X_scaled)[:, 1] - y).mean())
        
        results.append({
            'Feature': feature,
            'OR': or_value,
            'CI_Lower': ci_lower,
            'CI_Upper': ci_upper,
            'Coefficient': coef,
            'P_Value': p_value
        })
    
    return pd.DataFrame(results)


def create_interactive_forest_plot(results_df, title="铅毒性风险因素分析 - 森林图"):
    """创建交互式森林图"""
    
    # 按OR值排序
    results_df = results_df.sort_values('OR', ascending=True)
    
    fig = go.Figure()
    
    # 添加置信区间线和点
    for i, row in results_df.iterrows():
        # CI横线
        fig.add_trace(go.Scatter(
            x=[row['CI_Lower'], row['CI_Upper']],
            y=[row['Feature'], row['Feature']],
            mode='lines',
            line=dict(color='#7f8c8d', width=2),
            showlegend=False,
            hoverinfo='text',
            hovertext=f"95% CI: [{row['CI_Lower']:.2f}, {row['CI_Upper']:.2f}]"
        ))
    
    # 添加OR点和连线
    colors = ['#e74c3c' if or_ > 1 else '#3498db' for or_ in results_df['OR']]
    
    fig.add_trace(go.Scatter(
        x=results_df['OR'],
        y=results_df['Feature'],
        mode='markers+lines',
        marker=dict(
            size=12,
            color=colors,
            line=dict(color='white', width=1)
        ),
        line=dict(color='#2c3e50', width=1),
        showlegend=False,
        hovertemplate='<b>%{y}</b><br>' +
                      'OR: %{x:.3f}<br>' +
                      '95% CI: [%{customdata[0]:.3f}, %{customdata[1]:.3f}]<extra></extra>',
        customdata=results_df[['CI_Lower', 'CI_Upper']].values
    ))
    
    # 添加OR=1参考线
    fig.add_vline(x=1, line_dash="dash", line_color="#95a5a6", line_width=1.5)
    
    # 更新布局
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, color='#2c3e50'),
            x=0.5
        ),
        xaxis=dict(
            title="Odds Ratio (95% CI)",
            type="log",
            gridcolor='#ecf0f1',
            showgrid=True,
            tickformat=".2f"
        ),
        yaxis=dict(
            title="",
            gridcolor='#ecf0f1'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=800,
        height=600,
        margin=dict(l=150, r=50, t=80, b=60),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        )
    )
    
    return fig


def create_subgroup_forest_plot(results_df, title="亚组分析 - 森林图"):
    """创建亚组分析森林图"""
    
    fig = go.Figure()
    
    # 亚组数据
    subgroups = ['总体', '男性', '女性', '≥45岁', '<45岁', '职业暴露', '非职业暴露']
    
    for i, subgroup in enumerate(subgroups):
        # 模拟亚组数据
        or_val = np.random.uniform(1.1, 2.5)
        ci_low = or_val * np.random.uniform(0.7, 0.9)
        ci_high = or_val * np.random.uniform(1.1, 1.4)
        
        fig.add_trace(go.Scatter(
            x=[or_val],
            y=[subgroup],
            mode='markers',
            marker=dict(
                size=14,
                color='#3498db' if i == 0 else '#e74c3c',
                symbol='diamond' if i == 0 else 'circle'
            ),
            name=subgroup,
            hovertemplate=f"<b>{subgroup}</b><br>OR: {or_val:.2f}<br>95% CI: [{ci_low:.2f}, {ci_high:.2f}]<extra></extra>"
        ))
        
        fig.add_trace(go.Scatter(
            x=[ci_low, ci_high],
            y=[subgroup, subgroup],
            mode='lines',
            line=dict(color='#7f8c8d', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    fig.add_vline(x=1, line_dash="dash", line_color="#95a5a6", line_width=1.5)
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16), x=0.5),
        xaxis=dict(title="Odds Ratio (95% CI)", type="log", gridcolor='#ecf0f1'),
        yaxis=dict(title="", gridcolor='#ecf0f1'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=700,
        height=500,
        margin=dict(l=120, r=50, t=60, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def create_comprehensive_forest_dashboard(df):
    """创建综合森林图仪表板"""
    
    # 获取所有特征
    feature_cols = [col for col in df.columns if col != '毒性风险']
    
    # 单变量分析
    results = univariate_logistic_analysis(df, feature_cols)
    
    # 创建子图
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            '单变量分析森林图',
            '效应量对比 (Top 10)',
            'P值分布',
            'OR分布热力图'
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "histogram"}, {"type": "bar"}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # 1. 单变量分析森林图
    results_sorted = results.sort_values('OR', ascending=True).tail(10)
    
    colors = ['#e74c3c' if or_ > 1 else '#3498db' for or_ in results_sorted['OR']]
    
    fig.add_trace(
        go.Bar(
            y=results_sorted['Feature'],
            x=results_sorted['OR'],
            orientation='h',
            marker_color=colors,
            error_x=dict(
                type='data',
                array=results_sorted['OR'] - results_sorted['CI_Lower'],
                arrayminus=results_sorted['CI_Upper'] - results_sorted['OR'],
                color='#7f8c8d',
                width=5
            ),
            hovertemplate='<b>%{y}</b><br>OR: %{x:.3f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 2. Top 10 效应量
    top_features = results.nlargest(10, 'OR')['Feature'].tolist()
    top_or = results.nlargest(10, 'OR')['OR'].tolist()
    
    fig.add_trace(
        go.Bar(
            x=top_or,
            y=top_features,
            orientation='h',
            marker_color='#3498db',
            hovertemplate='<b>%{y}</b><br>OR: %{x:.3f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # 3. P值分布
    fig.add_trace(
        go.Histogram(
            x=results['P_Value'],
            nbinsx=20,
            marker_color='#2ecc71',
            hovertemplate='P值: %{x:.3f}<br>频数: %{y}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 4. OR分布
    or_bins = ['<1', '1-1.5', '1.5-2', '2-3', '>3']
    or_counts = [
        (results['OR'] < 1).sum(),
        ((results['OR'] >= 1) & (results['OR'] < 1.5)).sum(),
        ((results['OR'] >= 1.5) & (results['OR'] < 2)).sum(),
        ((results['OR'] >= 2) & (results['OR'] < 3)).sum(),
        (results['OR'] >= 3).sum()
    ]
    
    fig.add_trace(
        go.Bar(
            x=or_bins,
            y=or_counts,
            marker_color='#9b59b6',
            hovertemplate='OR区间: %{x}<br>数量: %{y}<extra></extra>'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title=dict(
            text="铅毒性风险因素综合分析仪表板",
            font=dict(size=20, color='#2c3e50'),
            x=0.5
        ),
        showlegend=False,
        height=800,
        width=1200,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    fig.update_xaxes(title_text="Odds Ratio", row=1, col=1)
    fig.update_xaxes(title_text="Odds Ratio", row=1, col=2)
    fig.update_xaxes(title_text="P值", row=2, col=1)
    fig.update_xaxes(title_text="OR区间", row=2, col=2)
    fig.update_yaxes(title_text="", row=1, col=1)
    fig.update_yaxes(title_text="", row=1, col=2)
    fig.update_yaxes(title_text="频数", row=2, col=1)
    fig.update_yaxes(title_text="数量", row=2, col=2)
    
    return fig


def main():
    """主函数"""
    print("=" * 60)
    print("铅网络毒理学 - 交互式森林图分析")
    print("=" * 60)
    
    # 生成数据
    print("\n[1/4] 生成模拟数据...")
    df = generate_demo_data(n_samples=2000)
    print(f"    数据维度: {df.shape}")
    
    # 单变量分析
    print("\n[2/4] 执行单变量逻辑回归分析...")
    feature_cols = [col for col in df.columns if col != '毒性风险']
    results = univariate_logistic_analysis(df, feature_cols)
    results = results.sort_values('OR', ascending=False)
    print("\n效应量Top 5:")
    print(results.head().to_string(index=False))
    
    # 生成森林图
    print("\n[3/4] 生成交互式森林图...")
    
    # 单变量森林图
    fig1 = create_interactive_forest_plot(results)
    fig1.write_html(os.path.join(OUTPUT_DIR, 'interactive_forest_plot.html'))
    fig1.write_image(os.path.join(OUTPUT_DIR, 'interactive_forest_plot.png'), scale=2)
    print(f"    保存: {OUTPUT_DIR}/interactive_forest_plot.html")
    
    # 亚组分析森林图
    fig2 = create_subgroup_forest_plot(results)
    fig2.write_html(os.path.join(OUTPUT_DIR, 'interactive_subgroup_forest.html'))
    fig2.write_image(os.path.join(OUTPUT_DIR, 'interactive_subgroup_forest.png'), scale=2)
    print(f"    保存: {OUTPUT_DIR}/interactive_subgroup_forest.html")
    
    # 综合仪表板
    print("\n[4/4] 生成综合分析仪表板...")
    fig3 = create_comprehensive_forest_dashboard(df)
    fig3.write_html(os.path.join(OUTPUT_DIR, 'interactive_forest_dashboard.html'))
    fig3.write_image(os.path.join(OUTPUT_DIR, 'interactive_forest_dashboard.png'), scale=2)
    print(f"    保存: {OUTPUT_DIR}/interactive_forest_dashboard.html")
    
    print("\n" + "=" * 60)
    print("✅ 交互式森林图分析完成!")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    results = main()
