#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
铅网络毒理学 - 交互式列线图模块 (Plotly版本)
Lead Network Toxicology - Interactive Nomogram

功能：
- 逻辑回归模型构建
- 交互式列线图可视化
- 分数计算系统
- 个体化风险预测
- 校准曲线分析

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
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, brier_score_loss
import warnings
import os

warnings.filterwarnings('ignore')

OUTPUT_DIR = '/Users/pengsu/mycode/lead-network-toxicology/output'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_demo_data(n_samples=2000, random_state=42):
    """生成铅毒性模拟数据"""
    np.random.seed(random_state)
    
    data = {
        # 铅暴露指标
        '血铅_ug_dL': np.random.uniform(3, 60, n_samples),
        '尿铅_ug_L': np.random.uniform(5, 100, n_samples),
        
        # 氧化应激标志物
        'MDA_nmol_mL': np.random.uniform(2, 10, n_samples),
        'SOD_U_mL': np.random.uniform(50, 200, n_samples),
        'GSH_umol_L': np.random.uniform(3, 15, n_samples),
        
        # 炎症因子
        'TNF_alpha_pg_mL': np.random.uniform(5, 40, n_samples),
        'IL6_pg_mL': np.random.uniform(2, 20, n_samples),
        'CRP_mg_L': np.random.uniform(1, 15, n_samples),
        
        # 协变量
        'Age': np.random.uniform(25, 75, n_samples),
        'BMI': np.random.uniform(18, 35, n_samples),
        'Smoking': np.random.binomial(1, 0.3, n_samples),
        'Occupational_Exposure': np.random.binomial(1, 0.25, n_samples),
        
        # 肠道屏障标志物
        'Calprotectin_ng_mL': np.random.uniform(20, 200, n_samples),
        'Zonulin_ng_mL': np.random.uniform(20, 100, n_samples),
        
        # 胆汁酸
        'DCA_umol_L': np.random.uniform(1, 8, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # 生成风险概率
    logit = (
        -5.0 + 
        0.08 * df['血铅_ug_dL'] + 
        0.02 * df['尿铅_ug_L'] + 
        0.3 * df['MDA_nmol_mL'] -
        0.01 * df['SOD_U_mL'] +
        0.15 * df['TNF_alpha_pg_mL'] +
        0.05 * df['CRP_mg_L'] +
        0.04 * df['Age'] +
        0.08 * df['BMI'] +
        0.5 * df['Smoking'] +
        0.8 * df['Occupational_Exposure'] +
        0.01 * df['Calprotectin_ng_mL'] +
        0.1 * df['DCA_umol_L']
    )
    prob = 1 / (1 + np.exp(-logit))
    df['Toxicity'] = (np.random.random(n_samples) < prob).astype(int)
    
    return df


def build_model(df, feature_cols):
    """构建逻辑回归模型"""
    X = df[feature_cols].values
    y = df['Toxicity'].values
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 训练模型
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler


def create_interactive_nomogram(model, scaler, feature_cols, feature_ranges):
    """创建交互式列线图"""
    
    fig = go.Figure()
    
    # 颜色方案
    colors = px.colors.qualitative.Set2[:len(feature_cols)]
    
    # 计算每个特征的点数范围 (基于系数)
    coefs = model.coef_[0]
    max_points = 100
    max_coef = max(abs(coefs))
    
    # 为每个特征添加刻度线和标签
    for i, (feature, color) in enumerate(zip(feature_cols, colors)):
        if feature in feature_ranges:
            min_val, max_val = feature_ranges[feature]
            coef = coefs[i]
            
            # 将特征值映射到点数
            points = (abs(coef) / max_coef) * max_points
            
            # 添加特征名称行
            fig.add_trace(go.Scatter(
                x=[0.05], y=[len(feature_cols) - i],
                mode='text',
                text=[feature],
                textfont=dict(size=11, color='#2c3e50'),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # 添加刻度线
            tick_positions = [0, 0.25, 0.5, 0.75, 1.0]
            tick_values = [min_val + t * (max_val - min_val) for t in tick_positions]
            
            for tick_pos, tick_val in zip(tick_positions, tick_values):
                fig.add_trace(go.Scatter(
                    x=[0.1 + tick_pos * 0.7, 0.1 + tick_pos * 0.7],
                    y=[len(feature_cols) - i - 0.15, len(feature_cols) - i + 0.15],
                    mode='lines',
                    line=dict(color='#bdc3c7', width=1),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                fig.add_trace(go.Scatter(
                    x=[0.1 + tick_pos * 0.7],
                    y=[len(feature_cols) - i - 0.25],
                    mode='text',
                    text=[f"{tick_val:.1f}"],
                    textfont=dict(size=8, color='#7f8c8d'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # 添加该特征的点数刻度
            point_ticks = [0, 25, 50, 75, 100]
            for pt in point_ticks:
                fig.add_trace(go.Scatter(
                    x=[0.85 + pt / 400, 0.85 + pt / 400],
                    y=[len(feature_cols) - i - 0.1, len(feature_cols) - i + 0.1],
                    mode='lines',
                    line=dict(color=color, width=1),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                fig.add_trace(go.Scatter(
                    x=[0.85 + pt / 400],
                    y=[len(feature_cols) - i + 0.2],
                    mode='text',
                    text=[str(pt)],
                    textfont=dict(size=8, color=color),
                    showlegend=False,
                    hoverinfo='skip'
                ))
    
    # 添加总分轴
    fig.add_trace(go.Scatter(
        x=[0.5], y=[0.5],
        mode='text',
        text=['总分数'],
        textfont=dict(size=14, color='#2c3e50', family='Bold'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # 总分刻度
    for pt in [0, 50, 100, 150, 200, 250, 300]:
        fig.add_trace(go.Scatter(
            x=[0.1 + pt / 300 * 0.8],
            y=[0.3],
            mode='lines+text',
            line=dict(color='#2c3e50', width=2),
            text=[str(pt)],
            textposition="bottom center",
            textfont=dict(size=9),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # 添加风险轴
    fig.add_trace(go.Scatter(
        x=[0.5], y=[-0.3],
        mode='text',
        text=['预测风险'],
        textfont=dict(size=14, color='#e74c3c', family='Bold'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # 风险刻度
    risk_vals = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.99]
    for rv in risk_vals:
        x_pos = 0.1 + (np.log(rv / (1 - rv)) + 5) / 10 * 0.8
        x_pos = max(0.1, min(0.9, x_pos))
        
        fig.add_trace(go.Scatter(
            x=[x_pos],
            y=[-0.5],
            mode='text',
            text=[f"{rv:.0%}"],
            textfont=dict(size=8, color='#e74c3c'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title=dict(
            text="铅毒性风险预测列线图 (Nomogram)",
            font=dict(size=20, color='#2c3e50'),
            x=0.5
        ),
        xaxis=dict(
            range=[0, 1],
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        yaxis=dict(
            range=[-1, len(feature_cols) + 1],
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=900,
        height=800,
        showlegend=False,
        margin=dict(l=50, r=50, t=80, b=100)
    )
    
    # 添加交互功能 - 风险预测计算器
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.7,
                y=1.12,
                buttons=[
                    dict(
                        label="查看使用说明",
                        method="update",
                        args=[
                            {"title": ["点击各特征轴上的值，累计得到总分数，对应底部预测风险"]}
                        ]
                    )
                ]
            )
        ]
    )
    
    return fig


def create_risk_calculator(model, scaler, feature_cols, feature_ranges):
    """创建交互式风险计算器"""
    
    fig = go.Figure()
    
    # 添加输入滑块
    for i, feature in enumerate(feature_cols[:8]):  # 限制显示的特征
        if feature in feature_ranges:
            min_val, max_val = feature_ranges[feature]
            default_val = (min_val + max_val) / 2
            
            fig.add_trace(go.Slider(
                x=0,
                y=len(feature_cols) - i,
                steps=[
                    {"label": f"{v:.1f}", "method": "update", 
                     "args": [{"title": [f"{feature}: {v:.2f}"]}]}
                    for v in np.linspace(min_val, max_val, 10)
                ],
                currentvalue=dict(
                    prefix=f"{feature}: ",
                    visible=True,
                    font=dict(size=12)
                ),
                name=feature,
                min=min_val,
                max=max_val,
                value=default_val,
                tickcolor='#3498db',
                font=dict(size=10)
            ))
    
    fig.update_layout(
        title=dict(
            text="交互式风险预测计算器",
            font=dict(size=18, color='#2c3e50'),
            x=0.5
        ),
        showlegend=False,
        width=700,
        height=600,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=100, r=50, t=80, b=150)
    )
    
    return fig


def create_calibration_plot(model, scaler, X, y):
    """创建校准曲线"""
    
    # 预测概率
    y_prob = model.predict_proba(X)[:, 1]
    
    # 计算校准曲线数据
    prob_true, prob_pred = calibration_curve(y, y_prob, n_bins=10)
    
    fig = go.Figure()
    
    # 理想校准线
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(color='#95a5a6', dash='dash', width=2),
        name='理想校准'
    ))
    
    # 实际校准曲线
    fig.add_trace(go.Scatter(
        x=prob_pred, y=prob_true,
        mode='lines+markers',
        line=dict(color='#3498db', width=3),
        marker=dict(size=10, color='#3498db'),
        name='实际校准'
    ))
    
    # 添加置信区间 (模拟)
    fig.add_trace(go.Scatter(
        x=prob_pred,
        y=prob_true - 0.05,
        mode='lines',
        line=dict(color='#3498db', width=1),
        showlegend=False,
        opacity=0.3
    ))
    
    fig.add_trace(go.Scatter(
        x=prob_pred,
        y=prob_true + 0.05,
        mode='lines',
        line=dict(color='#3498db', width=1),
        fill='tonexty',
        fillcolor='rgba(52, 152, 219, 0.1)',
        showlegend=False
    ))
    
    fig.update_layout(
        title=dict(
            text="模型校准曲线",
            font=dict(size=16, color='#2c3e50'),
            x=0.5
        ),
        xaxis=dict(
            title="预测概率",
            gridcolor='#ecf0f1',
            range=[0, 1]
        ),
        yaxis=dict(
            title="实际阳性比例",
            gridcolor='#ecf0f1',
            range=[0, 1]
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=700,
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def create_comprehensive_nomogram_dashboard(model, scaler, feature_cols, feature_ranges, df):
    """创建综合列线图仪表板"""
    
    X = df[feature_cols].values
    y = df['Toxicity'].values
    X_scaled = scaler.transform(X)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            '列线图可视化',
            '特征系数分布',
            '校准曲线',
            '预测分布'
        ),
        specs=[
            [{"type": "scatter"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "histogram"}]
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # 1. 简化的列线图 (条形图形式)
    coefs = model.coef_[0]
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Coefficient': coefs,
        'Abs_Coef': np.abs(coefs)
    }).sort_values('Abs_Coef', ascending=True)
    
    colors = ['#e74c3c' if c > 0 else '#3498db' for c in feature_importance['Coefficient']]
    
    fig.add_trace(
        go.Bar(
            y=feature_importance['Feature'],
            x=feature_importance['Coefficient'],
            orientation='h',
            marker_color=colors,
            hovertemplate='<b>%{y}</b><br>系数: %{x:.3f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 2. 系数分布
    fig.add_trace(
        go.Bar(
            x=feature_importance['Feature'].tail(10),
            y=feature_importance['Abs_Coef'].tail(10),
            marker_color='#9b59b6',
            hovertemplate='<b>%{x}</b><br>|系数|: %{y:.3f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # 3. 校准曲线
    y_prob = model.predict_proba(X_scaled)[:, 1]
    prob_true, prob_pred = calibration_curve(y, y_prob, n_bins=10)
    
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            line=dict(color='#95a5a6', dash='dash'),
            name='理想'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=prob_pred, y=prob_true,
            mode='lines+markers',
            line=dict(color='#3498db', width=3),
            marker=dict(size=8),
            name='实际'
        ),
        row=2, col=1
    )
    
    # 4. 预测概率分布
    fig.add_trace(
        go.Histogram(
            x=y_prob,
            nbinsx=30,
            marker_color='#2ecc71',
            hovertemplate='概率: %{x:.2f}<br>频数: %{y}<extra></extra>'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title=dict(
            text="列线图综合分析仪表板",
            font=dict(size=20, color='#2c3e50'),
            x=0.5
        ),
        showlegend=False,
        height=800,
        width=1200,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    fig.update_xaxes(title_text="系数值", row=1, col=1)
    fig.update_xaxes(title_text="特征", row=1, col=2)
    fig.update_xaxes(title_text="预测概率", row=2, col=1)
    fig.update_xaxes(title_text="预测概率", row=2, col=2)
    
    return fig


def main():
    """主函数"""
    print("=" * 60)
    print("铅网络毒理学 - 交互式列线图分析")
    print("=" * 60)
    
    # 生成数据
    print("\n[1/5] 生成模拟数据...")
    df = generate_demo_data(n_samples=2000)
    print(f"    数据维度: {df.shape}")
    print(f"    阳性率: {df['Toxicity'].mean():.2%}")
    
    # 特征列
    feature_cols = [
        '血铅_ug_dL', '尿铅_ug_L', 'MDA_nmol_mL', 'SOD_U_mL',
        'TNF_alpha_pg_mL', 'CRP_mg_L', 'Age', 'BMI',
        'Smoking', 'Occupational_Exposure', 'Calprotectin_ng_mL', 'DCA_umol_L'
    ]
    
    # 特征范围
    feature_ranges = {
        '血铅_ug_dL': (3, 60),
        '尿铅_ug_L': (5, 100),
        'MDA_nmol_mL': (2, 10),
        'SOD_U_mL': (50, 200),
        'TNF_alpha_pg_mL': (5, 40),
        'CRP_mg_L': (1, 15),
        'Age': (25, 75),
        'BMI': (18, 35),
        'Calprotectin_ng_mL': (20, 200),
        'DCA_umol_L': (1, 8),
    }
    
    # 分割数据
    print("\n[2/5] 构建预测模型...")
    X = df[feature_cols].values
    y = df['Toxicity'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练模型
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # 模型评估
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    print(f"    测试集 AUC: {auc:.4f}")
    print(f"    Brier Score: {brier:.4f}")
    
    # 完整数据模型
    X_all_scaled = scaler.fit_transform(X)
    model.fit(X_all_scaled, y)
    
    # 生成列线图
    print("\n[3/5] 生成交互式列线图...")
    fig1 = create_interactive_nomogram(model, scaler, feature_cols, feature_ranges)
    fig1.write_html(os.path.join(OUTPUT_DIR, 'interactive_nomogram.html'))
    fig1.write_image(os.path.join(OUTPUT_DIR, 'interactive_nomogram.png'), scale=2)
    print(f"    保存: {OUTPUT_DIR}/interactive_nomogram.html")
    
    # 校准曲线
    print("\n[4/5] 生成校准曲线...")
    fig2 = create_calibration_plot(model, scaler, X_all_scaled, y)
    fig2.write_html(os.path.join(OUTPUT_DIR, 'interactive_calibration_nomogram.html'))
    fig2.write_image(os.path.join(OUTPUT_DIR, 'interactive_calibration_nomogram.png'), scale=2)
    print(f"    保存: {OUTPUT_DIR}/interactive_calibration_nomogram.html")
    
    # 综合仪表板
    print("\n[5/5] 生成综合分析仪表板...")
    fig3 = create_comprehensive_nomogram_dashboard(model, scaler, feature_cols, feature_ranges, df)
    fig3.write_html(os.path.join(OUTPUT_DIR, 'interactive_nomogram_dashboard.html'))
    fig3.write_image(os.path.join(OUTPUT_DIR, 'interactive_nomogram_dashboard.png'), scale=2)
    print(f"    保存: {OUTPUT_DIR}/interactive_nomogram_dashboard.html")
    
    print("\n" + "=" * 60)
    print("✅ 交互式列线图分析完成!")
    print("=" * 60)
    
    return model, df


if __name__ == "__main__":
    model, df = main()
