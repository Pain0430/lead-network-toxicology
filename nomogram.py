#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
铅网络毒理学 - 列线图分析模块
Lead Network Toxicology - Nomogram Analysis

功能：
- 逻辑回归模型构建
- 列线图 (Nomogram) 可视化
- 分数计算系统
- 个体化风险预测
- 校准曲线分析

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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, brier_score_loss
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

# 配色方案
COLORS = {
    'primary': '#2C3E50',
    'secondary': '#E74C3C',
    'tertiary': '#3498DB',
    'quaternary': '#27AE60',
    'accent': '#F39C12',
    'purple': '#9B59B6',
    'light': '#ECF0F1',
    'dark': '#2C3E50',
    'grid': '#BDC3C7'
}

warnings.filterwarnings('ignore')

OUTPUT_DIR = '/Users/pengsu/mycode/lead-network-toxicology/output'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_demo_data(n_samples=2000, random_state=42):
    """生成铅毒性模拟数据集用于列线图分析"""
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
        '年龄': np.random.randint(18, 70, n_samples),
        'BMI': np.random.uniform(18, 35, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # 计算铅毒性风险评分（模拟真实关联）
    risk_score = (
        0.25 * (df['血铅_ug_dL'] - 3) / 57 +
        0.15 * (df['尿铅_ug_L'] - 5) / 95 +
        0.20 * (df['MDA_nmol_mL'] - 2) / 8 +
        0.10 * (1 - (df['SOD_U_mL'] - 50) / 150) +
        0.10 * (df['TNF_alpha_pg_mL'] - 5) / 35 +
        0.10 * (df['IL6_pg_mL'] - 2) / 18 +
        0.05 * (df['年龄'] - 18) / 52 +
        0.05 * (df['BMI'] - 18) / 17
    )
    
    # 添加噪声并转换为概率
    risk_prob = 1 / (1 + np.exp(-3 * (risk_score - 0.5 + np.random.normal(0, 0.1, n_samples))))
    df['铅毒性'] = (risk_prob > 0.5).astype(int)
    
    return df


class NomogramBuilder:
    """列线图构建器"""
    
    def __init__(self, model=None, feature_names=None):
        self.model = model
        self.feature_names = feature_names
        self.coefficients = None
        self.intercept = None
        self.scaler = StandardScaler()
        self.points_scale = None
        self.feature_ranges = {}
        
    def fit(self, X, y, scale_points=True):
        """拟合逻辑回归模型并计算列线图参数"""
        # 标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 训练逻辑回归
        if self.model is None:
            self.model = LogisticRegression(
                max_iter=1000, 
                random_state=42,
                solver='lbfgs'
            )
        self.model.fit(X_scaled, y)
        
        # 获取系数
        self.coefficients = self.model.coef_[0]
        self.intercept = self.model.intercept_[0]
        
        if self.feature_names is None:
            self.feature_names = [f'Feature_{i}' for i in range(len(self.coefficients))]
        
        # 计算特征范围
        for i, name in enumerate(self.feature_names):
            self.feature_ranges[name] = {
                'min': X.iloc[:, i].min(),
                'max': X.iloc[:, i].max(),
                'mean': X.iloc[:, i].mean(),
                'std': X.iloc[:, i].std()
            }
        
        # 设置分数范围
        if scale_points:
            self._calculate_points_scale()
        
        return self
    
    def _calculate_points_scale(self):
        """计算分数刻度"""
        # 使用系数的绝对值来确定分数权重
        abs_coefs = np.abs(self.coefficients)
        total_coef = abs_coefs.sum()
        
        # 将每个系数映射到 0-100 分的范围
        self.points_scale = 100 * abs_coefs / total_coef
        
    def get_feature_points(self, feature_name, value):
        """获取特征值对应的分数"""
        if feature_name not in self.feature_ranges:
            return 0
            
        feature_range = self.feature_ranges[feature_name]
        idx = self.feature_names.index(feature_name)
        coef = self.coefficients[idx]
        
        # 标准化值
        standardized = (value - feature_range['mean']) / (feature_range['std'] + 1e-8)
        
        # 计算分数 (使用绝对值)
        points = abs(standardized * coef) * 10
        
        return points
    
    def predict_risk(self, X):
        """预测风险概率"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def get_total_points(self, X):
        """计算总分数"""
        total_points = np.zeros(len(X))
        
        for i, name in enumerate(self.feature_names):
            points = self.get_feature_points(name, X.iloc[:, i].values)
            total_points += points
            
        return total_points
    
    def points_to_risk(self, points):
        """将分数转换为风险概率"""
        # 假设分数范围 0-1000
        # 使用 logistic 函数转换
        # 调整参数以获得合理的风险范围
        return 1 / (1 + np.exp(-(points - 500) / 150))


def create_nomogram(nomogram_builder, X_sample=None, output_path=None):
    """
    创建列线图
    
    Parameters:
    -----------
    nomogram_builder : NomogramBuilder
        列线图构建器
    X_sample : DataFrame, optional
        样本数据用于显示分布
    output_path : str, optional
        输出路径
    """
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    features = nomogram_builder.feature_names
    coefs = nomogram_builder.coefficients
    ranges = nomogram_builder.feature_ranges
    
    # 颜色
    primary_color = COLORS['primary']
    accent_color = COLORS['accent']
    
    # 标题
    ax.text(5, 9.5, '铅毒性风险预测列线图 (Nomogram)', 
            fontsize=16, fontweight='bold', ha='center', color=primary_color)
    ax.text(5, 9.1, 'Lead Toxicity Risk Prediction Nomogram', 
            fontsize=10, ha='center', style='italic', color='gray')
    
    # 绘制每个特征的标尺
    n_features = len(features)
    spacing = 7.5 / (n_features + 1)
    
    point_lines = []  # 存储分数刻度线位置
    
    for i, (feature, coef) in enumerate(zip(features, coefs)):
        y_pos = 8 - i * spacing
        
        # 特征名称
        ax.text(0.3, y_pos, feature, fontsize=9, ha='left', va='center', fontweight='bold')
        
        # 系数方向指示
        direction = '↑' if coef > 0 else '↓'
        ax.text(2.2, y_pos, f'({direction})', fontsize=8, ha='left', va='center', color='gray')
        
        # 绘制标尺线
        ax.plot([2.5, 7.5], [y_pos, y_pos], color='gray', linewidth=0.5, alpha=0.5)
        
        # 刻度值
        if feature in ranges:
            val_range = ranges[feature]
            min_val = val_range['min']
            max_val = val_range['max']
            
            # 最小值刻度
            ax.plot(2.5, y_pos - 0.1, 'k|', markersize=5)
            ax.text(2.5, y_pos - 0.35, f'{min_val:.1f}', fontsize=7, ha='center', va='top')
            
            # 中间刻度
            mid_val = (min_val + max_val) / 2
            ax.plot(5.0, y_pos - 0.08, 'k|', markersize=4)
            ax.text(5.0, y_pos - 0.3, f'{mid_val:.1f}', fontsize=7, ha='center', va='top')
            
            # 最大值刻度
            ax.plot(7.5, y_pos - 0.1, 'k|', markersize=5)
            ax.text(7.5, y_pos - 0.35, f'{max_val:.1f}', fontsize=7, ha='center', va='top')
    
    # Points 标尺
    ax.text(5, 1.5, 'Points', fontsize=11, fontweight='bold', ha='center', color=primary_color)
    ax.plot([2.5, 7.5], [1.2, 1.2], color=primary_color, linewidth=2)
    
    # Points 刻度
    for pts in [0, 25, 50, 75, 100]:
        x_pos = 2.5 + (pts / 100) * 5
        ax.plot(x_pos, 1.2, 'k|', markersize=8)
        ax.text(x_pos, 0.85, str(pts), fontsize=8, ha='center', va='top')
    
    # Total Points 线
    ax.plot([5, 5], [0.6, 1.0], color=accent_color, linewidth=2)
    ax.text(5, 0.4, 'Total Points', fontsize=11, fontweight='bold', ha='center', color=accent_color)
    
    # 风险预测轴
    ax.text(9.2, 4.5, 'Risk', fontsize=11, fontweight='bold', ha='center', rotation=-90, color=COLORS['secondary'])
    ax.plot([8.8, 8.8], [1, 8], color=COLORS['secondary'], linewidth=2)
    
    # 风险刻度
    risk_levels = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    for risk in risk_levels:
        y_norm = risk  # 线性映射
        y_pos = 1 + y_norm * 7
        ax.plot(8.7, y_pos, 'k|', markersize=5)
        ax.text(8.55, y_pos, f'{risk:.0%}', fontsize=7, ha='right', va='center')
    
    # 添加使用说明
    ax.text(0.5, 0.2, '使用方法:', fontsize=9, fontweight='bold')
    ax.text(0.5, 0.05, 
            '1. 找到每个特征值在对应标尺上的位置\n'
            '2. 向上画垂直线找到Points值\n'
            '3. 将所有Points相加得到Total Points\n'
            '4. 从Total Points向下画线得到预测风险', 
            fontsize=7, va='top', linespacing=1.5)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"列线图已保存: {output_path}")
    
    return fig


def create_interactive_nomogram(nomogram_builder, feature_values=None):
    """
    创建交互式列线图 (使用 Plotly)
    
    Parameters:
    -----------
    nomogram_builder : NomogramBuilder
        列线图构建器
    feature_values : dict, optional
        预设的特征值
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    features = nomogram_builder.feature_names
    coefs = nomogram_builder.coefficients
    ranges = nomogram_builder.feature_ranges
    
    fig = go.Figure()
    
    # 为每个特征创建滑块
    if feature_values is None:
        feature_values = {}
        for feat in features:
            if feat in ranges:
                feature_values[feat] = (ranges[feat]['min'] + ranges[feat]['max']) / 2
    
    # 计算当前总分
    total_points = 0
    for feat, val in feature_values.items():
        points = nomogram_builder.get_feature_points(feat, val)
        total_points += points
    
    current_risk = nomogram_builder.points_to_risk(total_points)
    
    # 添加特征滑块
    for i, feat in enumerate(features):
        if feat in ranges:
            val_range = ranges[feat]
            current_val = feature_values.get(feat, val_range['mean'])
            
            fig.add_trace(go.Slider(
                steps=[{
                    'label': f'{val_range["min"]:.1f}',
                    'method': 'update',
                    'args': [{'visible': [True] * len(features)}]
                }],
                currentvalue={'prefix': f'{feat}: '},
                min=val_range['min'],
                max=val_range['max'],
                value=current_val,
                name=feat
            ))
    
    return fig


def create_calibration_curve(y_true, y_prob, n_bins=10, output_path=None):
    """创建校准曲线"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # 计算校准曲线数据
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    
    # 绘制校准曲线
    ax.plot(prob_pred, prob_true, 's-', color=COLORS['tertiary'], 
            linewidth=2, markersize=8, label='Model')
    
    # 绘制理想曲线
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Ideal')
    
    # 填充区域
    ax.fill_between(prob_pred, prob_true, prob_pred, alpha=0.2, color=COLORS['tertiary'])
    
    # 计算 Brier 分数
    brier = brier_score_loss(y_true, y_prob)
    
    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('Observed Probability', fontsize=12)
    ax.set_title(f'Calibration Curve (Brier Score: {brier:.4f})', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"校准曲线已保存: {output_path}")
    
    return fig, brier


def analyze_nomogram_performance(X, y, feature_names=None, output_dir=None):
    """完整的列线图分析流程"""
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 构建列线图
    builder = NomogramBuilder(feature_names=feature_names)
    builder.fit(X_train, y_train)
    
    # 预测
    y_train_prob = builder.predict_risk(X_train)
    y_test_prob = builder.predict_risk(X_test)
    
    # 性能指标
    train_auc = roc_auc_score(y_train, y_train_prob)
    test_auc = roc_auc_score(y_test, y_test_prob)
    test_brier = brier_score_loss(y_test, y_test_prob)
    
    print(f"\n{'='*50}")
    print("列线图模型性能评估")
    print(f"{'='*50}")
    print(f"训练集 AUC: {train_auc:.4f}")
    print(f"测试集 AUC: {test_auc:.4f}")
    print(f"测试集 Brier Score: {test_brier:.4f}")
    print(f"{'='*50}\n")
    
    # 绘制列线图
    nomogram_path = os.path.join(output_dir, 'nomogram.png')
    create_nomogram(builder, output_path=nomogram_path)
    
    # 绘制校准曲线
    calibration_path = os.path.join(output_dir, 'nomogram_calibration.png')
    create_calibration_curve(y_test, y_test_prob, output_path=calibration_path)
    
    return builder, {
        'train_auc': train_auc,
        'test_auc': test_auc,
        'brier_score': test_brier
    }


def add_nomogram_to_streamlit(st, builder, X_columns):
    """为 Streamlit 添加列线图交互界面"""
    st.markdown("### 📏 列线图 (Nomogram)")
    
    st.info("💡 **使用方法**: 调整下方滑块设置患者特征值，系统将计算预测风险分数")
    
    col1, col2 = st.columns([2, 1])
    
    # 特征输入
    feature_values = {}
    with col1:
        st.markdown("#### 患者特征输入")
        
        # 创建两列布局
        c1, c2, c3 = st.columns(3)
        
        cols = [c1, c2, c3]
        for i, col_name in enumerate(X_columns):
            with cols[i % 3]:
                # 获取特征范围
                if hasattr(builder, 'feature_ranges') and col_name in builder.feature_ranges:
                    f_range = builder.feature_ranges[col_name]
                    min_val = f_range['min']
                    max_val = f_range['max']
                    default_val = f_range['mean']
                else:
                    min_val = 0
                    max_val = 100
                    default_val = 50
                
                feature_values[col_name] = st.slider(
                    f"📊 {col_name}",
                    float(min_val), float(max_val), float(default_val),
                    key=f"nomo_{col_name}"
                )
    
    # 计算风险
    with col2:
        st.markdown("#### 🎯 预测结果")
        
        if st.button("计算风险分数", type="primary"):
            # 计算总分数
            total_points = 0
            for feat, val in feature_values.items():
                points = builder.get_feature_points(feat, val)
                total_points += points
            
            # 转换为风险概率
            risk_prob = builder.points_to_risk(total_points)
            
            # 显示结果
            st.metric("总分数", f"{total_points:.1f}")
            st.metric("预测风险", f"{risk_prob:.1%}", delta_color="inverse")
            
            # 风险等级
            if risk_prob > 0.75:
                st.error("⚠️ **高风险** - 建议临床干预")
            elif risk_prob > 0.5:
                st.warning("⚡ **中高风险** - 建议密切监测")
            elif risk_prob > 0.25:
                st.info("📍 **中低风险** - 建议定期随访")
            else:
                st.success("✅ **低风险** - 保持健康生活方式")
            
            # 保存到 session state
            st.session_state['nomogram_risk'] = risk_prob
    
    return feature_values


if __name__ == "__main__":
    # 示例运行
    print("生成示例数据...")
    df = generate_demo_data(2000)
    
    # 准备特征
    feature_cols = ['血铅_ug_dL', '尿铅_ug_L', 'MDA_nmol_mL', 'SOD_U_mL', 
                    'TNF_alpha_pg_mL', 'IL6_pg_mL', '年龄', 'BMI']
    
    X = df[feature_cols]
    y = df['铅毒性']
    
    print("训练列线图模型...")
    builder, metrics = analyze_nomogram_performance(X, y, feature_cols)
    
    print("\n✅ 列线图分析完成!")
    print(f"输出文件: {OUTPUT_DIR}")
