#!/usr/bin/env python3
"""
铅网络毒理学 - Streamlit Web 应用
Lead Network Toxicology - Streamlit Web Application

功能：
1. 数据上传与管理
2. 交互式模型训练
3. 多模型性能对比
4. SHAP特征解释
5. 临床决策曲线
6. 风险分层工具
7. 报告生成

作者: Pain (重庆医科大学)
日期: 2026-02-26
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, brier_score_loss,
    roc_auc_score, f1_score, accuracy_score, recall_score, precision_score
)
import shap
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 页面配置
# ============================================================

st.set_page_config(
    page_title="铅网络毒理学分析平台",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #34495e;
        padding: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .warning-box {
        padding: 1rem;
        background-color: #fff3cd;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .info-box {
        padding: 1rem;
        background-color: #d1ecf1;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 工具函数
# ============================================================

@st.cache_data
def generate_demo_data(n_samples=2000, random_state=42):
    """生成演示数据集"""
    np.random.seed(random_state)
    
    data = {
        '年龄': np.random.normal(45, 15, n_samples).clip(18, 80),
        '性别': np.random.binomial(1, 0.5, n_samples),
        'BMI': np.random.normal(25, 4, n_samples).clip(15, 45),
        '血铅_ug_dL': np.random.lognormal(2.5, 0.8, n_samples).clip(1, 80),
        '尿铅_ug_L': np.random.lognormal(3.0, 0.9, n_samples).clip(5, 200),
        '发铅_ug_g': np.random.lognormal(1.5, 1.0, n_samples).clip(0.5, 50),
        'SOD_U_mL': np.random.normal(120, 25, n_samples),
        'GSH_umol_L': np.random.normal(8, 2, n_samples),
        'MDA_umol_L': np.random.lognormal(1.2, 0.5, n_samples),
        '8_OHdG_ng_mL': np.random.lognormal(1.5, 0.6, n_samples),
        'CRP_mg_L': np.random.lognormal(1.0, 1.2, n_samples).clip(0.1, 50),
        'IL6_pg_mL': np.random.lognormal(2.0, 0.8, n_samples).clip(1, 100),
        'TNF_alpha_pg_mL': np.random.lognormal(1.8, 0.7, n_samples).clip(5, 80),
        'ALT_U_L': np.random.normal(25, 10, n_samples).clip(5, 200),
        'AST_U_L': np.random.normal(28, 12, n_samples).clip(10, 250),
        '肌酐_umol_L': np.random.normal(80, 20, n_samples).clip(30, 200),
        '尿素氮_mmol_L': np.random.normal(5, 1.5, n_samples).clip(2, 20),
        '收缩压_mmHg': np.random.normal(130, 20, n_samples).clip(90, 200),
        '舒张压_mmHg': np.random.normal(82, 12, n_samples).clip(60, 120),
        '糖化血红蛋白_percent': np.random.normal(5.5, 1.0, n_samples).clip(4.0, 12.0),
        '总胆固醇_mmol_L': np.random.normal(5.2, 1.2, n_samples).clip(2.5, 10),
        'DCA_umol_L': np.random.lognormal(2.0, 0.7, n_samples).clip(0.5, 50),
        'LCA_umol_L': np.random.lognormal(1.0, 0.8, n_samples).clip(0.1, 30),
        '胆酸_umol_L': np.random.lognormal(2.5, 0.6, n_samples).clip(1, 60),
        '熊去氧胆酸_umol_L': np.random.lognormal(1.5, 0.7, n_samples).clip(0.5, 40),
        '钙卫蛋白_ug_g': np.random.lognormal(2.0, 1.0, n_samples).clip(10, 500),
        '连蛋白_ng_mL': np.random.lognormal(1.0, 0.5, n_samples).clip(5, 100),
        'LBP_ug_mL': np.random.normal(15, 5, n_samples).clip(3, 50),
    }
    
    df = pd.DataFrame(data)
    
    # 生成目标变量
    risk_score = (
        0.4 * (df['血铅_ug_dL'] / 20) +
        0.3 * (df['尿铅_ug_L'] / 100) +
        0.3 * (df['MDA_umol_L'] / 5) +
        0.2 * (df['CRP_mg_L'] / 10) -
        0.2 * (df['SOD_U_mL'] / 150) +
        0.15 * (df['DCA_umol_L'] / 20) +
        0.1 * (df['年龄'] / 50)
    )
    
    df['铅毒性'] = (risk_score > np.percentile(risk_score, 75)).astype(int)
    
    return df

def calculate_dca(y_true, y_prob, thresholds=None):
    """计算决策曲线分析数据"""
    if thresholds is None:
        thresholds = np.linspace(0, 1, 100)
    
    net_benefits = []
    for threshold in thresholds:
        if threshold == 0:
            net_benefits.append(0)
            continue
        
        pos_rate = np.mean(y_true)
        net_benefit = (np.mean(y_true == 1) * (1 - threshold) / threshold) - (1 - pos_rate)
        
        tp = np.sum((y_prob >= threshold) & (y_true == 1))
        fp = np.sum((y_prob >= threshold) & (y_true == 0))
        
        net_benefit = (tp / len(y_true)) - (fp / len(y_true)) * (threshold / (1 - threshold))
        net_benefits.append(net_benefit)
    
    # All treatment strategy
    all_benefit = pos_rate - (1 - pos_rate) * thresholds / (1 - thresholds)
    
    return thresholds, np.array(net_benefits), all_benefit

def train_model(X_train, y_train, model_type, random_state=42):
    """训练模型"""
    if model_type == 'Logistic Regression':
        model = LogisticRegression(max_iter=1000, random_state=random_state, class_weight='balanced')
        use_scaled = True
    elif model_type == 'Random Forest':
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=random_state, class_weight='balanced')
        use_scaled = False
    elif model_type == 'Gradient Boosting':
        model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=random_state)
        use_scaled = False
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model, use_scaled

# ============================================================
# 侧边栏
# ============================================================

st.sidebar.title("🧪 铅网络毒理学")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "选择功能模块",
    ["📊 数据概览", "🔬 模型训练", "📈 性能评估", "🎯 SHAP分析", 
     "📉 决策曲线", "🔮 风险预测", "📏 列线图", "🌲 森林图", "⏱️ 生存分析", "📋 报告生成"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**铅网络毒理学分析平台**

基于机器学习的铅毒性风险预测与生物标志物分析工具

- 多模型对比
- SHAP解释
- 临床决策支持
""")

# ============================================================
# 主页面
# ============================================================

if page == "📊 数据概览":
    st.markdown('<p class="main-header">🧪 铅网络毒理学分析平台</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # 数据来源选择
    data_source = st.radio("选择数据来源", ["使用演示数据", "上传数据文件"], horizontal=True)
    
    if data_source == "使用演示数据":
        df = generate_demo_data()
        st.success("✅ 已加载演示数据")
    else:
        uploaded_file = st.file_uploader("上传CSV文件", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            st.warning("请上传CSV文件")
            st.stop()
    
    # 数据基本信息
    st.markdown("### 📋 数据基本信息")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("样本数量", f"{len(df):,}")
    with col2:
        st.metric("特征数量", f"{len(df.columns) - 1}")
    with col3:
        st.metric("阳性样本", f"{df['铅毒性'].sum():,}")
    with col4:
        st.metric("阳性率", f"{df['铅毒性'].mean()*100:.1f}%")
    
    # 数据预览
    st.markdown("### 👀 数据预览")
    st.dataframe(df.head(10), use_container_width=True)
    
    # 描述性统计
    st.markdown("### 📊 描述性统计")
    st.dataframe(df.describe().T, use_container_width=True)
    
    # 缺失值检查
    st.markdown("### ❌ 缺失值检查")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        st.success("✅ 数据无缺失值")
    else:
        st.dataframe(missing[missing > 0])
    
    # 目标变量分布
    st.markdown("### 🎯 目标变量分布")
    fig = px.pie(df, names='铅毒性', title='铅毒性分布', 
                 color_discrete_sequence=['#3498db', '#e74c3c'])
    st.plotly_chart(fig, use_container_width=True)
    
    # 特征分布
    st.markdown("### 📊 特征分布")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != '铅毒性']
    
    selected_features = st.multiselect("选择特征", numeric_cols, default=numeric_cols[:6])
    
    if selected_features:
        cols = st.columns(3)
        for i, feature in enumerate(selected_features):
            with cols[i % 3]:
                fig = px.histogram(df, x=feature, color='铅毒性', 
                                   title=feature, barmode='overlay')
                st.plotly_chart(fig, use_container_width=True)

elif page == "🔬 模型训练":
    st.markdown('<p class="sub-header">🔬 模型训练</p>', unsafe_allow_html=True)
    
    # 加载数据
    df = generate_demo_data()
    
    # 特征选择
    st.markdown("### 特征选择")
    target_col = st.selectbox("选择目标变量", df.columns, index=df.columns.tolist().index('铅毒性'))
    
    feature_cols = st.multiselect(
        "选择特征变量",
        [c for c in df.columns if c != target_col],
        default=[c for c in df.columns if c != target_col]
    )
    
    if not feature_cols:
        st.error("请至少选择一个特征")
        st.stop()
    
    # 数据分割
    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("测试集比例", 0.1, 0.5, 0.3)
    with col2:
        random_state = st.number_input("随机种子", 42, 999, 42)
    
    # 准备数据
    X = df[feature_cols]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    st.markdown("### 模型选择")
    model_types = st.multiselect(
        "选择模型",
        ['Logistic Regression', 'Random Forest', 'Gradient Boosting'],
        default=['Logistic Regression', 'Random Forest', 'Gradient Boosting']
    )
    
    if st.button("🚀 训练模型", type="primary"):
        models = {}
        metrics = []
        
        progress_bar = st.progress(0)
        
        for i, model_type in enumerate(model_types):
            with st.spinner(f"正在训练 {model_type}..."):
                model, use_scaled = train_model(
                    X_train, y_train, model_type, random_state
                )
                
                X_tr = X_train_scaled if use_scaled else X_train
                X_te = X_test_scaled if use_scaled else X_test
                
                model.fit(X_tr, y_train)
                y_prob = model.predict_proba(X_te)[:, 1]
                y_pred = model.predict(X_te)
                
                # 计算指标
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                precision, recall, _ = precision_recall_curve(y_test, y_prob)
                pr_auc = average_precision_score(y_test, y_prob)
                
                metrics.append({
                    'Model': model_type,
                    'ROC-AUC': roc_auc,
                    'PR-AUC': pr_auc,
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'F1': f1_score(y_test, y_pred),
                    'Precision': precision_score(y_test, y_pred, zero_division=0),
                    'Recall': recall_score(y_test, y_pred),
                    'Brier': brier_score_loss(y_test, y_prob)
                })
                
                models[model_type] = {
                    'model': model,
                    'use_scaled': use_scaled,
                    'y_prob': y_prob,
                    'y_pred': y_pred,
                    'roc_auc': roc_auc,
                    'pr_auc': pr_auc
                }
            
            progress_bar.progress((i + 1) / len(model_types))
        
        st.session_state['trained_models'] = models
        st.session_state['metrics'] = pd.DataFrame(metrics)
        st.session_state['X_test'] = X_test
        st.session_state['X_test_scaled'] = X_test_scaled
        st.session_state['y_test'] = y_test
        st.session_state['scaler'] = scaler
        st.session_state['feature_cols'] = feature_cols
        
        st.success("✅ 模型训练完成!")
        
        # 显示指标
        st.markdown("### 📊 模型性能指标")
        st.dataframe(st.session_state['metrics'].style.background_gradient(
            cmap='RdYlGn', subset=['ROC-AUC', 'PR-AUC', 'Accuracy', 'F1']
        ), use_container_width=True)

elif page == "📈 性能评估":
    st.markdown('<p class="sub-header">📈 模型性能评估</p>', unsafe_allow_html=True)
    
    if 'trained_models' not in st.session_state:
        st.warning("请先在模型训练页面训练模型")
        if st.button("去训练模型"):
            st.rerun()
        st.stop()
    
    models = st.session_state['trained_models']
    y_test = st.session_state['y_test']
    
    # 图表类型选择
    chart_type = st.selectbox(
        "选择图表类型",
        ["ROC曲线", "PR曲线", "混淆矩阵", "特征重要性", "模型对比"]
    )
    
    if chart_type == "ROC曲线":
        fig = go.Figure()
        fig.add_shape(type='line', x0=0, y0=0, x1=1, y1=1,
                     line=dict(dash='dash', color='gray'))
        
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
        
        for i, (name, data) in enumerate(models.items()):
            fpr, tpr, _ = roc_curve(y_test, data['y_prob'])
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f"{name} (AUC={data['roc_auc']:.3f})",
                line=dict(color=colors[i % len(colors)], width=2)
            ))
        
        fig.update_layout(
            title='ROC曲线对比',
            xaxis_title='假阳性率 (False Positive Rate)',
            yaxis_title='真阳性率 (True Positive Rate)',
            hovermode='closest',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
        
    elif chart_type == "PR曲线":
        fig = go.Figure()
        
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
        
        for i, (name, data) in enumerate(models.items()):
            precision, recall, _ = precision_recall_curve(y_test, data['y_prob'])
            fig.add_trace(go.Scatter(
                x=recall, y=precision,
                mode='lines',
                name=f"{name} (AUC={data['pr_auc']:.3f})",
                line=dict(color=colors[i % len(colors)], width=2)
            ))
        
        fig.update_layout(
            title='PR曲线对比',
            xaxis_title='召回率 (Recall)',
            yaxis_title='精确率 (Precision)',
            hovermode='closest',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
        
    elif chart_type == "混淆矩阵":
        model_name = st.selectbox("选择模型", list(models.keys()))
        y_pred = models[model_name]['y_pred']
        
        cm = confusion_matrix(y_test, y_pred)
        
        fig = px.imshow(
            cm,
            text_auto=True,
            labels=dict(x="预测值", y="实际值", color="数量"),
            x=['阴性', '阳性'],
            y=['阴性', '阳性'],
            color_continuous_scale='Blues'
        )
        fig.update_layout(title=f'{model_name} 混淆矩阵')
        st.plotly_chart(fig, use_container_width=True)
        
    elif chart_type == "特征重要性":
        model_name = st.selectbox("选择模型", list(models.keys()))
        
        if model_name == 'Logistic Regression':
            coef = np.abs(models[model_name]['model'].coef_[0])
            importance = pd.DataFrame({
                'feature': st.session_state['feature_cols'],
                'importance': coef
            }).sort_values('importance', ascending=True)
            
            fig = px.barh(importance, x='importance', y='feature',
                         title='Logistic Regression 系数',
                         color='importance', color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
        else:
            importance = pd.DataFrame({
                'feature': st.session_state['feature_cols'],
                'importance': models[model_name]['model'].feature_importances_
            }).sort_values('importance', ascending=True)
            
            fig = px.barh(importance, x='importance', y='feature',
                         title=f'{model_name} 特征重要性',
                         color='importance', color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "模型对比":
        metrics_df = st.session_state['metrics']
        
        # 多指标对比
        metrics_to_show = st.multiselect(
            "选择指标",
            ['ROC-AUC', 'PR-AUC', 'Accuracy', 'F1', 'Precision', 'Recall'],
            default=['ROC-AUC', 'PR-AUC']
        )
        
        fig = go.Figure()
        
        for metric in metrics_to_show:
            fig.add_trace(go.Bar(
                name=metric,
                x=metrics_df['Model'],
                y=metrics_df[metric]
            ))
        
        fig.update_layout(
            barmode='group',
            title='多模型性能对比',
            yaxis_title='分数',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == "🎯 SHAP分析":
    st.markdown('<p class="sub-header">🎯 SHAP特征解释</p>', unsafe_allow_html=True)
    
    if 'trained_models' not in st.session_state:
        st.warning("请先训练模型")
        st.stop()
    
    model_name = st.selectbox("选择模型", list(st.session_state['trained_models'].keys()))
    
    if st.button("计算SHAP值"):
        with st.spinner("正在计算SHAP值..."):
            model_data = st.session_state['trained_models'][model_name]
            model = model_data['model']
            
            if model_name == 'Logistic Regression':
                X_test = st.session_state['X_test_scaled']
            else:
                X_test = st.session_state['X_test']
            
            # 使用KernelExplainer作为通用方法
            if model_name == 'Logistic Regression':
                # 使用线性解释器
                explainer = shap.LinearExplainer(model, X_test)
            else:
                # 使用TreeExplainer
                explainer = shap.TreeExplainer(model)
            
            shap_values = explainer.shap_values(X_test)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            st.session_state['shap_values'] = shap_values
            st.session_state['shap_explainer'] = explainer
        
        st.success("✅ SHAP值计算完成")
    
    if 'shap_values' in st.session_state:
        shap_values = st.session_state['shap_values']
        feature_cols = st.session_state['feature_cols']
        
        # SHAP Summary Plot
        st.markdown("### SHAP Summary Plot")
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, st.session_state['X_test'], 
                         feature_names=feature_cols, show=False)
        st.pyplot(fig)
        
        # SHAP Bar Plot
        st.markdown("### SHAP特征重要性")
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, st.session_state['X_test'],
                         feature_names=feature_cols, plot_type='bar', show=False)
        st.pyplot(fig)

elif page == "📉 决策曲线":
    st.markdown('<p class="sub-header">📉 决策曲线分析 (DCA)</p>', unsafe_allow_html=True)
    
    if 'trained_models' not in st.session_state:
        st.warning("请先训练模型")
        st.stop()
    
    models = st.session_state['trained_models']
    y_test = st.session_state['y_test']
    
    fig = go.Figure()
    
    # 绘制所有治疗和全部不治疗曲线
    thresholds = np.linspace(0, 1, 100)
    all_benefit = np.mean(y_test) - (1 - np.mean(y_test)) * thresholds / (1 - thresholds + 0.001)
    
    fig.add_trace(go.Scatter(
        x=thresholds, y=np.zeros_like(thresholds),
        mode='lines', name='全部不治疗',
        line=dict(dash='dash', color='gray')
    ))
    
    fig.add_trace(go.Scatter(
        x=thresholds, y=all_benefit,
        mode='lines', name='全部治疗',
        line=dict(dash='dash', color='black')
    ))
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    
    for i, (name, data) in enumerate(models.items()):
        y_prob = data['y_prob']
        
        net_benefits = []
        for threshold in thresholds:
            if threshold == 0:
                net_benefits.append(0)
            else:
                tp = np.sum((y_prob >= threshold) & (y_test == 1))
                fp = np.sum((y_prob >= threshold) & (y_test == 0))
                net_benefit = (tp / len(y_test)) - (fp / len(y_test)) * (threshold / (1 - threshold))
                net_benefits.append(net_benefit)
        
        fig.add_trace(go.Scatter(
            x=thresholds, y=net_benefits,
            mode='lines', name=name,
            line=dict(color=colors[i % len(colors)], width=2)
        ))
    
    fig.update_layout(
        title='决策曲线分析 (DCA)',
        xaxis_title='阈值概率',
        yaxis_title='净收益',
        template='plotly_white',
        hovermode='x'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("💡 决策曲线分析帮助评估不同阈值下使用模型的临床净收益")

elif page == "🔮 风险预测":
    st.markdown('<p class="sub-header">🔮 临床风险预测</p>', unsafe_allow_html=True)
    
    if 'trained_models' not in st.session_state:
        st.warning("请先训练模型")
        st.stop()
    
    models = st.session_state['trained_models']
    scaler = st.session_state['scaler']
    feature_cols = st.session_state['feature_cols']
    
    st.markdown("### 输入患者特征")
    
    # 创建输入表单
    col1, col2, col3 = st.columns(3)
    
    input_values = {}
    
    for i, feature in enumerate(feature_cols):
        with [col1, col2, col3][i % 3]:
            default_val = 0
            if '年龄' in feature:
                default_val = 45
            elif 'BMI' in feature:
                default_val = 25
            elif '血铅' in feature:
                default_val = 10
            
            input_values[feature] = st.number_input(
                feature,
                value=float(default_val),
                step=0.1
            )
    
    if st.button("🔮 预测风险", type="primary"):
        # 准备输入数据
        input_df = pd.DataFrame([input_values])
        
        results = {}
        
        for name, model_data in models.items():
            model = model_data['model']
            
            if model_data['use_scaled']:
                input_scaled = scaler.transform(input_df)
                prob = model.predict_proba(input_scaled)[0, 1]
            else:
                prob = model.predict_proba(input_df)[0, 1]
            
            results[name] = prob
        
        st.markdown("### 🎯 预测结果")
        
        # 显示风险等级
        avg_risk = np.mean(list(results.values()))
        
        if avg_risk < 0.25:
            risk_level = "🟢 低风险"
            risk_color = "success"
        elif avg_risk < 0.5:
            risk_level = "🟡 中低风险"
            risk_color = "warning"
        elif avg_risk < 0.75:
            risk_level = "🟠 中高风险"
            risk_color = "warning"
        else:
            risk_level = "🔴 高风险"
            risk_color = "danger"
        
        st.markdown(f"## {risk_level}")
        st.markdown(f"**综合风险评分: {avg_risk*100:.1f}%**")
        
        # 各模型预测对比
        st.markdown("### 各模型预测值")
        
        model_results = pd.DataFrame([
            {'模型': name, '风险概率': f"{prob*100:.1f}%", '风险等级': '高' if prob > 0.5 else '低'}
            for name, prob in results.items()
        ])
        st.dataframe(model_results, use_container_width=True)
        
        # 风险仪表盘
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = avg_risk * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "综合风险评分 (%)"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 25], 'color': "lightgreen"},
                    {'range': [25, 50], 'color': "lightyellow"},
                    {'range': [50, 75], 'color': "orange"},
                    {'range': [75, 100], 'color': "lightcoral"}
                ],
            }
        ))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 建议
        st.markdown("### 💊 临床建议")
        
        if avg_risk > 0.75:
            st.error("""
            **高风险患者建议:**
            - 立即进行血铅检测
            - 评估螯合治疗必要性
            - 密切监测神经系统症状
            - 建议职业暴露评估
            """)
        elif avg_risk > 0.5:
            st.warning("""
            **中高风险患者建议:**
            - 定期复查血铅水平
            - 补充抗氧化剂（维生素C、E）
            - 饮食调整（增加钙、铁摄入）
            - 减少铅暴露源
            """)
        elif avg_risk > 0.25:
            st.info("""
            **中低风险患者建议:**
            - 保持健康生活方式
            - 定期体检监测
            - 注意职业防护
            """)
        else:
            st.success("""
            **低风险患者建议:**
            - 继续保持良好生活习惯
            - 无需特殊干预
            """)

elif page == "📏 列线图":
    st.markdown('<p class="sub-header">📏 列线图 (Nomogram) 分析</p>', unsafe_allow_html=True)
    
    # 导入列线图模块
    try:
        from nomogram import NomogramBuilder, create_nomogram, create_calibration_curve, generate_demo_data as nomo_generate_data
    except ImportError:
        st.error("请确保 nomogram.py 文件存在")
        st.stop()
    
    st.markdown("""
    ### 💡 列线图说明
    
    列线图 (Nomogram) 是临床预测模型的可视化工具，可以：
    - 将复杂模型转化为直观的分数系统
    - 实现个体化风险预测
    - 无需计算器即可快速评估
    """)
    
    # 数据准备
    if 'df' not in st.session_state:
        st.info("使用演示数据进行分析")
        df = nomo_generate_data(n_samples=2000)
    else:
        df = st.session_state['df']
    
    # 选择特征
    feature_cols = [col for col in df.columns if col not in ['铅毒性', 'Lead_Toxicity', 'target']]
    
    selected_features = st.multiselect(
        "选择用于列线图的特征",
        feature_cols,
        default=feature_cols[:6]
    )
    
    if len(selected_features) < 2:
        st.warning("请至少选择2个特征")
        st.stop()
    
    if st.button("构建列线图模型", type="primary"):
        with st.spinner("训练模型中..."):
            X = df[selected_features]
            y = df['铅毒性']
            
            # 分割数据
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            # 构建列线图
            builder = NomogramBuilder(feature_names=selected_features)
            builder.fit(X_train, y_train)
            
            # 预测
            y_train_prob = builder.predict_risk(X_train)
            y_test_prob = builder.predict_risk(X_test)
            
            # 计算指标
            from sklearn.metrics import roc_auc_score, brier_score_loss
            train_auc = roc_auc_score(y_train, y_train_prob)
            test_auc = roc_auc_score(y_test, y_test_prob)
            brier = brier_score_loss(y_test, y_test_prob)
            
            # 保存到 session state
            st.session_state['nomogram_builder'] = builder
            st.session_state['nomogram_features'] = selected_features
            
            # 显示性能指标
            st.markdown("#### 📊 模型性能")
            col1, col2, col3 = st.columns(3)
            col1.metric("训练集 AUC", f"{train_auc:.4f}")
            col2.metric("测试集 AUC", f"{test_auc:.4f}")
            col3.metric("Brier Score", f"{brier:.4f}")
        
        st.success("✅ 列线图模型构建完成!")
    
    # 列线图交互预测
    if 'nomogram_builder' in st.session_state:
        st.markdown("---")
        st.markdown("#### 🎯 个体化风险预测")
        
        builder = st.session_state['nomogram_builder']
        features = st.session_state['nomogram_features']
        
        # 创建输入控件
        col1, col2 = st.columns([2, 1])
        
        feature_values = {}
        with col1:
            st.markdown("##### 患者特征输入")
            cols = st.columns(3)
            for i, feat in enumerate(features):
                with cols[i % 3]:
                    if feat in builder.feature_ranges:
                        f_range = builder.feature_ranges[feat]
                        feature_values[feat] = st.slider(
                            f"📊 {feat}",
                            float(f_range['min']), 
                            float(f_range['max']),
                            float(f_range['mean']),
                            key=f"nomo_{feat}"
                        )
        
        # 计算风险
        with col2:
            st.markdown("##### 预测结果")
            
            if st.button("计算风险分数", type="primary"):
                # 计算总分
                total_points = 0
                for feat, val in feature_values.items():
                    points = builder.get_feature_points(feat, val)
                    total_points += points
                
                # 转换为风险
                risk_prob = builder.points_to_risk(total_points)
                
                # 显示
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
        
        # 显示列线图
        st.markdown("---")
        st.markdown("##### 📈 列线图")
        
        import os
        output_dir = '/Users/pengsu/mycode/lead-network-toxicology/output'
        nomogram_path = os.path.join(output_dir, 'nomogram_interactive.png')
        
        # 检查是否已有图片
        if os.path.exists(nomogram_path):
            st.image(nomogram_path, caption="铅毒性风险预测列线图", use_container_width=True)
        else:
            st.info("请先构建列线图模型，系统将自动生成列线图")

# ============================================================
# 森林图模块
# ============================================================
elif page == "🌲 森林图":
    st.markdown('<p class="sub-header">🌲 森林图分析</p>', unsafe_allow_html=True)
    st.markdown("展示各风险因素的效应量 (Odds Ratio) 及 95% 置信区间")
    st.markdown("---")
    
    # 导入森林图模块
    from forest_plot import (
        generate_demo_data, univariate_logistic_regression, 
        multivariate_logistic_regression, create_forest_plot,
        create_subgroup_forest_plot, create_comprehensive_forest_dashboard
    )
    
    # 数据选择
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("##### 📊 数据设置")
        use_demo = st.checkbox("使用演示数据", value=True)
        
        if use_demo:
            df = generate_demo_data()
        else:
            uploaded_file = st.file_uploader("上传数据文件 (CSV)", type=['csv'])
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
            else:
                df = generate_demo_data()
        
        st.markdown(f"**样本量**: {len(df)}")
        st.markdown(f"**特征数**: {len(df.columns)-1}")
        
        # 特征选择
        target_col = st.selectbox("选择结局变量", df.columns, index=len(df.columns)-1)
        
        feature_cols = st.multiselect(
            "选择分析特征",
            [c for c in df.columns if c != target_col],
            default=[c for c in ['Blood_Lead', 'Urine_Lead', 'Hair_Lead', 
                                  'Occupational_Exposure', 'Smoking', 'Age', 'BMI'] 
                     if c in df.columns]
        )
    
    with col2:
        # 分析类型选择
        analysis_type = st.radio(
            "分析类型",
            ["单变量分析", "多变量分析", "亚组分析"],
            horizontal=True
        )
        
        if st.button("🔍 运行森林图分析", type="primary"):
            with st.spinner("分析中..."):
                output_dir = '/Users/pengsu/mycode/lead-network-toxicology/output'
                
                if analysis_type == "单变量分析":
                    # 单变量分析
                    results = []
                    for feat in feature_cols:
                        r = univariate_logistic_regression(df, feat, target_col)
                        results.append(r)
                    results_df = pd.DataFrame(results)
                    
                    # 绘制森林图
                    fig_path = create_forest_plot(
                        results_df,
                        title='单变量分析森林图 - 铅毒性风险因素',
                        save_path=f'{output_dir}/forest_univariate.png'
                    )
                    
                    st.image(fig_path, caption="单变量分析森林图", use_container_width=True)
                    
                    # 显示结果表格
                    st.markdown("##### 📋 结果摘要")
                    st.dataframe(results_df[['feature', 'or', 'ci_lower', 'ci_upper', 'p_value']])
                    
                elif analysis_type == "多变量分析":
                    # 多变量分析
                    results_df = multivariate_logistic_regression(df, feature_cols, target_col)
                    
                    fig_path = create_forest_plot(
                        results_df,
                        title='多变量分析森林图 - 铅毒性风险因素',
                        save_path=f'{output_dir}/forest_multivariate.png'
                    )
                    
                    st.image(fig_path, caption="多变量分析森林图", use_container_width=True)
                    
                    # 显示结果表格
                    st.markdown("##### 📋 结果摘要")
                    st.dataframe(results_df[['feature', 'or', 'ci_lower', 'ci_upper', 'p_value']])
                    
                elif analysis_type == "亚组分析":
                    # 亚组分析
                    subgroup_var = st.selectbox("选择亚组变量", ['Smoking', 'Occupational_Exposure', 'Alcohol_Consumption'])
                    
                    for feat in feature_cols[:2]:  # 最多2个特征
                        fig_path = create_subgroup_forest_plot(
                            df, feat, subgroup_var, target_col,
                            save_path=f'{output_dir}/forest_subgroup_{feat.lower()}_by_{subgroup_var.lower()}.png'
                        )
                        st.image(fig_path, caption=f"亚组分析: {feat} 按 {subgroup_var}", use_container_width=True)
    
    # 结果解释
    st.markdown("---")
    st.markdown("""
    ### 📖 森林图解读指南
    
    - **效应量 (OR)**: >1 表示风险因素，<1 表示保护因素
    - **置信区间**: 包含1表示无统计学意义
    - **点大小**: 通常表示样本量或权重
    - **红色**: 风险因素 (OR > 1)
    - **绿色**: 保护因素 (OR < 1)
    """)

elif page == "⏱️ 生存分析":
    st.markdown('<p class="sub-header">⏱️ 生存分析</p>', unsafe_allow_html=True)
    
    # 导入生存分析模块
    try:
        from survival_analysis import generate_survival_data, plot_kaplan_meier, fit_cox_model
        from lifelines import KaplanMeierFitter
    except ImportError:
        st.error("请安装生存分析依赖: pip install lifelines")
        st.stop()
    
    # 数据选择
    st.markdown("### 数据设置")
    survival_data_source = st.radio("选择数据来源", ["使用演示数据", "上传生存数据"], horizontal=True)
    
    if survival_data_source == "使用演示数据":
        df_survival = generate_survival_data(n=500)
        st.success(f"已生成 {len(df_survival)} 条模拟生存数据")
    else:
        uploaded_file = st.file_uploader("上传CSV文件", type=['csv'])
        if uploaded_file:
            df_survival = pd.read_csv(uploaded_file)
        else:
            st.info("请上传数据文件")
            st.stop()
    
    # 显示数据
    with st.expander("查看数据"):
        st.dataframe(df_survival.head(10))
        st.markdown(f"**数据维度**: {df_survival.shape[0]} 行 × {df_survival.shape[1]} 列")
    
    # 分析选项
    st.markdown("### 分析选项")
    analysis_type = st.selectbox(
        "选择分析类型",
        ["Kaplan-Meier曲线", "Cox比例风险模型", "综合生存分析"]
    )
    
    time_col = st.selectbox("选择时间变量", df_survival.columns, 
                           index=df_survival.columns.tolist().index('Time_Months') if 'Time_Months' in df_survival.columns else 0)
    event_col = st.selectbox("选择事件变量", df_survival.columns,
                           index=df_survival.columns.tolist().index('Event') if 'Event' in df_survival.columns else 0)
    
    if analysis_type == "Kaplan-Meier曲线":
        group_col = st.selectbox("选择分组变量 (可选)", ["无"] + list(df_survival.columns),
                                index=0)
        
        if st.button("绘制生存曲线", type="primary"):
            with st.spinner("绘制中..."):
                if group_col == "无":
                    fig = plot_kaplan_meier(df_survival, time_col, event_col)
                else:
                    fig = plot_kaplan_meier(df_survival, time_col, event_col, group_col)
                
                st.pyplot(fig)
                
                # 保存按钮
                import io
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                st.download_button(
                    "下载生存曲线",
                    buf.getvalue(),
                    "kaplan_meier.png",
                    "image/png"
                )
    
    elif analysis_type == "Cox比例风险模型":
        covariates = st.multiselect(
            "选择协变量",
            ['Age', 'BMI', 'Blood_Lead', 'Urine_Lead', 'Hair_Lead', 'Smoking', 'Alcohol', 'Occupational_Exposure'],
            default=['Age', 'BMI', 'Blood_Lead', 'Smoking', 'Alcohol', 'Occupational_Exposure']
        )
        
        if st.button("拟合Cox模型", type="primary"):
            with st.spinner("拟合中..."):
                cph = fit_cox_model(df_survival, time_col, event_col, covariates)
                
                # 显示模型摘要
                st.markdown("### Cox模型结果")
                summary_df = pd.DataFrame({
                    '特征': covariates,
                    '系数': cph.summary['coef'].values,
                    'HR': cph.summary['exp(coef)'].values,
                    '95% CI lower': cph.summary['exp(coef) lower 95%'].values,
                    '95% CI upper': cph.summary['exp(coef) upper 95%'].values,
                    'P值': cph.summary['p'].values
                })
                st.dataframe(summary_df.style.format({
                    '系数': '{:.4f}',
                    'HR': '{:.3f}',
                    '95% CI lower': '{:.3f}',
                    '95% CI upper': '{:.3f}',
                    'P值': '{:.4f}'
                }))
                
                # 绘制森林图
                from survival_analysis import plot_cox_summary
                fig = plot_cox_summary(cph)
                st.pyplot(fig)
                
                # Concordance index
                st.metric("一致性指数 (C-index)", f"{cph.concordance_index_:.3f}")
    
    elif analysis_type == "综合生存分析":
        if st.button("运行综合生存分析", type="primary"):
            with st.spinner("分析中..."):
                from survival_analysis import create_comprehensive_survival_dashboard
                
                results = create_comprehensive_survival_dashboard(df_survival, output_dir='output')
                
                st.success("分析完成!")
                
                # 显示结果
                st.markdown("### Kaplan-Meier 曲线")
                st.image('output/km_overall.png', caption='整体生存曲线', use_container_width=True)
                st.image('output/km_by_lead.png', caption='按血铅分组的生存曲线', use_container_width=True)
                
                st.markdown("### Cox模型森林图")
                st.image('output/cox_forest.png', caption='Cox模型森林图', use_container_width=True)
                
                st.markdown("### 生存列线图")
                st.image('output/survival_nomogram.png', caption='生存预测列线图', use_container_width=True)
                
                # Log-rank检验结果
                st.markdown("### Log-rank检验")
                logrank = results['logrank_test']
                st.write(f"比较: {logrank['group1']} vs {logrank['group2']}")
                st.write(f"χ² = {logrank['test_statistic']:.3f}, P = {logrank['p_value']:.4f}")
    
    # 结果解释
    st.markdown("---")
    st.markdown("""
    ### 📖 生存分析解读指南
    
    - **Kaplan-Meier曲线**: 展示不同时间点的生存概率，置信带表示不确定性
    - **Log-rank检验**: 比较两组生存曲线是否有统计学差异
    - **Cox模型**: 评估各因素对风险的影响，HR>1表示风险增加
    - **C-index**: 模型区分度，>0.5表示有区分能力，越接近1越好
    - **列线图**: 可视化预测工具，可用于个体化风险评估
    """)

elif page == "📋 报告生成":
    st.markdown('<p class="sub-header">📋 分析报告生成</p>', unsafe_allow_html=True)
    
    if 'trained_models' not in st.session_state:
        st.warning("请先训练模型")
        st.stop()
    
    # 报告内容
    report_sections = st.multiselect(
        "选择报告内容",
        ["数据概览", "模型性能", "SHAP分析", "决策曲线", "风险预测"],
        default=["数据概览", "模型性能", "SHAP分析", "决策曲线"]
    )
    
    if st.button("生成报告", type="primary"):
        report_content = []
        
        report_content.append("# 铅网络毒理学分析报告\n")
        report_content.append(f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        if "数据概览" in report_sections:
            report_content.append("## 1. 数据概览\n")
            df = generate_demo_data()
            report_content.append(f"- 样本数量: {len(df)}\n")
            report_content.append(f"- 特征数量: {len(df.columns) - 1}\n")
            report_content.append(f"- 阳性率: {df['铅毒性'].mean()*100:.1f}%\n")
        
        if "模型性能" in report_sections:
            report_content.append("\n## 2. 模型性能\n")
            metrics_df = st.session_state['metrics']
            report_content.append("\n### 性能指标\n")
            report_content.append(metrics_df.to_markdown(index=False))
        
        if "决策曲线" in report_sections:
            report_content.append("\n## 3. 决策曲线分析\n")
            report_content.append("决策曲线分析显示模型在不同阈值下的临床净收益。\n")
        
        if "风险预测" in report_sections:
            report_content.append("\n## 4. 风险预测说明\n")
            report_content.append("""
- 低风险 (< 25%): 建议保持健康生活方式
- 中低风险 (25-50%): 定期体检监测
- 中高风险 (50-75%): 需积极干预
- 高风险 (> 75%): 建议临床治疗
""")
        
        # 显示报告
        st.markdown("".join(report_content))
        
        # 下载报告
        st.download_button(
            "📥 下载报告 (Markdown)",
            "".join(report_content),
            "lead_toxicology_report.md",
            "text/markdown"
        )

# ============================================================
# 主页脚注
# ============================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🧪 铅网络毒理学分析平台 | 重庆医科大学</p>
    <p>基于机器学习的铅毒性风险预测与生物标志物分析</p>
</div>
""", unsafe_allow_html=True)
