#!/usr/bin/env python3
"""
铅网络毒理学 - 生物标志物网络分析
Lead Network Toxicology - Biomarker Network Analysis

功能：
1. 生物标志物相关性网络构建
2. 风险因素交互网络
3. 毒理学通路网络可视化
4. 社区检测与模块识别
5. 网络中心性分析
6. 动态网络可视化

作者: Pain (重庆医科大学)
日期: 2026-02-27
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 数据生成 (模拟铅毒性数据)
# ============================================================

def generate_toxicology_data(n_samples=500, seed=42):
    """生成模拟的铅毒性生物标志物数据 - 带真实相关性"""
    np.random.seed(seed)
    
    # 基础变量
    data = pd.DataFrame()
    
    # 人口学特征
    data['Age'] = np.random.normal(45, 15, n_samples).clip(18, 80)
    data['Sex'] = np.random.binomial(1, 0.5, n_samples)
    data['BMI'] = np.random.normal(24, 4, n_samples).clip(16, 40)
    
    # 首先生成潜在变量 (造成相关性)
    latent = np.random.normal(0, 1, n_samples)
    lead_exposure = np.random.normal(0, 1, n_samples) + latent
    oxidative_stress = np.random.normal(0, 1, n_samples) + 0.7 * lead_exposure + np.random.normal(0, 0.5, n_samples)
    inflammation = np.random.normal(0, 1, n_samples) + 0.6 * oxidative_stress + np.random.normal(0, 0.5, n_samples)
    gut_dysfunction = np.random.normal(0, 1, n_samples) + 0.5 * inflammation + np.random.normal(0, 0.5, n_samples)
    
    # 铅暴露指标 (核心) - 高度相关
    data['Blood_Lead'] = np.exp(2.5 + 0.8 * lead_exposure).clip(1, 80)
    data['Urine_Lead'] = np.exp(1.8 + 0.9 * lead_exposure + np.random.normal(0, 0.3, n_samples)).clip(0.5, 25)
    data['Hair_Lead'] = np.exp(3.0 + 0.85 * lead_exposure + np.random.normal(0, 0.25, n_samples)).clip(1, 50)
    
    # 其他重金属 - 与铅暴露相关
    data['Blood_Arsenic'] = np.exp(2.0 + 0.5 * lead_exposure + np.random.normal(0, 0.4, n_samples)).clip(1, 30)
    data['Blood_Cadmium'] = np.exp(1.5 + 0.4 * lead_exposure + np.random.normal(0, 0.4, n_samples)).clip(0.5, 15)
    data['Blood_Manganese'] = np.exp(3.5 + 0.3 * lead_exposure + np.random.normal(0, 0.4, n_samples)).clip(5, 50)
    
    # 氧化应激标志物 - 相互高度相关
    data['SOD'] = (120 - 20 * oxidative_stress + np.random.normal(0, 10, n_samples)).clip(50, 250)
    data['GSH'] = (8 - 1.5 * oxidative_stress + np.random.normal(0, 0.8, n_samples)).clip(3, 15)
    data['MDA'] = np.exp(1.2 + 0.6 * oxidative_stress + np.random.normal(0, 0.3, n_samples)).clip(0.5, 10)
    data['8_OHdG'] = np.exp(2.5 + 0.55 * oxidative_stress + np.random.normal(0, 0.3, n_samples)).clip(1, 20)
    
    # 炎症标志物 - 相互高度相关
    data['CRP'] = np.exp(1.0 + 0.8 * inflammation + np.random.normal(0, 0.5, n_samples)).clip(0.1, 50)
    data['IL6'] = np.exp(1.5 + 0.7 * inflammation + np.random.normal(0, 0.4, n_samples)).clip(0.5, 30)
    data['TNF_alpha'] = np.exp(2.0 + 0.65 * inflammation + np.random.normal(0, 0.35, n_samples)).clip(1, 25)
    
    # 肝肾功能
    data['ALT'] = (25 + 8 * inflammation + np.random.normal(0, 5, n_samples)).clip(5, 100)
    data['AST'] = (28 + 10 * inflammation + np.random.normal(0, 6, n_samples)).clip(10, 120)
    data['Creatinine'] = (80 + 15 * inflammation + np.random.normal(0, 10, n_samples)).clip(40, 200)
    data['BUN'] = (5 + 1.2 * inflammation + np.random.normal(0, 0.8, n_samples)).clip(2, 15)
    
    # 心血管标志物
    data['Systolic_BP'] = (130 + 15 * inflammation + np.random.normal(0, 8, n_samples)).clip(90, 200)
    data['Diastolic_BP'] = (82 + 10 * inflammation + np.random.normal(0, 6, n_samples)).clip(60, 120)
    data['Cholesterol'] = (5.2 + 1.0 * inflammation + np.random.normal(0, 0.6, n_samples)).clip(3, 10)
    data['HbA1c'] = (5.5 + 0.6 * inflammation + np.random.normal(0, 0.4, n_samples)).clip(4, 10)
    
    # 胆汁酸 (肠-肝轴) - 相互相关
    data['DCA'] = np.exp(0.5 + 0.5 * gut_dysfunction + np.random.normal(0, 0.4, n_samples)).clip(0.1, 10)
    data['LCA'] = np.exp(-0.5 + 0.55 * gut_dysfunction + np.random.normal(0, 0.35, n_samples)).clip(0.05, 5)
    data['CA'] = np.exp(1.0 + 0.45 * gut_dysfunction + np.random.normal(0, 0.35, n_samples)).clip(0.5, 15)
    data['UDCA'] = np.exp(0.3 + 0.5 * gut_dysfunction + np.random.normal(0, 0.35, n_samples)).clip(0.1, 8)
    
    # 肠道屏障标志物 - 与肠道功能障碍相关
    data['Calprotectin'] = np.exp(2.5 + 0.7 * gut_dysfunction + np.random.normal(0, 0.4, n_samples)).clip(10, 500)
    data['Zonulin'] = np.exp(3.0 + 0.6 * gut_dysfunction + np.random.normal(0, 0.35, n_samples)).clip(20, 200)
    data['LBP'] = np.exp(10 + 0.4 * gut_dysfunction + np.random.normal(0, 0.3, n_samples)).clip(5, 50)
    
    # 风险因素
    data['Smoking'] = np.random.binomial(1, 0.3, n_samples)
    data['Occupational_Exposure'] = np.random.binomial(1, 0.25, n_samples)
    
    # 创建铅毒性二分类结局 (基于风险评分 + 噪声)
    risk_score = (
        0.4 * (data['Blood_Lead'] > 10).astype(int) +
        0.3 * (data['Occupational_Exposure'] == 1).astype(int) +
        0.3 * (data['Smoking'] == 1).astype(int) +
        0.2 * (data['MDA'] > 3).astype(int) +
        0.2 * (data['Calprotectin'] > 100).astype(int) +
        np.random.normal(0, 0.3, n_samples)
    )
    data['Lead_Toxicity'] = (risk_score > risk_score.mean()).astype(int)
    
    return data

# ============================================================
# 网络构建
# ============================================================

def build_correlation_network(data, variables, threshold=0.15):
    """构建相关性网络"""
    # 计算相关矩阵
    corr_matrix = data[variables].corr(method='spearman')
    
    # 创建网络图
    G = nx.Graph()
    
    # 添加节点
    for var in variables:
        G.add_node(var)
    
    # 添加边 (相关性超过阈值的)
    edges = []
    for i, var1 in enumerate(variables):
        for j, var2 in enumerate(variables):
            if i < j:
                corr = corr_matrix.loc[var1, var2]
                if abs(corr) > threshold:
                    G.add_edge(var1, var2, weight=abs(corr), correlation=corr)
                    edges.append((var1, var2, corr))
    
    return G, corr_matrix, edges

def calculate_network_metrics(G):
    """计算网络中心性指标"""
    metrics = {}
    
    if G.number_of_nodes() > 0:
        # 度中心性
        metrics['degree_centrality'] = nx.degree_centrality(G)
        
        # 介数中心性
        metrics['betweenness_centrality'] = nx.betweenness_centrality(G)
        
        # 接近中心性
        metrics['closeness_centrality'] = nx.closeness_centrality(G)
        
        # 特征向量中心性
        try:
            metrics['eigenvector_centrality'] = nx.eigenvector_centrality(G, max_iter=1000)
        except:
            metrics['eigenvector_centrality'] = {n: 0 for n in G.nodes()}
        
        # PageRank
        metrics['pagerank'] = nx.pagerank(G)
        
        # 聚类系数
        metrics['clustering_coefficient'] = nx.clustering(G)
    
    return metrics

def detect_communities(G):
    """检测网络社区"""
    try:
        communities = nx.community.louvain_communities(G, seed=42)
        return communities
    except:
        return []

# ============================================================
# 可视化函数
# ============================================================

def plot_static_network(G, corr_matrix, metrics, output_path):
    """绘制静态网络图"""
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # 图1: 网络图
    ax1 = axes[0]
    
    if G.number_of_nodes() > 0:
        # 节点布局
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # 节点大小 (基于度中心性)
        degrees = dict(G.degree())
        node_sizes = [300 + degrees.get(n, 0) * 200 for n in G.nodes()]
        
        # 节点颜色 (基于PageRank)
        pagerank = metrics.get('pagerank', {})
        node_colors = [pagerank.get(n, 0) for n in G.nodes()]
        
        # 边宽度 (基于相关性)
        edge_weights = [G[u][v]['weight'] * 3 for u, v in G.edges()]
        
        # 边颜色 (正相关=红色, 负相关=蓝色)
        edge_colors = ['#e74c3c' if G[u][v]['correlation'] > 0 else '#3498db' 
                       for u, v in G.edges()]
        
        # 绘制网络
        nx.draw_networkx_edges(G, pos, ax=ax1, width=edge_weights, 
                               edge_color=edge_colors, alpha=0.6)
        nodes = nx.draw_networkx_nodes(G, pos, ax=ax1, node_size=node_sizes,
                                       node_color=node_colors, cmap=plt.cm.YlOrRd,
                                       alpha=0.8)
        nx.draw_networkx_labels(G, pos, ax=ax1, font_size=8, font_weight='bold')
        
        ax1.set_title('Biomarker Correlation Network\n(Node size: degree, Color: PageRank)', 
                     fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # 添加颜色条
        sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, 
                                   norm=plt.Normalize(vmin=min(node_colors), 
                                                     vmax=max(node_colors)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax1, shrink=0.5)
        cbar.set_label('PageRank Score', fontsize=10)
    
    # 图2: 相关矩阵热力图
    ax2 = axes[1]
    
    # 简化相关矩阵显示
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, mask=mask, ax=ax2, cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.5,
                cbar_kws={'shrink': 0.5, 'label': 'Spearman Correlation'})
    ax2.set_title('Biomarker Correlation Matrix', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='both', labelsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")
    
    return fig

def plot_interactive_network(G, corr_matrix, metrics, communities, output_path):
    """绘制交互式网络图 (Plotly)"""
    
    if G.number_of_nodes() == 0:
        print("No network to visualize")
        return
    
    # 节点布局
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # 准备节点数据
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []
    
    pagerank = metrics.get('pagerank', {})
    degree_cent = metrics.get('degree_centrality', {})
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # 节点标签
        pr = pagerank.get(node, 0)
        dc = degree_cent.get(node, 0)
        node_text.append(f"<b>{node}</b><br>PageRank: {pr:.4f}<br>Degree Centrality: {dc:.4f}")
        
        # 节点大小
        node_size.append(15 + dc * 50)
        
        # 节点颜色
        node_color.append(pr)
    
    # 准备边数据
    edge_x = []
    edge_y = []
    edge_weights = []
    edge_colors = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        corr = G[edge[0]][edge[1]]['correlation']
        edge_weights.append(abs(corr))
        edge_colors.append(corr)
    
    # 创建图形
    fig = go.Figure()
    
    # 添加边 - 简化颜色处理
    for i in range(0, len(edge_x), 3):
        corr_val = edge_colors[i//3]
        edge_color = 'rgba(231, 76, 60, 0.5)' if corr_val > 0 else 'rgba(52, 152, 219, 0.5)'
        fig.add_trace(go.Scatter(
            x=[edge_x[i], edge_x[i+1]],
            y=[edge_y[i], edge_y[i+1]],
            mode='lines',
            line=dict(width=edge_weights[i//3] * 3, 
                     color=edge_color),
            hoverinfo='text',
            text=f"Correlation: {corr_val:.3f}",
            showlegend=False
        ))
    
    # 添加节点
    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        marker=dict(
            size=node_size,
            color=node_color,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='PageRank', x=1.02),
            line=dict(width=1, color='white')
        ),
        text=[n for n in G.nodes()],
        textposition='top center',
        textfont=dict(size=8),
        hoverinfo='text',
        hovertext=node_text,
        showlegend=False
    ))
    
    # 更新布局
    fig.update_layout(
        title=dict(
            text='<b>Interactive Biomarker Network</b><br><sub>Drag nodes, hover for details</sub>',
            x=0.5,
            font=dict(size=18)
        ),
        showlegend=False,
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        width=1000,
        height=800,
        margin=dict(l=50, r=50, t=100, b=50)
    )
    
    # 保存HTML
    fig.write_html(output_path)
    print(f"Saved: {output_path}")
    
    # 保存PNG
    png_path = output_path.replace('.html', '.png')
    fig.write_image(png_path, scale=2)
    print(f"Saved: {png_path}")
    
    return fig

def plot_centrality_comparison(metrics, output_path):
    """绘制中心性指标对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    centrality_names = {
        'degree_centrality': 'Degree Centrality',
        'betweenness_centrality': 'Betweenness Centrality',
        'closeness_centrality': 'Closeness Centrality',
        'eigenvector_centrality': 'Eigenvector Centrality'
    }
    
    for idx, (key, title) in enumerate(centrality_names.items()):
        ax = axes[idx // 2, idx % 2]
        
        if key in metrics and metrics[key]:
            # 排序
            sorted_vals = sorted(metrics[key].items(), key=lambda x: x[1], reverse=True)
            names = [x[0] for x in sorted_vals[:15]]
            values = [x[1] for x in sorted_vals[:15]]
            
            # 颜色渐变
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
            
            bars = ax.barh(range(len(names)), values, color=colors)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel('Score', fontsize=10)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # 添加数值标签
            for i, (bar, val) in enumerate(zip(bars, values)):
                ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")
    
    return fig

def plot_community_network(G, communities, output_path):
    """绘制社区检测结果"""
    fig, ax = plt.subplots(figsize=(14, 12))
    
    if G.number_of_nodes() > 0:
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # 为每个社区分配颜色
        colors = plt.cm.Set3(np.linspace(0, 1, len(communities)))
        node_colors = []
        
        for node in G.nodes():
            for i, comm in enumerate(communities):
                if node in comm:
                    node_colors.append(colors[i])
                    break
            else:
                node_colors.append('#808080')
        
        # 绘制网络
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, edge_color='gray')
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                               node_size=500, alpha=0.8)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_weight='bold')
        
        # 添加图例
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor=colors[i], markersize=12,
                                      label=f'Community {i+1}')
                          for i in range(len(communities))]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
        
        ax.set_title('Biomarker Network - Community Detection\n(Louvain Algorithm)', 
                    fontsize=14, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")
    
    return fig

def plot_risk_factor_network(data, output_path):
    """绘制风险因素与生物标志物的关联网络"""
    # 选择关键风险因素和生物标志物
    risk_factors = ['Blood_Lead', 'Urine_Lead', 'Smoking', 'Occupational_Exposure']
    biomarkers = ['MDA', '8_OHdG', 'CRP', 'IL6', 'Calprotectin', 'Zonulin', 
                  'SOD', 'GSH', 'DCA', 'ALT', 'AST', 'Creatinine']
    
    # 计算相关性
    all_vars = risk_factors + biomarkers
    corr_matrix = data[all_vars].corr(method='spearman')
    
    # 创建网络
    G = nx.Graph()
    
    # 添加节点
    for var in all_vars:
        if var in risk_factors:
            G.add_node(var, node_type='risk')
        else:
            G.add_node(var, node_type='biomarker')
    
    # 添加边
    for rf in risk_factors:
        for bm in biomarkers:
            corr = corr_matrix.loc[rf, bm]
            if abs(corr) > 0.15:
                G.add_edge(rf, bm, weight=abs(corr), correlation=corr)
    
    # 绘图
    fig, ax = plt.subplots(figsize=(14, 12))
    
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # 分离节点类型
    risk_nodes = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'risk']
    biomarker_nodes = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'biomarker']
    
    # 绘制边
    edge_colors = ['#e74c3c' if G[u][v]['correlation'] > 0 else '#3498db' 
                   for u, v in G.edges()]
    edge_weights = [abs(G[u][v]['weight']) * 4 for u, v in G.edges()]
    
    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_weights, 
                           edge_color=edge_colors, alpha=0.5)
    
    # 绘制风险因素节点 (红色)
    nx.draw_networkx_nodes(G, pos, nodelist=risk_nodes, ax=ax,
                          node_color='#e74c3c', node_size=1500,
                          node_shape='s', alpha=0.9)
    
    # 绘制生物标志物节点 (绿色)
    nx.draw_networkx_nodes(G, pos, nodelist=biomarker_nodes, ax=ax,
                          node_color='#27ae60', node_size=1000,
                          alpha=0.9)
    
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_weight='bold')
    
    ax.set_title('Risk Factor - Biomarker Association Network\n(Red: Risk Factors, Green: Biomarkers)', 
                fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")
    
    return fig

def plot_pathway_network(data, output_path):
    """绘制毒理学通路网络"""
    # 定义通路
    pathways = {
        'Oxidative Stress': ['SOD', 'GSH', 'MDA', '8_OHdG'],
        'Inflammation': ['CRP', 'IL6', 'TNF_alpha'],
        'Liver Function': ['ALT', 'AST', 'Creatinine', 'BUN'],
        'Cardiovascular': ['Systolic_BP', 'Diastolic_BP', 'Cholesterol', 'HbA1c'],
        'Gut-Liver Axis': ['DCA', 'LCA', 'CA', 'UDCA', 'Calprotectin', 'Zonulin', 'LBP'],
        'Heavy Metal Exposure': ['Blood_Lead', 'Urine_Lead', 'Blood_Arsenic', 
                                  'Blood_Cadmium', 'Blood_Manganese']
    }
    
    # 计算通路间相关性
    pathway_corr = {}
    pathway_pairs = []
    
    for i, (p1, vars1) in enumerate(pathways.items()):
        for j, (p2, vars2) in enumerate(pathways.items()):
            if i < j:
                # 计算两个通路之间的平均相关性
                valid_vars1 = [v for v in vars1 if v in data.columns]
                valid_vars2 = [v for v in vars2 if v in data.columns]
                
                if valid_vars1 and valid_vars2:
                    corr = data[valid_vars1 + valid_vars2].corr().loc[valid_vars1, valid_vars2]
                    avg_corr = corr.values.mean()
                    pathway_corr[(p1, p2)] = avg_corr
                    pathway_pairs.append((p1, p2, avg_corr))
    
    # 创建网络
    G = nx.Graph()
    
    for pathway in pathways.keys():
        G.add_node(pathway)
    
    for (p1, p2), corr in pathway_corr.items():
        if abs(corr) > 0.1:
            G.add_edge(p1, p2, weight=abs(corr), correlation=corr)
    
    # 绘图
    fig, ax = plt.subplots(figsize=(12, 10))
    
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # 节点大小 (基于连接数)
    degrees = dict(G.degree())
    node_sizes = [2000 + degrees.get(n, 0) * 500 for n in G.nodes()]
    
    # 节点颜色
    pathway_colors = {
        'Oxidative Stress': '#e74c3c',
        'Inflammation': '#f39c12',
        'Liver Function': '#9b59b6',
        'Cardiovascular': '#3498db',
        'Gut-Liver Axis': '#27ae60',
        'Heavy Metal Exposure': '#34495e'
    }
    colors = [pathway_colors.get(n, '#95a5a6') for n in G.nodes()]
    
    # 边
    edge_weights = [G[u][v]['weight'] * 10 for u, v in G.edges()]
    edge_colors = ['#e74c3c' if G[u][v]['correlation'] > 0 else '#3498db' 
                   for u, v in G.edges()]
    
    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_weights,
                           edge_color=edge_colors, alpha=0.6)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                          node_color=colors, alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight='bold')
    
    ax.set_title('Toxicological Pathway Network\n(Node size: connectivity)', 
                fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")
    
    return fig

def generate_network_report(G, metrics, communities, output_path):
    """生成网络分析报告"""
    report = []
    report.append("=" * 60)
    report.append("Biomarker Network Analysis Report")
    report.append("=" * 60)
    report.append("")
    
    # 网络基本信息
    report.append("## Network Overview")
    report.append(f"- Number of Nodes: {G.number_of_nodes()}")
    report.append(f"- Number of Edges: {G.number_of_edges()}")
    report.append(f"- Network Density: {nx.density(G):.4f}")
    
    if G.number_of_nodes() > 0:
        report.append(f"- Average Degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
        
        if nx.is_connected(G):
            report.append(f"- Network Diameter: {nx.diameter(G)}")
            report.append(f"- Average Path Length: {nx.average_shortest_path_length(G):.2f}")
        
        report.append(f"- Average Clustering Coefficient: {nx.average_clustering(G):.4f}")
    
    report.append("")
    
    # 社区检测
    report.append("## Community Detection")
    report.append(f"- Number of Communities: {len(communities)}")
    for i, comm in enumerate(communities):
        report.append(f"- Community {i+1}: {', '.join(sorted(comm)[:10])}" + 
                     (f" ... (+{len(comm)-10} more)" if len(comm) > 10 else ""))
    
    report.append("")
    
    # Top中心性
    report.append("## Top Centrality Measures")
    
    for metric_name, metric_dict in metrics.items():
        if metric_dict:
            sorted_vals = sorted(metric_dict.items(), key=lambda x: x[1], reverse=True)[:5]
            report.append(f"\n### {metric_name.replace('_', ' ').title()}")
            for node, val in sorted_vals:
                report.append(f"- {node}: {val:.4f}")
    
    report.append("")
    report.append("=" * 60)
    
    # 保存报告
    report_text = "\n".join(report)
    with open(output_path, 'w') as f:
        f.write(report_text)
    
    print(f"Saved: {output_path}")
    
    return report_text

# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print("Biomarker Network Analysis")
    print("=" * 60)
    
    # 生成数据
    print("\n[1/7] Generating data...")
    data = generate_toxicology_data(n_samples=500)
    print(f"Generated {len(data)} samples with {len(data.columns)} variables")
    
    # 选择网络分析变量
    variables = [
        # 铅暴露
        'Blood_Lead', 'Urine_Lead', 'Hair_Lead',
        # 其他重金属
        'Blood_Arsenic', 'Blood_Cadmium', 'Blood_Manganese',
        # 氧化应激
        'SOD', 'GSH', 'MDA', '8_OHdG',
        # 炎症
        'CRP', 'IL6', 'TNF_alpha',
        # 肝肾功能
        'ALT', 'AST', 'Creatinine', 'BUN',
        # 心血管
        'Systolic_BP', 'Cholesterol', 'HbA1c',
        # 肠-肝轴
        'DCA', 'LCA', 'CA', 'UDCA',
        # 肠道屏障
        'Calprotectin', 'Zonulin', 'LBP'
    ]
    
    # 构建网络
    print("\n[2/7] Building correlation network...")
    G, corr_matrix, edges = build_correlation_network(data, variables, threshold=0.25)
    print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # 计算网络指标
    print("\n[3/7] Calculating network metrics...")
    metrics = calculate_network_metrics(G)
    
    # 社区检测
    print("\n[4/7] Detecting communities...")
    communities = detect_communities(G)
    print(f"Found {len(communities)} communities")
    
    # 输出目录
    output_dir = 'output'
    
    # 绘制网络图
    print("\n[5/7] Generating visualizations...")
    plot_static_network(G, corr_matrix, metrics, f'{output_dir}/biomarker_network.png')
    
    # 交互式网络
    plot_interactive_network(G, corr_matrix, metrics, communities, 
                            f'{output_dir}/interactive_biomarker_network.html')
    
    # 中心性对比
    plot_centrality_comparison(metrics, f'{output_dir}/network_centrality_comparison.png')
    
    # 社区网络
    if communities:
        plot_community_network(G, communities, f'{output_dir}/community_network.png')
    
    # 风险因素网络
    plot_risk_factor_network(data, f'{output_dir}/risk_factor_network.png')
    
    # 通路网络
    plot_pathway_network(data, f'{output_dir}/pathway_network.png')
    
    # 生成报告
    print("\n[6/7] Generating network report...")
    report = generate_network_report(G, metrics, communities, f'{output_dir}/network_analysis_report.txt')
    print(report)
    
    # 保存相关性矩阵
    print("\n[7/7] Saving correlation matrix...")
    corr_matrix.to_csv(f'{output_dir}/biomarker_correlation_matrix.csv')
    print(f"Saved: {output_dir}/biomarker_correlation_matrix.csv")
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
