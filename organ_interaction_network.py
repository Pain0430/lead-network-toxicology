"""
铅暴露对器官互作网络的扰动分析
================================
基于网络毒理学方法
分析铅暴露对心-肝-脾-肺-肾五脏互作网络的影响

作者: Pain's AI Assistant  
日期: 2026-03-01
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

# ============= 数据模拟 =============
def simulate_organ_data(n=5000):
    """模拟NHANES多器官健康数据"""
    
    np.random.seed(42)
    
    data = {}
    
    # 铅暴露
    data['BPB'] = np.random.lognormal(mean=0.5, sigma=0.8, size=n)
    
    # 心血管系统
    data['HR'] = np.random.normal(72, 10, n)  # 心率
    data['SBP'] = np.random.normal(125, 15, n)  # 收缩压
    data['DBP'] = np.random.normal(80, 10, n)  # 舒张压
    data['EF'] = np.random.normal(65, 5, n)  # 射血分数
    
    # 肝脏系统
    data['ALT'] = np.random.normal(25, 10, n)  # 谷丙转氨酶
    data['AST'] = np.random.normal(28, 12, n)  # 谷草转氨酶
    data['ALB'] = np.random.normal(45, 5, n)  # 白蛋白
    
    # 脾脏/免疫系统
    data['WBC'] = np.random.normal(6.5, 1.5, n)  # 白细胞
    data['PLT'] = np.random.normal(250, 50, n)  # 血小板
    data['LYM'] = np.random.normal(2.0, 0.5, n)  # 淋巴细胞
    
    # 肺部系统
    data['FEV1'] = np.random.normal(3.0, 0.5, n)  # 用力呼气一秒量
    data['FVC'] = np.random.normal(4.0, 0.6, n)  # 用力肺活量
    data['FEV1_FVC'] = data['FEV1'] / data['FVC']
    
    # 肾脏系统
    data['BUN'] = np.random.normal(15, 5, n)  # 血尿素氮
    data['CREA'] = np.random.normal(80, 20, n)  # 肌酐
    data['eGFR'] = np.random.normal(90, 15, n)  # 估算肾小球滤过率
    
    df = pd.DataFrame(data)
    
    # 添加铅暴露对各器官的效应（模拟）
    # 铅暴露越高，某些指标越差
    df['SBP'] += df['BPB'] * 2
    df['ALT'] += df['BPB'] * 1.5
    df['BUN'] += df['BPB'] * 0.8
    df['eGFR'] -= df['BPB'] * 0.5
    
    return df

# ============= 构建器官互作网络 =============
def build_organ_network():
    """构建五脏相生相克网络"""
    
    G = nx.DiGraph()
    
    # 节点属性
    organs = {
        'Heart': {'chinese': '心', 'element': '火', 'system': 'cardiovascular'},
        'Liver': {'chinese': '肝', 'element': '木', 'system': 'metabolic'},
        'Spleen': {'chinese': '脾', 'element': '土', 'system': 'immune'},
        'Lung': {'chinese': '肺', 'element': '金', 'system': 'respiratory'},
        'Kidney': {'chinese': '肾', 'element': '水', 'system': 'renal'}
    }
    
    for organ, attrs in organs.items():
        G.add_node(organ, **attrs)
    
    # 相生关系 (五行相生)
    generating = [
        ('Heart', 'Liver'),  # 火生土 → 心生脾 (修正: 心对应火，肝对应木)
        ('Liver', 'Spleen'),  # 木克土 → 肝克脾
        ('Spleen', 'Lung'),   # 土生金 → 脾生肺
        ('Lung', 'Kidney'),   # 金生水 → 肺生肾
        ('Kidney', 'Heart'),  # 水生火 → 肾生心
    ]
    
    # 相克关系 (五行相克)
    controlling = [
        ('Heart', 'Spleen'),  # 火克土 → 心克脾
        ('Spleen', 'Kidney'), # 土克水 → 脾克肾
        ('Kidney', 'Lung'),   # 水克金 → 肾克肺
        ('Lung', 'Liver'),    # 金克木 → 肺克肝
        ('Liver', 'Heart'),   # 木生火 → 肝生心 (修正为相克)
    ]
    
    # 添加边和权重
    for src, dst in generating:
        G.add_edge(src, dst, relation='generating', weight=0.8)
    
    for src, dst in controlling:
        G.add_edge(src, dst, relation='controlling', weight=0.6)
    
    return G

# ============= 计算器官健康评分 =============
def calculate_organ_scores(df):
    """计算各器官健康评分"""
    
    scores = {}
    
    # 心血管评分 (归一化)
    heart_metrics = ['HR', 'SBP', 'DBP', 'EF']
    heart_data = df[heart_metrics].copy()
    for col in heart_metrics:
        heart_data[col] = (heart_data[col] - heart_data[col].min()) / (heart_data[col].max() - heart_data[col].min())
    scores['Heart'] = 1 - heart_data.mean(axis=1)
    
    # 肝脏评分
    liver_metrics = ['ALT', 'AST', 'ALB']
    liver_data = df[liver_metrics].copy()
    for col in ['ALT', 'AST']:
        liver_data[col] = (liver_data[col] - liver_data[col].min()) / (liver_data[col].max() - liver_data[col].min())
    liver_data['ALB'] = (liver_data['ALB'] - liver_data['ALB'].min()) / (liver_data['ALB'].max() - liver_data['ALB'].min())
    scores['Liver'] = 1 - liver_data.mean(axis=1)
    
    # 脾脏/免疫评分
    spleen_metrics = ['WBC', 'PLT', 'LYM']
    spleen_data = df[spleen_metrics].copy()
    for col in spleen_metrics:
        spleen_data[col] = (spleen_data[col] - spleen_data[col].min()) / (spleen_data[col].max() - spleen_data[col].min())
    scores['Spleen'] = 1 - spleen_data.mean(axis=1)
    
    # 肺评分
    lung_metrics = ['FEV1', 'FVC', 'FEV1_FVC']
    lung_data = df[lung_metrics].copy()
    for col in lung_metrics:
        lung_data[col] = (lung_data[col] - lung_data[col].min()) / (lung_data[col].max() - lung_data[col].min())
    scores['Lung'] = 1 - lung_data.mean(axis=1)
    
    # 肾脏评分
    kidney_metrics = ['BUN', 'CREA', 'eGFR']
    kidney_data = df[kidney_metrics].copy()
    for col in ['BUN', 'CREA']:
        kidney_data[col] = (kidney_data[col] - kidney_data[col].min()) / (kidney_data[col].max() - kidney_data[col].min())
    kidney_data['eGFR'] = (kidney_data['eGFR'] - kidney_data['eGFR'].min()) / (kidney_data['eGFR'].max() - kidney_data['eGFR'].min())
    scores['Kidney'] = 1 - kidney_data.mean(axis=1)
    
    return pd.DataFrame(scores)

# ============= 网络扰动分析 =============
def network_perturbation_analysis(df, G):
    """分析铅暴露对器官网络的扰动"""
    
    print("=" * 60)
    print("铅暴露对器官互作网络的扰动分析")
    print("=" * 60)
    
    # 计算器官健康评分
    organ_scores = calculate_organ_scores(df)
    
    # 按铅暴露分组
    lead_quartiles = pd.qcut(df['BPB'], q=4, labels=['Q1(低)', 'Q2', 'Q3', 'Q4(高)'])
    df['Lead_Group'] = lead_quartiles
    
    # 分析各组器官评分
    print("\n各铅暴露水平下的器官健康评分:")
    print("-" * 50)
    
    results = {}
    for group in ['Q1(低)', 'Q2', 'Q3', 'Q4(高)']:
        mask = df['Lead_Group'] == group
        results[group] = organ_scores[mask].mean()
    
    results_df = pd.DataFrame(results).T
    print(results_df.round(4))
    
    # 计算网络指标
    print("\n网络拓扑指标:")
    print("-" * 50)
    
    # 计算各器官之间的相关性（边权重）
    organ_correlations = organ_scores.corr()
    print("\n器官互作相关系数矩阵:")
    print(organ_correlations.round(4))
    
    # 计算网络中心性
    degree_centrality = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G)
    
    print("\n网络中心性指标:")
    for organ in G.nodes():
        print(f"  {organ}: 度中心性={degree_centrality[organ]:.3f}, "
              f"介数中心性={betweenness[organ]:.3f}")
    
    return organ_scores, results_df

# ============= 可视化 =============
def visualize_network(G, organ_scores, results_df):
    """可视化器官互作网络"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 1. 五脏网络图
    ax1 = axes[0]
    
    # 节点位置 (五行布局)
    pos = {
        'Heart': (0, 1),
        'Liver': (1, 0.5),
        'Spleen': (0.5, -0.5),
        'Lung': (-0.5, -0.5),
        'Kidney': (-1, 0.5)
    }
    
    # 节点颜色
    colors = {
        'Heart': '#FF6B6B',  # 红 - 火
        'Liver': '#4ECDC4',  # 青 - 木
        'Spleen': '#FFE66D', # 黄 - 土
        'Lung': '#95E1D3',  # 白 - 金
        'Kidney': '#6C5CE7'  # 紫 - 水
    }
    
    # 绘制边
    edge_colors = []
    for u, v in G.edges():
        if G[u][v]['relation'] == 'generating':
            edge_colors.append('#2ECC71')  # 绿色 - 相生
        else:
            edge_colors.append('#E74C3C')  # 红色 - 相克
    
    nx.draw_networkx_edges(G, pos, ax=ax1, edge_color=edge_colors, 
                           arrows=True, arrowsize=20, width=2,
                           connectionstyle="arc3,rad=0.1")
    
    # 绘制节点
    node_colors = [colors[node] for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, ax=ax1, node_color=node_colors, 
                         node_size=3000, alpha=0.9)
    
    # 节点标签
    labels = {node: f"{node}\n({G.nodes[node]['chinese']})" for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, ax=ax1, font_size=10, 
                           font_weight='bold')
    
    # 图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#2ECC71', linewidth=2, label='相生 (Generating)'),
        Line2D([0], [0], color='#E74C3C', linewidth=2, label='相克 (Controlling)')
    ]
    ax1.legend(handles=legend_elements, loc='lower center', ncol=2)
    ax1.set_title('五脏相生相克网络\n(Five Organ Interaction Network)', fontsize=14)
    ax1.axis('off')
    
    # 2. 铅暴露对器官评分的影响
    ax2 = axes[1]
    
    # 准备数据
    organs = ['Heart', 'Liver', 'Spleen', 'Lung', 'Kidney']
    x = np.arange(len(organs))
    width = 0.2
    
    for i, group in enumerate(['Q1(低)', 'Q2', 'Q3', 'Q4(高)']):
        values = results_df.loc[group, organs].values
        bars = ax2.bar(x + i * width, values, width, label=group, alpha=0.8)
    
    ax2.set_xlabel('器官 (Organ)', fontsize=12)
    ax2.set_ylabel('健康评分 (Health Score)', fontsize=12)
    ax2.set_title('铅暴露水平对各器官健康评分的影响\n(Effect of Lead Exposure on Organ Health Scores)', fontsize=14)
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels([f"{o}\n({G.nodes[o]['chinese']})" for o in organs])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('output/organ_interaction_network.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n可视化结果已保存至: output/organ_interaction_network.png")

# ============= 主函数 =============
def main():
    print("=" * 60)
    print("铅暴露对器官互作网络的扰动分析")
    print("=" * 60)
    
    # 1. 模拟数据
    print("\n[1/3] 模拟NHANES多器官数据...")
    df = simulate_organ_data(n=5000)
    print(f"  样本数: {len(df)}")
    print(f"  铅暴露范围: {df['BPB'].min():.2f} - {df['BPB'].max():.2f} μg/dL")
    
    # 2. 构建网络
    print("\n[2/3] 构建五脏互作网络...")
    G = build_organ_network()
    print(f"  节点数: {G.number_of_nodes()}")
    print(f"  边数: {G.number_of_edges()}")
    
    # 3. 网络扰动分析
    print("\n[3/3] 网络扰动分析...")
    organ_scores, results_df = network_perturbation_analysis(df, G)
    
    # 4. 可视化
    visualize_network(G, organ_scores, results_df)
    
    # 保存结果
    organ_scores.to_csv('output/organ_health_scores.csv', index=False)
    results_df.to_csv('output/lead_organ_impact.csv')
    
    print("\n✅ 分析完成!")
    print("  - output/organ_interaction_network.png")
    print("  - output/organ_health_scores.csv")
    print("  - output/lead_organ_impact.csv")

if __name__ == "__main__":
    main()
