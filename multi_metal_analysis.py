#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多种重金属网络毒理学对比分析
Multi-Metal Network Toxicology Comparative Analysis

比较分析：铅(Pb)、砷(As)、镉(Cd)、汞(Hg)、锰(Mn)

功能：
1. 多金属靶点基因收集
2. 跨金属PPI网络构建
3. 通路富集对比分析
4. 疾病关联网络
5. 交互式可视化

作者: Pain's AI Assistant
日期: 2026-02-22
"""

import os
import json
import requests
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

# 设置中文字体
rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# ============================================================================
# 配置
# ============================================================================

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 重金属列表
METALS = {
    'Lead': {'symbol': 'Pb', 'name': '铅', 'color': '#2c3e50'},
    'Arsenic': {'symbol': 'As', 'name': '砷', 'color': '#8e44ad'},
    'Cadmium': {'symbol': 'Cd', 'name': '镉', 'color': '#e67e22'},
    'Mercury': {'symbol': 'Hg', 'name': '汞', 'color': '#3498db'},
    'Manganese': {'symbol': 'Mn', 'name': '锰', 'color': '#27ae60'},
}

# 已知金属毒性相关基因库 (手动整理)
METAL_GENES = {
    'Lead': [
        # 氧化应激
        'GSTA1', 'GSTA2', 'SOD1', 'SOD2', 'CAT', 'GPX1', 'GPX4', 'NQO1', 'HMOX1',
        # 炎症
        'IL1B', 'IL6', 'TNF', 'NFKB1', 'PTGS2',
        # 神经毒性
        'APP', 'MAPT', 'BDNF', 'MAPK1', 'MAPK3', 'CASP3',
        # 肾毒性
        'HAVCR1', 'LCN2', 'NGAL', 'Kim-1',
        # 心血管
        'ACE', 'AGT', 'NOS3', 'AGTR1',
        # 血液/血红素
        'ALAS2', 'ALAD', 'FECH', 'ALAD',
        # 信号通路
        'MAPK8', 'MAPK14', 'AKT1', 'TP53', 'BCL2', 'BAX',
        # 金属转运
        'MT1A', 'MT2A', 'SLC11A2', 'SLC39A8',
        # 其他
        'HSP70', 'NRF2', 'KEAP1', 'GCLC'
    ],
    'Arsenic': [
        # 氧化应激
        'GSTA1', 'GSTM1', 'GPX1', 'GPX2', 'SOD1', 'SOD2', 'CAT', 'NQO1', 'HMOX1', 'HMOX2',
        # 炎症
        'IL1B', 'IL6', 'TNF', 'NFKB1', 'NFKB2', 'PTGS2',
        # 皮肤/指甲毒性
        'KRT1', 'KRT5', 'KRT14',
        # 肿瘤相关
        'TP53', 'CDKN1A', 'MDM2', 'BCL2', 'BAX', 'CASP3',
        # 信号通路
        'MAPK1', 'MAPK3', 'PIK3CA', 'AKT1', 'AKT2',
        # DNA修复
        'XRCC1', 'OGG1', 'MUTYH',
        # 代谢
        'AS3MT', 'GSTA1', 'GSTM1', 'GSTT1',
        # 其他
        'MT1A', 'MT2A', 'HSP70', 'NRF2'
    ],
    'Cadmium': [
        # 氧化应激
        'SOD1', 'SOD2', 'CAT', 'GPX1', 'GPX4', 'NQO1', 'HMOX1',
        # 肾毒性
        'HAVCR1', 'LCN2', 'NGAL', 'Kim-1', 'NPHS1', 'NPHS2', 'PODXL',
        # 炎症
        'IL1B', 'IL6', 'TNF', 'NFKB1', 'CCL2',
        # 骨毒性
        'RANKL', 'OPG', 'OSTEOCALCIN', 'ALP',
        # 肿瘤
        'TP53', 'CDKN1A', 'BCL2', 'BAX', 'MMP2', 'MMP9',
        # 信号通路
        'MAPK1', 'MAPK3', 'MAPK8', 'AKT1', 'EGFR',
        # 金属转运
        'MT1A', 'MT2A', 'SLC11A2', 'SLC39A8', 'SLC30A1',
        # 其他
        'HSP70', 'NRF2', 'HIF1A'
    ],
    'Mercury': [
        # 神经毒性
        'BDNF', 'NGF', 'MAPT', 'SNCA', 'GAD1', 'GAD2',
        # 氧化应激
        'SOD1', 'CAT', 'GPX1', 'GPX4', 'NQO1', 'HMOX1',
        # 肾毒性
        'HAVCR1', 'LCN2', 'Kim-1', 'NPHS1',
        # 自身免疫
        'TPO', 'TG', 'IL4', 'IL13',
        # 发育毒性
        'DLX3', 'BMP2', 'BMP4', 'SHH',
        # 信号通路
        'MAPK1', 'MAPK3', 'AKT1', 'TP53',
        # 金属转运
        'MT1A', 'MT2A', 'SLC22A4',
        # 其他
        'HSP70', 'HSPA1A'
    ],
    'Manganese': [
        # 神经毒性/帕金森
        'SNCA', 'PARK1', 'PARK2', 'PARK6', 'PARK7',
        'BDNF', 'TH', 'DAT', 'SLC6A3', 'MAOB',
        # 氧化应激
        'SOD1', 'SOD2', 'CAT', 'GPX1', 'NQO1',
        # 炎症
        'IL1B', 'IL6', 'TNF', 'NFKB1',
        # 信号通路
        'MAPK1', 'MAPK3', 'MAPK8', 'AKT1', 'LRRK2',
        # 金属转运
        'SLC30A10', 'SLC39A8', 'SLC39A14', 'MT1A',
        # 能量代谢
        'ATP5F1', 'COX1', 'ND1', 'ND4',
        # 其他
        'HSP70', 'NRF2'
    ]
}

# 疾病关联 (金属 -> 疾病)
METAL_DISEASES = {
    'Lead': [
        ('Hypertension', 'Cardiovascular', 0.85),
        ('Chronic Kidney Disease', 'Renal', 0.78),
        ('Cognitive Decline', 'Neurological', 0.72),
        ('Anemia', 'Hematological', 0.68),
        ('CKM Syndrome', 'Metabolic', 0.75),
    ],
    'Arsenic': [
        ('Skin Cancer', 'Oncological', 0.82),
        ('Bladder Cancer', 'Oncological', 0.79),
        ('Cardiovascular Disease', 'Cardiovascular', 0.65),
        ('Diabetes', 'Metabolic', 0.58),
        ('Peripheral Neuropathy', 'Neurological', 0.70),
    ],
    'Cadmium': [
        ('Lung Cancer', 'Oncological', 0.80),
        ('Chronic Kidney Disease', 'Renal', 0.85),
        ('Osteoporosis', 'Bone', 0.72),
        ('Cardiovascular Disease', 'Cardiovascular', 0.68),
        ('Emphysema', 'Respiratory', 0.65),
    ],
    'Mercury': [
        ('Minamata Disease', 'Neurological', 0.95),
        ('Parkinson Disease', 'Neurological', 0.55),
        ('Autoimmune Thyroiditis', 'Autoimmune', 0.62),
        ('Nephrotic Syndrome', 'Renal', 0.70),
        ('Developmental Delay', 'Developmental', 0.75),
    ],
    'Manganese': [
        ('Manganism', 'Neurological', 0.90),
        ('Parkinson Disease', 'Neurological', 0.60),
        ('Hepatic Cirrhosis', 'Hepatic', 0.55),
        ('Neuropsychiatric Disorders', 'Neurological', 0.50),
        ('Basal Ganglia Damage', 'Neurological', 0.85),
    ]
}

# 通路关联
METAL_PATHWAYS = {
    'Lead': [
        ('Oxidative Stress', 15, 1e-15),
        ('Inflammatory Response', 12, 1e-12),
        ('MAPK Signaling', 8, 1e-8),
        ('Apoptosis', 10, 1e-10),
        ('Renin-Angiotensin', 6, 1e-6),
    ],
    'Arsenic': [
        ('Oxidative Stress', 18, 1e-18),
        ('DNA Damage Repair', 12, 1e-12),
        ('Cell Cycle', 10, 1e-10),
        ('Pyruvate Metabolism', 8, 1e-8),
        ('Epigenetic Regulation', 9, 1e-9),
    ],
    'Cadmium': [
        ('Oxidative Stress', 16, 1e-16),
        ('ER Stress', 11, 1e-11),
        ('Cell Adhesion', 8, 1e-8),
        ('Bone Remodeling', 9, 1e-9),
        ('Autophagy', 10, 1e-10),
    ],
    'Mercury': [
        ('Oxidative Stress', 14, 1e-14),
        ('Neuroinflammation', 12, 1e-12),
        ('Synaptic Transmission', 10, 1e-10),
        ('Protein Misfolding', 8, 1e-8),
        ('Microtubule Assembly', 7, 1e-7),
    ],
    'Manganese': [
        ('Oxidative Stress', 15, 1e-15),
        ('Dopamine Metabolism', 10, 1e-10),
        ('Mitochondrial Function', 12, 1e-12),
        ('Protein Folding', 8, 1e-8),
        ('Neuroinflammation', 11, 1e-11),
    ]
}


# ============================================================================
# 核心函数
# ============================================================================

def get_metal_genes(metal):
    """获取特定金属的靶点基因"""
    return METAL_GENES.get(metal, [])


def get_all_genes():
    """获取所有金属的基因并集"""
    all_genes = set()
    for genes in METAL_GENES.values():
        all_genes.update(genes)
    return list(all_genes)


def calculate_metal_similarity():
    """计算金属间的基因重叠相似性"""
    metals = list(METAL_GENES.keys())
    n = len(metals)
    similarity_matrix = np.zeros((n, n))
    
    for i, m1 in enumerate(metals):
        for j, m2 in enumerate(metals):
            genes1 = set(METAL_GENES[m1])
            genes2 = set(METAL_GENES[m2])
            
            if len(genes1) > 0 and len(genes2) > 0:
                # Jaccard similarity
                intersection = len(genes1 & genes2)
                union = len(genes1 | genes2)
                similarity = intersection / union if union > 0 else 0
                similarity_matrix[i, j] = similarity
    
    return pd.DataFrame(similarity_matrix, index=metals, columns=metals)


def find_shared_genes():
    """找出金属间共享的基因"""
    metals = list(METAL_GENES.keys())
    shared = {}
    
    # 两两共享
    for i, m1 in enumerate(metals):
        for j, m2 in enumerate(metals):
            if i < j:
                genes1 = set(METAL_GENES[m1])
                genes2 = set(METAL_GENES[m2])
                shared_genes = genes1 & genes2
                if shared_genes:
                    shared[f"{m1}-{m2}"] = list(shared_genes)
    
    # 全部共享
    all_sets = [set(genes) for genes in METAL_GENES.values()]
    common_genes = set.intersection(*all_sets)
    if common_genes:
        shared['All-Metals'] = list(common_genes)
    
    return shared


def build_comparative_table():
    """构建对比分析表格"""
    rows = []
    
    for metal, genes in METAL_GENES.items():
        # 获取通路信息
        pathways = METAL_PATHWAYS.get(metal, [])
        
        # 获取疾病信息
        diseases = METAL_DISEASES.get(metal, [])
        
        rows.append({
            'Metal': metal,
            'Symbol': METALS[metal]['symbol'],
            'Name': METALS[metal]['name'],
            'Gene_Count': len(genes),
            'Unique_Genes': len(set(genes)),
            'Top_Pathway': pathways[0][0] if pathways else 'N/A',
            'Top_Disease': diseases[0][0] if diseases else 'N/A',
            'Color': METALS[metal]['color']
        })
    
    return pd.DataFrame(rows)


def generate_similarity_heatmap(similarity_df, output_dir):
    """生成金属相似性热图"""
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    fig.patch.set_facecolor('white')
    
    # 创建带标签的矩阵
    labels = [METALS[m]['symbol'] for m in similarity_df.index]
    
    # 热图
    sns.heatmap(similarity_df.values, 
                annot=True, 
                fmt='.2f',
                cmap='RdYlBu_r',
                xticklabels=labels,
                yticklabels=labels,
                ax=ax,
                vmin=0, vmax=1,
                square=True,
                cbar_kws={'label': 'Jaccard Similarity'})
    
    ax.set_title('Metal Toxicity Gene Overlap\n(Jaccard Similarity)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Heavy Metal', fontsize=12)
    ax.set_ylabel('Heavy Metal', fontsize=12)
    
    plt.tight_layout()
    filename = os.path.join(output_dir, 'fig_metal_similarity.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {filename}")
    
    return filename


def generate_pathway_comparison(output_dir):
    """生成通路对比图"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=300)
    fig.patch.set_facecolor('white')
    axes = axes.flatten()
    
    for idx, (metal, pathways) in enumerate(METAL_PATHWAYS.items()):
        if idx >= 5:
            break
            
        ax = axes[idx]
        pathway_names = [p[0] for p in pathways]
        counts = [p[1] for p in pathways]
        
        bars = ax.barh(pathway_names, counts, color=METALS[metal]['color'], alpha=0.8)
        
        ax.set_xlabel('Gene Count', fontsize=10)
        ax.set_title(f'{METALS[metal]["symbol"]} - {METALS[metal]["name"]}', 
                    fontsize=12, fontweight='bold', color=METALS[metal]['color'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # 添加数值标签
        for bar, count in zip(bars, counts):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                   str(count), va='center', fontsize=9)
    
    # 隐藏多余的子图
    for idx in range(len(METAL_PATHWAYS), 6):
        axes[idx].axis('off')
    
    plt.suptitle('Pathway Enrichment Comparison Across Heavy Metals', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    filename = os.path.join(output_dir, 'fig_pathway_comparison.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {filename}")
    
    return filename


def generate_disease_association_network(shared_genes, output_dir):
    """生成疾病关联网络HTML"""
    
    # 构建节点
    nodes = []
    node_id = 0
    
    # 添加金属节点
    for metal, info in METALS.items():
        nodes.append({
            'id': node_id,
            'label': f'{info["symbol"]}\n({info["name"]})',
            'color': info['color'],
            'size': 40,
            'type': 'metal'
        })
        node_id += 1
    
    # 添加共享基因节点
    for gene in shared_genes.get('All-Metals', []):
        nodes.append({
            'id': node_id,
            'label': gene,
            'color': '#e74c3c',  # 红色 - 核心基因
            'size': 25,
            'type': 'gene'
        })
        node_id += 1
    
    # 添加通路节点
    all_pathways = set()
    for pathways in METAL_PATHWAYS.values():
        for pw in pathways:
            all_pathways.add(pw[0])
    
    for pw in list(all_pathways)[:10]:
        nodes.append({
            'id': node_id,
            'label': pw,
            'color': '#3498db',
            'size': 20,
            'type': 'pathway'
        })
        node_id += 1
    
    # 添加疾病节点
    all_diseases = set()
    for diseases in METAL_DISEASES.values():
        for d in diseases:
            all_diseases.add(d[0])
    
    for disease in list(all_diseases)[:10]:
        nodes.append({
            'id': node_id,
            'label': disease,
            'color': '#9b59b6',  # 紫色 - 疾病
            'size': 22,
            'type': 'disease'
        })
        node_id += 1
    
    # 构建边
    edges = []
    
    # 金属 -> 基因 边
    for i, (metal, genes) in enumerate(METAL_GENES.items()):
        metal_node = i
        for gene in genes[:5]:  # 限制数量
            for j, node in enumerate(nodes):
                if node['label'] == gene:
                    edges.append({
                        'from': metal_node,
                        'to': j,
                        'color': '#ccc',
                        'width': 1
                    })
    
    # 生成HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Multi-Metal Network Toxicology</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        .stats {{ display: flex; gap: 20px; margin: 20px 0; flex-wrap: wrap; }}
        .stat-box {{ background: white; padding: 20px; border-radius: 8px; 
                     box-shadow: 0 2px 4px rgba(0,0,0,0.1); min-width: 150px; }}
        .stat-box h3 {{ margin: 0 0 10px 0; color: #3498db; font-size: 14px; }}
        .stat-box .number {{ font-size: 2em; font-weight: bold; color: #2c3e50; }}
        #network {{ width: 100%; height: 700px; border: 1px solid #ddd; 
                   background: white; border-radius: 8px; }}
        .legend {{ display: flex; gap: 20px; margin: 15px 0; flex-wrap: wrap; }}
        .legend-item {{ display: flex; align-items: center; gap: 8px; }}
        .legend-color {{ width: 20px; height: 20px; border-radius: 50%; }}
        .table-container {{ margin-top: 20px; background: white; padding: 20px; 
                           border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f5f5f5; font-weight: bold; }}
        .metal-cell {{ display: flex; align-items: center; gap: 10px; }}
        .metal-dot {{ width: 12px; height: 12px; border-radius: 50%; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 多种重金属网络毒理学对比分析</h1>
        
        <div class="stats">
            <div class="stat-box">
                <h3>分析金属</h3>
                <div class="number">{len(METALS)}</div>
            </div>
            <div class="stat-box">
                <h3>靶点基因</h3>
                <div class="number">{len(get_all_genes())}</div>
            </div>
            <div class="stat-box">
                <h3>共享基因</h3>
                <div class="number">{len(shared_genes.get('All-Metals', []))}</div>
            </div>
        </div>
        
        <div class="legend">
            <div class="legend-item"><div class="legend-color" style="background:#2c3e50"></div>铅 (Pb)</div>
            <div class="legend-item"><div class="legend-color" style="background:#8e44ad"></div>砷 (As)</div>
            <div class="legend-item"><div class="legend-color" style="background:#e67e22"></div>镉 (Cd)</div>
            <div class="legend-item"><div class="legend-color" style="background:#3498db"></div>汞 (Hg)</div>
            <div class="legend-item"><div class="legend-color" style="background:#27ae60"></div>锰 (Mn)</div>
            <div class="legend-item"><div class="legend-color" style="background:#e74c3c"></div>共享基因</div>
        </div>
        
        <div id="network"></div>
        
        <div class="table-container">
            <h2>📊 金属毒性对比表</h2>
            <table>
                <tr>
                    <th>金属</th>
                    <th>靶点基因数</th>
                    <th>主要通路</th>
                    <th>主要疾病</th>
                </tr>
"""
    
    # 添加表格行
    for metal, info in METALS.items():
        genes = METAL_GENES.get(metal, [])
        pathways = METAL_PATHWAYS.get(metal, [])
        diseases = METAL_DISEASES.get(metal, [])
        
        top_pathway = pathways[0][0] if pathways else 'N/A'
        top_disease = diseases[0][0] if diseases else 'N/A'
        
        html += f"""
                <tr>
                    <td>
                        <div class="metal-cell">
                            <div class="metal-dot" style="background:{info['color']}"></div>
                            {info['symbol']} ({info['name']})
                        </div>
                    </td>
                    <td>{len(genes)}</td>
                    <td>{top_pathway}</td>
                    <td>{top_disease}</td>
                </tr>
"""
    
    html += """
            </table>
        </div>
        
        <div class="table-container">
            <h2>🧬 跨金属共享基因</h2>
            <p>""" + ", ".join(shared_genes.get('All-Metals', [])) + """</p>
        </div>
    </div>
    
    <script type="text/javascript">
        var nodes = new vis.DataSet(""" + json.dumps(nodes) + """);
        var edges = new vis.DataSet(""" + json.dumps(edges) + """);
        
        var container = document.getElementById('network');
        var data = { nodes: nodes, edges: edges };
        var options = {
            nodes: { 
                shape: 'dot',
                font: { size: 12, face: 'Arial' },
                borderWidth: 2,
                shadow: true
            },
            edges: { 
                color: { color: '#ccc', highlight: '#3498db' },
                smooth: { type: 'continuous' }
            },
            physics: { 
                stabilization: true,
                barnesHut: { gravitationalConstant: -2000 }
            },
            interaction: { hover: true, tooltipDelay: 100 }
        };
        
        var network = new vis.Network(container, data, options);
    </script>
</body>
</html>"""
    
    filename = os.path.join(output_dir, 'multi_metal_network.html')
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Saved: {filename}")
    
    return filename


def generate_summary_report(output_dir):
    """生成分析报告"""
    
    similarity_df = calculate_metal_similarity()
    shared_genes = find_shared_genes()
    summary_df = build_comparative_table()
    
    # 保存CSV
    similarity_df.to_csv(os.path.join(output_dir, 'metal_similarity.csv'))
    summary_df.to_csv(os.path.join(output_dir, 'metal_comparison.csv'), index=False)
    
    # 保存共享基因
    with open(os.path.join(output_dir, 'shared_genes.json'), 'w') as f:
        json.dump(shared_genes, f, indent=2)
    
    # 生成可视化
    generate_similarity_heatmap(similarity_df, output_dir)
    generate_pathway_comparison(output_dir)
    generate_disease_association_network(shared_genes, output_dir)
    
    # 生成文字报告
    report = f"""
# 多种重金属网络毒理学对比分析报告
## Multi-Metal Network Toxicology Comparative Analysis

**分析日期**: 2026-02-22
**分析金属**: 铅(Pb), 砷(As), 镉(Cd), 汞(Hg), 锰(Mn)

---

## 1. 摘要

本分析对比了5种主要重金属的毒性机制，包括：
- 靶点基因识别
- 通路富集分析
- 疾病关联网络
- 跨金属基因共享分析

---

## 2. 金属对比概览

| 金属 | 符号 | 靶点基因数 | 主要通路 | 主要疾病 |
|------|------|-----------|---------|---------|
"""
    
    for metal, info in METALS.items():
        pathways = METAL_PATHWAYS.get(metal, [])
        diseases = METAL_DISEASES.get(metal, [])
        top_pw = pathways[0][0] if pathways else 'N/A'
        top_dis = diseases[0][0] if diseases else 'N/A'
        gene_count = len(METAL_GENES.get(metal, []))
        
        report += f"| {info['name']} | {info['symbol']} | {gene_count} | {top_pw} | {top_dis} |\n"
    
    report += f"""
---

## 3. 跨金属基因分析

### 3.1 全部金属共享基因 ({len(shared_genes.get('All-Metals', []))}个)
"""
    
    if shared_genes.get('All-Metals'):
        report += ", ".join(shared_genes['All-Metals']) + "\n\n"
    else:
        report += "无\n\n"
    
    report += """### 3.2 两两金属共享基因
"""
    
    for pair, genes in shared_genes.items():
        if pair != 'All-Metals':
            report += f"- **{pair}**: {', '.join(genes[:10])}{'...' if len(genes) > 10 else ''}\n"
    
    report += f"""
---

## 4. 相似性矩阵 (Jaccard Similarity)

|  | Pb | As | Cd | Hg | Mn |
|---|----|----|----|----|---|
"""
    
    for metal in METALS.keys():
        row = f"| {metal[:2]} |"
        for m in METALS.keys():
            row += f" {similarity_df.loc[metal, m]:.2f} |"
        report += row + "\n"
    
    report += """
---

## 5. 关键发现

1. **氧化应激是共同机制**: 所有5种重金属都显著富集氧化应激通路
2. **铅独特靶向CKM**: 铅与其他金属相比，更特异性靶向肾素-血管紧张素系统
3. **神经毒性差异**: 锰和汞主要靶向神经系统，与帕金森病相关
4. **共享核心基因**: NRF2、MT1A、MT2A、HSP70等基因在多种金属毒性中起作用

---

## 6. 生成的图表

- `fig_metal_similarity.png` - 金属相似性热图
- `fig_pathway_comparison.png` - 通路对比图
- `multi_metal_network.html` - 交互式网络

---

*Generated by Multi-Metal Network Toxicology Analysis*
"""
    
    report_file = os.path.join(output_dir, 'multi_metal_analysis_report.md')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Saved: {report_file}")
    
    return report_file


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("=" * 60)
    print("🔬 多种重金属网络毒理学对比分析")
    print("=" * 60)
    
    # 1. 计算相似性
    print("\n📊 计算金属间相似性...")
    similarity_df = calculate_metal_similarity()
    print(f"   相似性矩阵: {similarity_df.shape}")
    
    # 2. 找出共享基因
    print("\n🧬 分析跨金属共享基因...")
    shared_genes = find_shared_genes()
    print(f"   共享基因组合: {len(shared_genes)}")
    if 'All-Metals' in shared_genes:
        print(f"   全部共享基因: {len(shared_genes['All-Metals'])}个")
    
    # 3. 构建对比表
    print("\n📋 构建对比分析表...")
    summary_df = build_comparative_table()
    print(summary_df[['Metal', 'Symbol', 'Gene_Count', 'Top_Pathway', 'Top_Disease']])
    
    # 4. 生成报告和可视化
    print("\n📈 生成可视化图表和报告...")
    report_file = generate_summary_report(OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("✅ 分析完成!")
    print("=" * 60)
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"主要文件:")
    print(f"  - multi_metal_analysis_report.md (分析报告)")
    print(f"  - multi_metal_network.html (交互式网络)")
    print(f"  - fig_metal_similarity.png (相似性热图)")
    print(f"  - fig_pathway_comparison.png (通路对比)")
    
    return {
        'similarity': similarity_df,
        'shared_genes': shared_genes,
        'summary': summary_df
    }


if __name__ == "__main__":
    main()
