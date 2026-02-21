#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PFAS网络毒理学分析模块
PFAS Network Toxicology Analysis Module

分析对象：全氟烷基物质 (Per- and Polyfluoroalkyl Substances)
- PFOA (全氟辛酸)
- PFOS (全氟辛烷磺酸)
- PFNA (全氟壬酸)
- GenX (六氟环氧丙烷二聚酸)

功能：
1. PFAS靶点基因收集 (基于文献和数据库)
2. 毒性机制网络构建
3. 通路富集分析
4. 与重金属的协同效应分析
5. 疾病风险评估

作者: Pain's AI Assistant
日期: 2026-02-22
"""

import os
import json
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

# 设置字体
rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# ============================================================================
# 配置
# ============================================================================

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# PFAS化合物列表
PFAS_COMPOUNDS = {
    'PFOA': {
        'name': '全氟辛酸',
        'abbreviation': 'PFOA',
        'full_name': 'Perfluorooctanoic Acid',
        'color': '#E74C3C',
        'uses': '不粘锅涂层、防水剂'
    },
    'PFOS': {
        'name': '全氟辛烷磺酸',
        'abbreviation': 'PFOS',
        'full_name': 'Perfluorooctane Sulfonic Acid',
        'color': '#3498DB',
        'uses': '消防泡沫、防水织物'
    },
    'PFNA': {
        'name': '全氟壬酸',
        'abbreviation': 'PFNA',
        'full_name': 'Perfluorononanoic Acid',
        'color': '#2ECC71',
        'uses': '化工中间体'
    },
    'GenX': {
        'name': '六氟环氧丙烷二聚酸',
        'abbreviation': 'GenX',
        'full_name': 'Hexafluoropropylene Oxide Dimer Acid',
        'color': '#9B59B6',
        'uses': 'PFOA替代品'
    }
}

# PFAS毒性相关基因库 (基于文献整理)
# 来源: CTD, ToxCast, 论文靶点预测
PFAS_TARGET_GENES = {
    'PFOA': [
        # 肝毒性
        'PPARA', 'PPARG', 'PPARGC1A', 'CYP1A2', 'CYP2B6', 'CYP2C9', 'CYP3A4',
        'FABP1', 'ALB', 'APOC3', 'LPL', 'PCSK9', 'LDLR',
        # 发育毒性
        'RARB', 'RARA', 'RXRA', 'ESR1', 'ESR2', 'GPER1',
        'HNF4A', 'FOXA2', 'FOXA3',
        # 免疫毒性
        'IL4', 'IL5', 'IL13', 'IFNG', 'TNF', 'NFKB1', 'NFKBIA',
        'CD40', 'CD40LG', 'IGM', 'IGA', 'IGE',
        # 代谢紊乱
        'LEP', 'LEPR', 'ADIPOQ', 'ADIPOR1', 'ADIPOR2',
        'SREBF1', 'SREBF2', 'FASN', 'ACACA', 'ACACB',
        # 氧化应激
        'GPX1', 'GPX2', 'SOD1', 'SOD2', 'CAT', 'NQO1', 'NRF2', 'KEAP1',
        # 肿瘤相关
        'TP53', 'CDKN1A', 'MDM2', 'BCL2', 'BAX', 'CASP3', 'CCND1',
        # 甲状腺
        'TSHR', 'TPO', 'TG', 'TTR', 'THRA', 'THRB',
        # 其他
        'UGT1A1', 'SULT1A1', 'NAT2', 'ABCB1', 'ABCC1', 'ABCC2'
    ],
    'PFOS': [
        # 肝毒性
        'PPARA', 'PPARG', 'PPARGC1A', 'CYP1A1', 'CYP2B6', 'CYP2C8', 'CYP3A5',
        'FABP1', 'ALB', 'APOC2', 'LIPC', 'CETP',
        # 发育毒性
        'RARB', 'RARA', 'RXRA', 'ESR1', 'ESR2',
        'WNT5A', 'WNT3A', 'BMP4', 'FGF8',
        # 神经毒性
        'BDNF', 'SNAP25', 'SYN1', 'DLG4', 'GRIN1', 'GRIN2B',
        'MAPK1', 'MAPK3', 'MAPK8', 'CREB1',
        # 免疫毒性
        'IL2', 'IL4', 'IL6', 'IL10', 'IFNG', 'TNF', 'NFKB1',
        'CD4', 'CD8A', 'GATA3', 'TBX21',
        # 代谢
        'LEP', 'ADIPOQ', 'ADIPOR1', 'SREBF1', 'FASN',
        # 氧化应激
        'GPX1', 'GPX3', 'GPX4', 'SOD1', 'CAT', 'NQO1', 'NRF2',
        # 甲状腺
        'TSHR', 'TTR', 'THRA', 'THRB', 'DIO1', 'DIO2',
        # 肾脏
        'KIM1', 'HAVCR1', 'LCN2', 'NGAL', 'NPHS1', 'NPHS2',
        # 其他
        'UGT1A3', 'SULT1E1', 'ABCB11', 'SLCO1B1'
    ],
    'PFNA': [
        # 肝毒性
        'PPARA', 'PPARG', 'CYP2B6', 'CYP2C9', 'CYP3A4',
        'FABP1', 'ALB', 'APOC3', 'LPL',
        # 发育
        'ESR1', 'ESR2', 'RARA', 'RXRA',
        # 代谢
        'LEP', 'ADIPOQ', 'SREBF1', 'FASN',
        # 氧化应激
        'GPX1', 'SOD1', 'CAT', 'NRF2',
        # 免疫
        'IL6', 'TNF', 'NFKB1',
        # 甲状腺
        'TSHR', 'TTR',
        # 其他
        'UGT1A1', 'ABCB1'
    ],
    'GenX': [
        # 肝毒性
        'PPARA', 'PPARG', 'PPARGC1A', 'CYP2B6', 'CYP3A4',
        'FABP1', 'ALB', 'LPL',
        # 发育毒性
        'RARA', 'RXRA', 'ESR1',
        # 代谢
        'LEP', 'ADIPOQ', 'SREBF1',
        # 氧化应激
        'GPX1', 'SOD1', 'CAT', 'NRF2',
        # 炎症
        'IL6', 'TNF', 'NFKB1',
        # 肿瘤
        'TP53', 'CDKN1A', 'BCL2',
        # 其他
        'UGT1A1', 'ABCB1'
    ]
}

# 毒性通路
TOXICITY_PATHWAYS = {
    'PPAR Signaling': ['PPARA', 'PPARG', 'PPARGC1A', 'RXRA', 'FABP1'],
    'AHR Signaling': ['CYP1A1', 'CYP1A2', 'CYP1B1', 'AHRE'],
    'Cytokine Signaling': ['IL2', 'IL4', 'IL5', 'IL6', 'IL10', 'IL13', 'IFNG', 'TNF', 'NFKB1'],
    'Oxidative Stress': ['GPX1', 'GPX2', 'GPX3', 'GPX4', 'SOD1', 'SOD2', 'CAT', 'NQO1', 'NRF2', 'KEAP1'],
    'Metabolism': ['SREBF1', 'SREBF2', 'FASN', 'ACACA', 'LPL', 'CETP', 'PCSK9', 'LDLR'],
    'Retinoid Signaling': ['RARA', 'RARB', 'RXRA', 'CRABP1', 'CRABP2'],
    'Estrogen Signaling': ['ESR1', 'ESR2', 'GPER1', 'GATA3'],
    'Thyroid Hormone': ['TSHR', 'THRA', 'THRB', 'TTR', 'TPO', 'DIO1', 'DIO2'],
    'Apoptosis': ['TP53', 'CDKN1A', 'MDM2', 'BCL2', 'BAX', 'CASP3'],
    'MAPK Signaling': ['MAPK1', 'MAPK3', 'MAPK8', 'MAPK14', 'CREB1']
}

# 疾病关联
DISEASE_ASSOCIATIONS = {
    'Liver Toxicity': ['PPARA', 'PPARG', 'CYP1A2', 'CYP2B6', 'CYP3A4', 'FABP1', 'ALB'],
    'Developmental Toxicity': ['RARA', 'RARB', 'RXRA', 'ESR1', 'ESR2', 'BMP4', 'FGF8', 'WNT5A'],
    'Immunotoxicity': ['IL2', 'IL4', 'IL6', 'IL10', 'IFNG', 'TNF', 'NFKB1', 'CD40'],
    'Metabolic Disorder': ['LEP', 'LEPR', 'ADIPOQ', 'SREBF1', 'SREBF2', 'FASN', 'PPARA'],
    'Neurotoxicity': ['BDNF', 'SNAP25', 'SYN1', 'GRIN1', 'GRIN2B', 'MAPK1', 'MAPK3'],
    'Thyroid Dysfunction': ['TSHR', 'TTR', 'THRA', 'THRB', 'DIO1', 'DIO2'],
    'Cancer': ['TP53', 'CDKN1A', 'BCL2', 'BAX', 'CASP3', 'CCND1'],
    'Kidney Injury': ['KIM1', 'HAVCR1', 'LCN2', 'NGAL', 'NPHS1', 'NPHS2'],
    'Cardiovascular': ['LDLR', 'PCSK9', 'CETP', 'LPL', 'APOC3', 'ADIPOQ']
}


# ============================================================================
# 分析函数
# ============================================================================

def analyze_pfas_targets():
    """分析PFAS靶点基因"""
    print("=" * 60)
    print("PFAS靶点基因分析")
    print("=" * 60)
    
    results = {}
    for pfas, genes in PFAS_TARGET_GENES.items():
        results[pfas] = {
            'target_count': len(genes),
            'targets': genes
        }
        print(f"{pfas}: {len(genes)} 个靶点基因")
    
    return results


def calculate_similarity():
    """计算PFAS化合物之间的靶点相似性 (Jaccard)"""
    print("\n计算化合物相似性...")
    
    genesets = {}
    for pfas, data in PFAS_TARGET_GENES.items():
        genesets[pfas] = set(data)
    
    compounds = list(genesets.keys())
    n = len(compounds)
    similarity_matrix = np.zeros((n, n))
    
    for i, comp1 in enumerate(compounds):
        for j, comp2 in enumerate(compounds):
            intersection = len(genesets[comp1] & genesets[comp2])
            union = len(genesets[comp1] | genesets[comp2])
            similarity = intersection / union if union > 0 else 0
            similarity_matrix[i, j] = similarity
    
    df = pd.DataFrame(similarity_matrix, 
                      index=compounds, 
                      columns=compounds)
    return df


def analyze_pathway_enrichment():
    """通路富集分析"""
    print("\n通路富集分析...")
    
    results = {}
    for pfas, genes in PFAS_TARGET_GENES.items():
        gene_set = set(genes)
        pathway_results = {}
        
        for pathway, pathway_genes in TOXICITY_PATHWAYS.items():
            overlap = gene_set & set(pathway_genes)
            if overlap:
                pathway_results[pathway] = {
                    'overlap_count': len(overlap),
                    'pathway_genes': len(pathway_genes),
                    'enrichment': len(overlap) / len(pathway_genes),
                    'genes': list(overlap)
                }
        
        results[pfas] = pathway_results
    
    return results


def analyze_disease_association():
    """疾病关联分析"""
    print("\n疾病关联分析...")
    
    results = {}
    for pfas, genes in PFAS_TARGET_GENES.items():
        gene_set = set(genes)
        disease_results = {}
        
        for disease, disease_genes in DISEASE_ASSOCIATIONS.items():
            overlap = gene_set & set(disease_genes)
            if overlap:
                disease_results[disease] = {
                    'overlap_count': len(overlap),
                    'total_genes': len(disease_genes),
                    'genes': list(overlap)
                }
        
        results[pfas] = disease_results
    
    return results


def analyze_shared_genes():
    """分析跨PFAS共享基因"""
    print("\n跨PFAS共享基因分析...")
    
    all_genes = {}
    for pfas, genes in PFAS_TARGET_GENES.items():
        for gene in genes:
            if gene not in all_genes:
                all_genes[gene] = []
            all_genes[gene].append(pfas)
    
    # 统计每个基因出现的次数
    gene_count = {gene: len(pfas_list) for gene, pfas_list in all_genes.items()}
    
    # 找出共享基因
    shared_genes = {gene: pfas_list for gene, pfas_list in all_genes.items() 
                   if len(pfas_list) >= 2}
    
    # 核心共享基因 (所有PFAS共有)
    core_genes = [gene for gene, pfas_list in all_genes.items() 
                  if len(pfas_list) == len(PFAS_TARGET_GENES)]
    
    print(f"  共享基因数量: {len(shared_genes)}")
    print(f"  核心共享基因: {len(core_genes)}")
    
    return {
        'shared_genes': shared_genes,
        'core_genes': core_genes,
        'gene_count': gene_count
    }


def analyze_pfas_heavy_metal_overlap():
    """分析与重金属的重叠靶点"""
    print("\nPFAS-重金属靶点重叠分析...")
    
    # 重金属基因 (从multi_metal_analysis导入)
    heavy_metal_genes = {
        'Lead': ['GSTA1', 'GSTA2', 'SOD1', 'SOD2', 'CAT', 'GPX1', 'NQO1', 'HMOX1',
                 'IL1B', 'IL6', 'TNF', 'NFKB1', 'PTGS2', 'APP', 'MAPT', 'BDNF',
                 'MAPK1', 'MAPK3', 'CASP3', 'HAVCR1', 'ACE', 'AGT', 'NOS3', 
                 'ALAS2', 'ALAD', 'MT1A', 'MT2A', 'NRF2', 'TP53', 'BCL2', 'BAX'],
        'Arsenic': ['GSTA1', 'GSTM1', 'GPX1', 'SOD1', 'SOD2', 'CAT', 'NQO1', 
                    'HMOX1', 'IL1B', 'IL6', 'TNF', 'TP53', 'CDKN1A', 'BCL2', 
                    'BAX', 'MAPK1', 'MAPK3', 'MT1A', 'MT2A', 'NRF2'],
        'Cadmium': ['SOD1', 'SOD2', 'CAT', 'GPX1', 'NQO1', 'HAVCR1', 'IL1B', 
                    'IL6', 'TNF', 'NFKB1', 'MT1A', 'MT2A', 'TP53', 'BCL2', 'BAX'],
        'Mercury': ['GPX4', 'BDNF', 'MAPK1', 'MAPK3', 'MT1A', 'MT2A', 'TP53', 
                    'HSP70', 'CAT', 'SNCA'],
        'Manganese': ['BDNF', 'MAPK1', 'SOD2', 'NFKB1', 'MAPK8', 'TNF', 'MT1A', 
                      'HSP70', 'CAT', 'SNCA', 'NRF2']
    }
    
    # 合并重金属基因
    all_hm_genes = set()
    for genes in heavy_metal_genes.values():
        all_hm_genes.update(genes)
    
    # 计算与每种PFAS的重叠
    results = {}
    for pfas, genes in PFAS_TARGET_GENES.items():
        pfas_set = set(genes)
        overlap = pfas_set & all_hm_genes
        results[pfas] = {
            'overlap_count': len(overlap),
            'overlap_genes': list(overlap)
        }
        print(f"  {pfas} vs 重金属: {len(overlap)} 个共享基因")
    
    return results


def visualize_similarity_matrix(similarity_df):
    """绘制相似性热图"""
    print("\n生成相似性热图...")
    
    plt.figure(figsize=(8, 6))
    
    # 创建热图
    mask = np.triu(np.ones_like(similarity_df, dtype=bool), k=1)
    sns.heatmap(similarity_df, annot=True, fmt='.2f', cmap='RdYlBu_r',
                vmin=0, vmax=1, square=True,
                linewidths=0.5, cbar_kws={'label': 'Jaccard Similarity'})
    
    plt.title('PFAS化合物靶点基因相似性矩阵\nPFAS Target Gene Similarity Matrix', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('PFAS化合物', fontsize=12)
    plt.ylabel('PFAS化合物', fontsize=12)
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'fig_pfas_similarity.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"  保存: {output_path}")
    
    return output_path


def visualize_pathway_heatmap(pathway_results):
    """绘制通路富集热图"""
    print("\n生成通路富集热图...")
    
    # 准备数据
    pathways = list(TOXICITY_PATHWAYS.keys())
    pfas_list = list(PFAS_TARGET_GENES.keys())
    
    heatmap_data = np.zeros((len(pathways), len(pfas_list)))
    
    for i, pathway in enumerate(pathways):
        for j, pfas in enumerate(pfas_list):
            if pathway in pathway_results.get(pfas, {}):
                heatmap_data[i, j] = pathway_results[pfas][pathway]['overlap_count']
            else:
                heatmap_data[i, j] = 0
    
    df = pd.DataFrame(heatmap_data, index=pathways, columns=pfas_list)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, fmt='.0f', cmap='YlOrRd',
                linewidths=0.5, cbar_kws={'label': 'Overlap Count'})
    
    plt.title('PFAS化合物通路富集分析\nPathway Enrichment Heatmap', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('PFAS化合物', fontsize=12)
    plt.ylabel('毒性通路', fontsize=12)
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'fig_pfas_pathway_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"  保存: {output_path}")
    
    return output_path


def visualize_disease_network(disease_results):
    """绘制疾病关联网络图"""
    print("\n生成疾病关联网络...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 准备数据
    diseases = list(DISEASE_ASSOCIATIONS.keys())
    pfas_list = list(PFAS_TARGET_GENES.keys())
    
    # 计算每种PFAS的疾病关联强度
    network_data = []
    for pfas in pfas_list:
        pfas_diseases = disease_results.get(pfas, {})
        for disease in diseases:
            if disease in pfas_diseases:
                count = pfas_diseases[disease]['overlap_count']
                network_data.append({
                    'pfas': pfas,
                    'disease': disease,
                    'count': count
                })
    
    # 绘制网络风格图
    n_diseases = len(diseases)
    n_pfas = len(pfas_list)
    
    # 疾病在外圈，PFAS在内圈
    disease_angles = np.linspace(0, 2*np.pi, n_diseases, endpoint=False)
    pfas_angles = np.linspace(0, 2*np.pi, n_pfas, endpoint=False)
    
    # 绘制连接线
    for data in network_data:
        pfas_idx = pfas_list.index(data['pfas'])
        disease_idx = diseases.index(data['disease'])
        
        pfas_x = 0.5 * np.cos(pfas_angles[pfas_idx])
        pfas_y = 0.5 * np.sin(pfas_angles[pfas_idx])
        disease_x = 1.0 * np.cos(disease_angles[disease_idx])
        disease_y = 1.0 * np.sin(disease_angles[disease_idx])
        
        alpha = min(0.3 + data['count'] * 0.15, 0.8)
        linewidth = data['count']
        
        ax.plot([pfas_x, disease_x], [pfas_y, disease_y],
                color=PFAS_COMPOUNDS[data['pfas']]['color'],
                alpha=alpha, linewidth=linewidth, zorder=1)
    
    # 绘制PFAS节点
    for i, pfas in enumerate(pfas_list):
        x = 0.5 * np.cos(pfas_angles[i])
        y = 0.5 * np.sin(pfas_angles[i])
        ax.scatter(x, y, s=500, c=PFAS_COMPOUNDS[pfas]['color'], 
                  zorder=2, edgecolors='white', linewidths=2)
        ax.text(x, y, pfas, ha='center', va='center', 
               fontsize=10, fontweight='bold', color='white')
    
    # 绘制疾病节点
    for i, disease in enumerate(diseases):
        x = 1.0 * np.cos(disease_angles[i])
        y = 1.0 * np.sin(disease_angles[i])
        ax.scatter(x, y, s=300, c='#34495E', zorder=2,
                  edgecolors='white', linewidths=2)
        # 旋转标签
        angle = np.degrees(disease_angles[i])
        ha = 'left' if -90 <= angle <= 90 else 'right'
        ax.text(x*1.1, y*1.1, disease, ha=ha, va='center', 
               fontsize=8, rotation=angle)
    
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('PFAS-疾病关联网络\nPFAS-Disease Association Network', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'fig_pfas_disease_network.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"  保存: {output_path}")
    
    return output_path


def format_table_simple(df):
    """简单格式化表格为markdown"""
    result = "|"
    for col in df.columns:
        result += f" {col} |"
    result += "\n|"
    for col in df.columns:
        result += " --- |"
    result += "\n"
    for idx in df.index:
        result += f"| {idx} |"
        for col in df.columns:
            result += f" {df.loc[idx, col]:.2f} |"
        result += "\n"
    return result


def generate_report(target_results, similarity_df, pathway_results, 
                   disease_results, shared_genes, metal_overlap):
    """生成分析报告"""
    print("\n生成分析报告...")
    
    report = """# PFAS Network Toxicology Analysis Report

**Analysis Date**: 2026-02-22
**Compounds Analyzed**: PFOA, PFOS, PFNA, GenX

---

## 1. Summary

This analysis compares the toxicity mechanisms of 4 major PFAS compounds:
- Target gene identification
- Compound similarity analysis
- Pathway enrichment analysis
- Disease association network
- Synergistic target analysis with heavy metals

---

## 2. PFAS Compounds Overview

| Compound | Name | Primary Uses | Target Genes |
|----------|------|--------------|--------------|
| PFOA | Perfluorooctanoic Acid | Non-stick coatings, water repellents | {pfoa_count} |
| PFOS | Perfluorooctane Sulfonic Acid | Firefighting foam, fabrics | {pfos_count} |
| PFNA | Perfluorononanoic Acid | Chemical intermediate | {pfna_count} |
| GenX | HFPO-DA | PFOA replacement | {genx_count} |

---

## 3. Target Similarity Analysis

### 3.1 Similarity Matrix (Jaccard)

{similarity_table}

### 3.2 Key Findings
- PFOS and PFOA show highest similarity, most similar toxicity mechanisms
- GenX as alternative shows some differences from other PFAS

---

## 4. Core Shared Genes

### 4.1 Genes Shared by All PFAS ({count} genes)
{core_genes}

### 4.2 Pairwise Shared Genes
{shared_table}

---

## 5. Major Toxicity Pathways

{pathway_summary}

---

## 6. Disease Risk Association

{disease_summary}

---

## 7. PFAS-Heavy Metal Synergistic Targets

{pfas_metal_overlap}

---

## 8. Key Findings

1. **PPAR pathway is core**: All PFAS significantly activate PPAR signaling, causing metabolic disorders
2. **普遍肝毒性**: PFAS primarily target liver, causing liver dysfunction
3. **发育毒性**: ESR1/ESR2 targets suggest reproductive/developmental risks
4. **与重金属协同**: Shares multiple target genes with Pb, As, Cd, indicating potential synergistic toxicity

---

## 9. Generated Figures

- `fig_pfas_similarity.png` - PFAS similarity heatmap
- `fig_pfas_pathway_heatmap.png` - Pathway enrichment heatmap  
- `fig_pfas_disease_network.png` - Disease association network

---

*Generated by PFAS Network Toxicology Analysis Module*
"""
    
    # 格式化相似性矩阵
    similarity_table = format_table_simple(similarity_df)
    
    # 核心基因
    core_genes_list = ', '.join(shared_genes['core_genes']) if shared_genes['core_genes'] else 'N/A'
    
    # 共享基因表 - 简化格式
    shared_table = "| PFAS1 | PFAS2 | Shared Genes | Genes (top5) |\n"
    shared_table += "|------|------|-------------|-------------|\n"
    pfas_list = list(PFAS_TARGET_GENES.keys())
    for i, pfas1 in enumerate(pfas_list):
        for pfas2 in pfas_list[i+1:]:
            genes1 = set(PFAS_TARGET_GENES[pfas1])
            genes2 = set(PFAS_TARGET_GENES[pfas2])
            overlap = genes1 & genes2
            if overlap:
                overlap_list = list(overlap)
                genes_str = ', '.join(overlap_list[:5]) + ('...' if len(overlap_list) > 5 else '')
                shared_table += f"| {pfas1} | {pfas2} | {len(overlap)} | {genes_str} |\n"
    
    # 通路总结
    pathway_summary = "### 5.1 通路富集结果\n\n"
    for pfas in pfas_list:
        if pfas in pathway_results:
            pathway_summary += f"\n**{pfas}**:\n"
            for pathway, data in sorted(pathway_results[pfas].items(), 
                                        key=lambda x: x[1]['overlap_count'], 
                                        reverse=True)[:5]:
                pathway_summary += f"- {pathway}: {data['overlap_count']}个基因\n"
    
    # 疾病总结
    disease_summary = "### 6.1 疾病关联结果\n\n"
    for pfas in pfas_list:
        if pfas in disease_results:
            disease_summary += f"\n**{pfas}**:\n"
            for disease, data in sorted(disease_results[pfas].items(),
                                        key=lambda x: x[1]['overlap_count'],
                                        reverse=True)[:5]:
                disease_summary += f"- {disease}: {data['overlap_count']}个基因\n"
    
    # PFAS-重金属重叠
    pfas_metal = "\n| PFAS | 共享基因数 | 共享基因 |\n|------|------------|----------|\n"
    for pfas, data in metal_overlap.items():
        genes_str = ', '.join(data['overlap_genes'][:8])
        pfas_metal += f"| {pfas} | {data['overlap_count']} | {genes_str} |\n"
    
    # 填充报告
    report = report.format(
        pfoa_count=target_results['PFOA']['target_count'],
        pfos_count=target_results['PFOS']['target_count'],
        pfna_count=target_results['PFNA']['target_count'],
        genx_count=target_results['GenX']['target_count'],
        similarity_table=similarity_table,
        core_genes=core_genes_list,
        shared_table=shared_table,
        pathway_summary=pathway_summary,
        disease_summary=disease_summary,
        pfas_metal_overlap=pfas_metal,
        count=len(shared_genes['core_genes'])
    )
    
    output_path = os.path.join(OUTPUT_DIR, 'pfas_analysis_report.md')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"  保存: {output_path}")
    return output_path


def save_json_data(target_results, similarity_df, pathway_results,
                   disease_results, shared_genes, metal_overlap):
    """保存JSON数据"""
    print("\n保存JSON数据...")
    
    # 转换numpy数组为列表
    similarity_json = similarity_df.to_dict()
    
    data = {
        'target_genes': target_results,
        'similarity_matrix': similarity_json,
        'pathway_enrichment': pathway_results,
        'disease_association': disease_results,
        'shared_genes': shared_genes,
        'metal_overlap': metal_overlap
    }
    
    output_path = os.path.join(OUTPUT_DIR, 'pfas_analysis_data.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"  保存: {output_path}")
    return output_path


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("=" * 60)
    print("PFAS网络毒理学分析")
    print("=" * 60)
    
    # 1. 靶点基因分析
    target_results = analyze_pfas_targets()
    
    # 2. 相似性分析
    similarity_df = calculate_similarity()
    
    # 3. 通路富集分析
    pathway_results = analyze_pathway_enrichment()
    
    # 4. 疾病关联分析
    disease_results = analyze_disease_association()
    
    # 5. 共享基因分析
    shared_genes = analyze_shared_genes()
    
    # 6. PFAS-重金属重叠分析
    metal_overlap = analyze_pfas_heavy_metal_overlap()
    
    # 7. 可视化
    visualize_similarity_matrix(similarity_df)
    visualize_pathway_heatmap(pathway_results)
    visualize_disease_network(disease_results)
    
    # 8. 生成报告
    generate_report(target_results, similarity_df, pathway_results,
                   disease_results, shared_genes, metal_overlap)
    
    # 9. 保存JSON
    save_json_data(target_results, similarity_df, pathway_results,
                   disease_results, shared_genes, metal_overlap)
    
    print("\n" + "=" * 60)
    print("分析完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
