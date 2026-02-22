#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基因集富集分析模块 (GSEA) - 铅神经毒性通路分析
Gene Set Enrichment Analysis: Lead Neurotoxicity Pathways

功能：
1. 超几何检验富集分析
2. 预定义基因集（KEGG, GO, Reactome）
3. 铅毒性相关自定义通路
4. 可视化：条形图、热图、网络图

作者: Pain's AI Assistant
日期: 2026-02-23
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import stats
from scipy.stats import hypergeom, fisher_exact
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 配置
# ============================================================================

rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# 铅毒性相关基因集定义
# ============================================================================

# 铅神经毒性相关通路
LEAD_NEUROTOXICITY_PATHWAYS = {
    "Oxidative Stress Response": {
        "genes": ["SOD1", "SOD2", "CAT", "GPX1", "GPX4", "GPX6", "NQO1", "HMOX1", "HMOX2", 
                  "GSTA1", "GSTA2", "GSTM1", "GSTM2", "GSTP1", "MT1A", "MT2A", "MT3"],
        "description": "Response to oxidative stress and ROS detoxification"
    },
    "Neuroinflammation": {
        "genes": ["IL1B", "IL6", "TNF", "NFKB1", "NFKB2", "RELA", "PTGS2", "COX2", "CXCL8", 
                  "CXCL10", "CCL2", "CCL5", "TLR2", "TLR4", "MYD88", "MAPK1", "MAPK3"],
        "description": "Inflammatory response in neural cells"
    },
    "Apoptosis Signaling": {
        "genes": ["BCL2", "BCL2L1", "BAX", "BAD", "BAK1", "CASP3", "CASP8", "CASP9", 
                  "TP53", "PMAIP1", "BBC3", "FAS", "FASLG", "TNFRSF1A", "TNFRSF1B"],
        "description": "Programmed cell death pathways"
    },
    "Calcium Signaling": {
        "genes": ["CALM1", "CALM2", "CALB1", "CAMK2A", "CAMK2B", "PPP3CA", "PPP3CB", 
                  "CALCR", "CALR", "CASQ2", "RCAN1", "NCOR1", "S100B", "GFAP"],
        "description": "Calcium homeostasis disruption by lead"
    },
    "Synaptic Function": {
        "genes": ["SNAP25", "SYN1", "SYN2", "SYN3", "PSD95", "DLG4", "NLGN1", "NRXN1", 
                  "NRXN2", "NRXN3", "SHANK2", "SHANK3", "GRIN1", "GRIN2A", "GRIK2"],
        "description": "Synaptic transmission and plasticity"
    },
    "Blood-Brain Barrier": {
        "genes": ["CLDN5", "OCLN", "TJP1", "TJP2", "FLT1", "VWF", "CDH5", "PECAM1", 
                  "CAV1", "CAV2", "ABCB1", "ABCG2", "SLC2A1", "SLC7A5"],
        "description": "BBB integrity and transport"
    },
    "Metal Ion Transport": {
        "genes": ["SLC11A2", "SLC39A8", "ATP13A2", "ATP1A1", "ATP1A3", "MT1A", "MT2A", 
                  "MT3", "SLC30A1", "SLC30A3", "SLC30A10", "SLC40A1", "FPN1"],
        "description": "Metal ion homeostasis and transport"
    },
    "Neurodevelopment": {
        "genes": ["BDNF", "NT3", "NT4", "NGF", "NRG1", "RELN", "DCX", "TUBB3", "MAP2", 
                  "NES", "SOX2", "POU5F1", "NOTCH1", "JAG1", "DLL1"],
        "description": "Neural development and differentiation"
    },
    "DNA Repair": {
        "genes": ["TP53", "XPA", "XPC", "ERCC1", "ERCC2", "ERCC3", "ERCC4", "OGG1", 
                  "MUTYH", "PARP1", "PARP2", "RAD51", "BRCA1", "BRCA2"],
        "description": "DNA damage response and repair"
    },
    "Autophagy": {
        "genes": ["BECN1", "MAP1LC3A", "MAP1LC3B", "ATG5", "ATG7", "ATG12", "SQSTM1", 
                  "ULK1", "ULK2", "AMBRA1", "Bclin1", "ATG14", "WIPI1", "WIPI2"],
        "description": "Autophagy regulation in neurotoxicity"
    },
    "Mitochondrial Function": {
        "genes": ["MT-CO1", "MT-CO2", "MT-CO3", "MT-CYB", "MT-ATP6", "ND1", "ND2", "ND4", 
                  "ND5", "ND6", "COX4I1", "COX5A", "ATP5A1", "TFAM", "PGC1A", "PPARGC1A"],
        "description": "Mitochondrial dysfunction in lead toxicity"
    },
    "Glutamatergic Signaling": {
        "genes": ["GRIN1", "GRIN2A", "GRIN2B", "GRIN2C", "GRIA1", "GRIA2", "GRIA3", "GRIA4",
                  "GRIK1", "GRIK2", "GRIK3", "GRIK4", "GRIK5", "SLC1A1", "SLC1A2", "EAAT1", "EAAT2"],
        "description": "Glutamate excitotoxicity"
    },
    "GABAergic Signaling": {
        "genes": ["GABRA1", "GABRA2", "GABRA3", "GABRA4", "GABRA5", "GABRB1", "GABRB2", 
                  "GABRB3", "GABRG1", "GABRG2", "GABRG3", "GAD1", "GAD2", "SLC6A11", "SLC6A13"],
        "description": "GABAergic neurotransmission"
    },
    "Dopaminergic Signaling": {
        "genes": ["TH", "DDC", "DAT1", "SLC6A3", "DRD1", "DRD2", "DRD3", "DRD4", "DRD5",
                  "COMT", "MAOA", "MAOB", "SLC18A1", "SLC18A2", "VMAT1", "VMAT2"],
        "description": "Dopamine metabolism and signaling"
    },
    "Cholinergic Signaling": {
        "genes": ["CHAT", "SLC5A7", "CHRNA1", "CHRNA2", "CHRNA3", "CHRNA4", "CHRNA7",
                  "CHRNB1", "CHRND", "ACHE", "BCHE", "SLC18A3", "VAChT"],
        "description": "Acetylcholine neurotransmission"
    },
}

# KEGG通路（简化版）
KEGG_PATHWAYS = {
    "hsa05010": {"name": "Alzheimer's disease", "genes": ["APP", "MAPT", "PSEN1", "PSEN2", "APOE", "SNCA", "LRP1"]},
    "hsa05012": {"name": "Parkinson's disease", "genes": ["SNCA", "PARK1", "PARK2", "PARK6", "LRRK2", "DJ1", "PINK1"]},
    "hsa05014": {"name": "Amyotrophic lateral sclerosis", "genes": ["SOD1", "TARDBP", "FUS", "C9orf72", "ALS1", "ALS2"]},
    "hsa05016": {"name": "Huntington's disease", "genes": ["HTT", "BDNF", "CABD2", "ATXN1", "ATXN3", "MJD1"]},
    "hsa05020": {"name": "Prion disease", "genes": ["PRNP", "PRND", "PRNT", "APP", "NCAM1"]},
    "hsa04713": {"name": "Circadian rhythm", "genes": ["CLOCK", "BMAL1", "PER1", "PER2", "PER3", "CRY1", "CRY2"]},
    "hsa04720": {"name": "Long-term potentiation", "genes": ["GRIN1", "GRIN2A", "CAMK2A", "CREB1", "MAPK1", "MAPK3"]},
    "hsa04721": {"name": "Long-term depression", "genes": ["GRM1", "GRM5", "PPP1R1A", "PPP1CA", "MAPK1", "MAPK3"]},
    "hsa04722": {"name": "Neurotrophin signaling", "genes": ["NGF", "BDNF", "NT3", "NTRK1", "NTRK2", "PIK3R1", "AKT1"]},
    "hsa04080": {"name": "Neuroactive ligand-receptor", "genes": ["GABRA1", "GLRA1", "GRIN1", "DRD2", "HTR1A", "ADRA1A"]},
    "hsa04010": {"name": "MAPK signaling", "genes": ["MAPK1", "MAPK3", "MAPK8", "MAPK9", "MAPK10", "MAP2K1", "MAP2K2"]},
    "hsa04014": {"name": "Ras signaling", "genes": ["RASGRP1", "RASGRP2", "KRAS", "NRAS", "RAF1", "PIK3CA"]},
    "hsa04064": {"name": "NF-kappa B signaling", "genes": ["NFKB1", "NFKB2", "RELA", "REL", "IKBKB", "IKBKG"]},
    "hsa04210": {"name": "Apoptosis", "genes": ["BCL2", "BCL2L1", "BAX", "CASP3", "CASP9", "TP53", "FAS"]},
    "hsa04137": {"name": "Mitochondrial apoptosis", "genes": ["BAX", "BAK1", "BCL2", "BCL2L1", "CYCS", "CASP9"]},
    "hsa04140": {"name": "Autophagy", "genes": ["BECN1", "MAP1LC3A", "ATG5", "ATG7", "SQSTM1", "ULK1"]},
    "hsa00190": {"name": "Oxidative phosphorylation", "genes": ["ND1", "ND2", "COX1", "COX2", "ATP6", "ATP5A1"]},
    "hsa00280": {"name": "Valine, leucine degradation", "genes": ["BCAT1", "BCKDHA", "DBT", "DLD", "PDHA1"]},
    "hsa00480": {"name": "Glutathione metabolism", "genes": ["GSTA1", "GSTM1", "GSS", "GPX1", "GPX2", "GPX4"]},
    "hsa00860": {"name": "Porphyrin metabolism", "genes": ["ALAS1", "ALAS2", "FECH", "UROD", "UROS", "CPOX"]},
}

# ============================================================================
# 富集分析函数
# ============================================================================

def hypergeometric_test(query_genes, gene_set_genes, universe_size):
    """
    超几何检验
    
    参数:
        query_genes: 差异表达基因列表
        gene_set_genes: 基因集中的基因列表
        universe_size: 背景基因总数
    
    返回:
        p_value, overlap_count, overlap_genes
    """
    # 计算交集
    query_set = set(query_genes)
    gene_set = set(gene_set_genes)
    overlap = query_set & gene_set
    overlap_genes = list(overlap)
    overlap_count = len(overlap)
    
    if overlap_count == 0:
        return 1.0, 0, []
    
    # 超几何检验
    # M = universe_size (背景基因数)
    # n = len(gene_set_genes) (通路中的基因数)
    # N = len(query_genes) (查询基因数)
    # k = overlap_count (交集数)
    
    M = universe_size
    n = len(gene_set_genes)
    N = len(query_genes)
    k = overlap_count
    
    # P(X >= k) = 1 - P(X < k)
    p_value = 1 - hypergeom.cdf(k - 1, M, n, N)
    
    return p_value, overlap_count, overlap_genes


def enrich_with_fisher(query_genes, gene_set_genes, universe_genes):
    """
    Fisher精确检验
    
    参数:
        query_genes: 查询基因列表
        gene_set_genes: 基因集基因列表
        universe_genes: 背景基因列表
    
    返回:
        p_value, odds_ratio, overlap
    """
    query_set = set(query_genes)
    gene_set = set(gene_set_genes)
    universe_set = set(universe_genes)
    
    # 构建2x2列联表
    #              基因集内    基因集外
    # 查询基因      a          b
    # 非查询基因    c          d
    
    a = len(query_set & gene_set)
    b = len(query_set - gene_set)
    c = len(gene_set - query_set)
    d = len(universe_set - query_set - gene_set)
    
    # 确保所有值非负
    if a < 0 or b < 0 or c < 0 or d < 0:
        return 1.0, 0.0, 0
    
    # Fisher精确检验
    try:
        table = [[a, b], [c, d]]
        odds_ratio, p_value = fisher_exact(table)
        return p_value, odds_ratio, a
    except:
        return 1.0, 0.0, a


def perform_gsea(query_genes, pathway_dict, universe_size=20000, method='hypergeometric'):
    """
    执行基因集富集分析
    
    参数:
        query_genes: 差异表达/关注基因列表
        pathway_dict: 通路字典 {通路名: {genes: [...]}}
        universe_size: 背景基因总数
        method: 'hypergeometric' 或 'fisher'
    
    返回:
        结果DataFrame
    """
    results = []
    
    for pathway_name, pathway_info in pathway_dict.items():
        gene_set_genes = pathway_info.get("genes", [])
        description = pathway_info.get("description", "")
        
        if not gene_set_genes:
            continue
        
        # 统一基因名为大写
        query_upper = [g.upper() for g in query_genes]
        gene_set_upper = [g.upper() for g in gene_set_genes]
        
        if method == 'hypergeometric':
            p_value, overlap_count, overlap_genes = hypergeometric_test(
                query_upper, gene_set_upper, universe_size
            )
            odds_ratio = overlap_count / (len(gene_set_upper) + 1)
        else:
            universe_genes_list = [f"Gene_{i}" for i in range(universe_size)]
            p_value, odds_ratio, overlap_count = enrich_with_fisher(
                query_upper, gene_set_upper, universe_genes_list
            )
            overlap_genes = list(set(query_upper) & set(gene_set_upper))
        
        # 计算富集倍数
        enrichment_ratio = overlap_count / (len(gene_set_genes) + 1)
        
        results.append({
            'Pathway': pathway_name,
            'Description': description,
            'Pathway_Genes': len(gene_set_genes),
            'Overlap_Genes': overlap_count,
            'Overlap_Ratio': enrichment_ratio,
            'P_Value': p_value,
            'Log_P_Value': -np.log10(p_value + 1e-10),
            'Odds_Ratio': odds_ratio,
            'Genes': ', '.join(overlap_genes[:10]),  # 最多显示10个
        })
    
    # 转为DataFrame并排序
    df = pd.DataFrame(results)
    if len(df) > 0:
        df = df.sort_values('P_Value')
        
        # FDR校正（Benjamini-Hochberg）
        from scipy.stats import false_discovery_control
        df['P_Adj'] = false_discovery_control(df['P_Value'].values, method='bh')
    
    return df


def perform_lead_pathway_analysis(query_genes, universe_size=20000):
    """
    专门针对铅神经毒性的通路分析
    """
    print("\n=== 铅神经毒性通路富集分析 ===")
    print(f"查询基因数: {len(query_genes)}")
    
    # 1. 铅特异性通路
    print("\n分析铅特异性通路...")
    lead_results = perform_gsea(query_genes, LEAD_NEUROTOXICITY_PATHWAYS, universe_size)
    
    # 2. KEGG神经疾病通路
    print("分析KEGG神经疾病通路...")
    kegg_results = perform_gsea(query_genes, KEGG_PATHWAYS, universe_size)
    
    return lead_results, kegg_results


# ============================================================================
# 可视化函数
# ============================================================================

def plot_enrichment_barplot(df, title, filename, top_n=15, color='#E74C3C'):
    """
    绘制富集分析条形图
    """
    if len(df) == 0:
        print(f"  无数据可绘图: {title}")
        return
    
    df_top = df.head(top_n).copy()
    df_top = df_top.sort_values('Log_P_Value')
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(df_top) * 0.4)))
    
    colors = [color if p < 0.05 else '#95A5A6' for p in df_top['P_Value']]
    
    bars = ax.barh(range(len(df_top)), df_top['Log_P_Value'], color=colors, edgecolor='white')
    
    ax.set_yticks(range(len(df_top)))
    ax.set_yticklabels(df_top['Pathway'], fontsize=9)
    ax.set_xlabel('-log10(P-value)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    
    # 添加显著性标记
    for i, (idx, row) in enumerate(df_top.iterrows()):
        if row['P_Value'] < 0.001:
            marker = '***'
        elif row['P_Value'] < 0.01:
            marker = '**'
        elif row['P_Value'] < 0.05:
            marker = '*'
        else:
            marker = ''
        
        if marker:
            ax.text(row['Log_P_Value'] + 0.1, i, marker, va='center', fontsize=10)
    
    # 添加FDR阈值线
    ax.axvline(x=-np.log10(0.05), color='gray', linestyle='--', alpha=0.7, label='P=0.05')
    ax.axvline(x=-np.log10(0.01), color='gray', linestyle=':', alpha=0.7, label='P=0.01')
    
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {filename}")


def plot_enrichment_network(df, title, filename, min_pvalue=0.1):
    """
    绘制富集分析网络图（简化版）
    """
    if len(df) == 0:
        print(f"  无数据可绘图: {title}")
        return
    
    # 过滤显著的结果
    df_sig = df[df['P_Value'] < min_pvalue].head(20)
    
    if len(df_sig) < 2:
        print(f"  显著通路不足，跳过网络图")
        return
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 创建气泡图
    sizes = df_sig['Overlap_Genes'] * 50
    colors = -np.log10(df_sig['P_Value'] + 1e-10)
    
    scatter = ax.scatter(
        range(len(df_sig)),
        df_sig['Log_P_Value'],
        s=sizes,
        c=colors,
        cmap='YlOrRd',
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5
    )
    
    ax.set_xticks(range(len(df_sig)))
    ax.set_xticklabels(df_sig['Pathway'], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('-log10(P-value)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('-log10(P-value)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {filename}")


def plot_pathway_heatmap(lead_results, kegg_results, query_genes, filename):
    """
    绘制通路-基因热图
    """
    # 合并结果
    all_results = pd.concat([lead_results, kegg_results], ignore_index=True)
    all_results = all_results[all_results['P_Value'] < 0.1].head(15)
    
    if len(all_results) == 0:
        print("  无显著通路，跳过热图")
        return
    
    # 构建矩阵
    pathways = all_results['Pathway'].tolist()
    query_upper = [g.upper() for g in query_genes]
    
    matrix = []
    gene_labels = set()
    for _, row in all_results.iterrows():
        pathway_genes = [g.upper() for g in LEAD_NEUROTOXICITY_PATHWAYS.get(row['Pathway'], {}).get('genes', [])]
        gene_labels.update(pathway_genes)
    
    gene_list = sorted(list(gene_labels))[:50]  # 限制基因数
    
    matrix = np.zeros((len(pathways), len(gene_list)))
    for i, pathway in enumerate(pathways):
        pathway_genes = [g.upper() for g in LEAD_NEUROTOXICITY_PATHWAYS.get(pathway, {}).get('genes', [])]
        for j, gene in enumerate(gene_list):
            if gene in pathway_genes and gene in query_upper:
                matrix[i, j] = 1
    
    # 绘制热图
    fig, ax = plt.subplots(figsize=(max(12, len(gene_list) * 0.2), max(6, len(pathways) * 0.4)))
    
    cmap = matplotlib.colors.ListedColormap(['#F5F5F5', '#E74C3C'])
    im = ax.imshow(matrix, aspect='auto', cmap=cmap, interpolation='nearest')
    
    ax.set_xticks(range(len(gene_list)))
    ax.set_xticklabels(gene_list, rotation=90, fontsize=7)
    ax.set_yticks(range(len(pathways)))
    ax.set_yticklabels(pathways, fontsize=9)
    
    ax.set_title('Pathway-Gene Association Heatmap', fontsize=13, fontweight='bold')
    ax.set_xlabel('Genes', fontsize=11)
    ax.set_ylabel('Pathways', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {filename}")


def plot_pathway_comparison(lead_results, kegg_results, filename):
    """
    绘制通路富集比较图
    """
    if len(lead_results) == 0 and len(kegg_results) == 0:
        print("  无数据可绘图")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 铅特异性通路
    if len(lead_results) > 0:
        df = lead_results.head(10).sort_values('Log_P_Value')
        colors = ['#E74C3C' if p < 0.05 else '#BDC3C7' for p in df['P_Value']]
        axes[0].barh(range(len(df)), df['Log_P_Value'], color=colors)
        axes[0].set_yticks(range(len(df)))
        axes[0].set_yticklabels(df['Pathway'], fontsize=9)
        axes[0].set_xlabel('-log10(P-value)')
        axes[0].set_title('Lead Neurotoxicity Pathways', fontweight='bold')
        axes[0].axvline(x=-np.log10(0.05), color='gray', linestyle='--', alpha=0.7)
    
    # KEGG通路
    if len(kegg_results) > 0:
        df = kegg_results.head(10).sort_values('Log_P_Value')
        colors = ['#3498DB' if p < 0.05 else '#BDC3C7' for p in df['P_Value']]
        axes[1].barh(range(len(df)), df['Log_P_Value'], color=colors)
        axes[1].set_yticks(range(len(df)))
        axes[1].set_yticklabels(df['Pathway'].tolist(), fontsize=9)
        axes[1].set_xlabel('-log10(P-value)')
        axes[1].set_title('KEGG Neurological Pathways', fontweight='bold')
        axes[1].axvline(x=-np.log10(0.05), color='gray', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {filename}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """
    主函数：执行完整的基因集富集分析
    """
    print("=" * 60)
    print("基因集富集分析 (GSEA) - 铅神经毒性通路")
    print("=" * 60)
    
    # 示例基因列表（铅暴露相关的差异表达基因）
    # 这些基因来源于文献和数据库
    demo_genes = [
        # 氧化应激
        "SOD1", "SOD2", "CAT", "GPX1", "GPX4", "NQO1", "HMOX1", "MT1A", "MT2A",
        # 神经炎症
        "IL1B", "IL6", "TNF", "NFKB1", "PTGS2", "CXCL8", "CXCL10", "CCL2",
        # 凋亡
        "BCL2", "BAX", "CASP3", "CASP9", "TP53", "BBC3", "FAS",
        # 钙信号
        "CALM1", "CALM2", "CAMK2A", "CAMK2B", "S100B", "GFAP",
        # 突触功能
        "SNAP25", "SYN1", "PSD95", "DLG4", "NLGN1", "NRXN1",
        # BBB
        "CLDN5", "OCLN", "TJP1", "FLT1", "VWF", "ABCB1",
        # 金属转运
        "SLC11A2", "SLC39A8", "ATP13A2", "ATP1A1", "SLC30A1",
        # 神经发育
        "BDNF", "NT3", "DCX", "MAP2", "NES", "SOX2", "NOTCH1",
        # DNA修复
        "XPA", "XPC", "ERCC1", "OGG1", "PARP1", "RAD51",
        # 自噬
        "BECN1", "MAP1LC3B", "ATG5", "SQSTM1", "ULK1",
        # 神经递质
        "GRIN1", "GRIN2A", "GABRA1", "TH", "DAT1", "DRD2", "CHAT",
    ]
    
    print(f"\n分析 {len(demo_genes)} 个铅相关基因...")
    
    # 执行富集分析
    lead_results, kegg_results = perform_lead_pathway_analysis(demo_genes)
    
    # 保存结果
    if len(lead_results) > 0:
        output_file = os.path.join(OUTPUT_DIR, "gsea_lead_pathways.csv")
        lead_results.to_csv(output_file, index=False)
        print(f"\n铅毒性通路结果已保存: {output_file}")
        
        # 打印显著通路
        print("\n=== 显著铅毒性通路 (P < 0.05) ===")
        sig = lead_results[lead_results['P_Value'] < 0.05]
        for _, row in sig.iterrows():
            print(f"  {row['Pathway']}: P={row['P_Value']:.4f}, "
                  f"Overlap={row['Overlap_Genes']}/{row['Pathway_Genes']}")
    
    if len(kegg_results) > 0:
        output_file = os.path.join(OUTPUT_DIR, "gsea_kegg_pathways.csv")
        kegg_results.to_csv(output_file, index=False)
        print(f"\nKEGG通路结果已保存: {output_file}")
    
    # 生成可视化
    print("\n生成可视化...")
    
    # 条形图
    plot_enrichment_barplot(
        lead_results, 
        "Lead Neurotoxicity Pathway Enrichment",
        os.path.join(OUTPUT_DIR, "fig_gsea_lead_pathways.png"),
        color='#E74C3C'
    )
    
    plot_enrichment_barplot(
        kegg_results,
        "KEGG Neurological Pathway Enrichment",
        os.path.join(OUTPUT_DIR, "fig_gsea_kegg_pathways.png"),
        color='#3498DB'
    )
    
    # 网络图
    plot_enrichment_network(
        lead_results,
        "Lead Pathway Network",
        os.path.join(OUTPUT_DIR, "fig_gsea_network.png")
    )
    
    # 热图
    plot_pathway_heatmap(
        lead_results, kegg_results, demo_genes,
        os.path.join(OUTPUT_DIR, "fig_gsea_heatmap.png")
    )
    
    # 比较图
    plot_pathway_comparison(
        lead_results, kegg_results,
        os.path.join(OUTPUT_DIR, "fig_gsea_comparison.png")
    )
    
    print("\n" + "=" * 60)
    print("基因集富集分析完成!")
    print("=" * 60)
    
    return lead_results, kegg_results


if __name__ == "__main__":
    main()
