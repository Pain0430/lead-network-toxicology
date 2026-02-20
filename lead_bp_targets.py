#!/usr/bin/env python3
"""
铅与血压/CKM关键靶点分析
Network Toxicology + Molecular Docking + Target Prediction

目标:
1. 识别铅诱导高血压的关键靶点
2. 分子对接预测结合位点
3. 为小分子干预提供线索
"""

import requests
import json
import pandas as pd
import numpy as np
from collections import defaultdict
import os

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 血压调控相关基因 (基于文献和通路分析)
BLOOD_PRESSURE_TARGETS = {
    # 肾素-血管紧张素系统 (RAS) - 核心!
    "RAAS": {
        "genes": ["REN", "AGT", "ACE", "ACE2", "AGTR1", "AGTR2", "AGTRAP", "MAS1", "LRP1", "NPPA", "NPPB"],
        "description": "肾素-血管紧张素-醛固酮系统",
        "pathway": "血压调节核心通路"
    },
    # 氧化应激
    "Oxidative_Stress": {
        "genes": ["NOS3", "NOS2", "NOS1", "SOD1", "SOD2", "CAT", "GPX1", "NQO1", "HMOX1", "CYBA", "NCF1"],
        "description": "氧化应激与血管功能",
        "pathway": "NO生物利用度下降"
    },
    # 炎症反应
    "Inflammation": {
        "genes": ["IL1B", "IL6", "TNF", "NFKB1", "NFKB2", "PTGS2", "COX2", "ICAM1", "VCAM1", "SELE"],
        "description": "血管炎症反应",
        "pathway": "内皮功能障碍"
    },
    # 钙信号
    "Calcium": {
        "genes": ["CALM1", "CALM2", "CALM3", "CALML4", "CALML5", "CALD1", "MYL6", "MYH11", "ACTA2", "CNN1"],
        "description": "钙信号与血管收缩",
        "pathway": "血管平滑肌收缩"
    },
    # 内皮功能
    "Endothelial": {
        "genes": ["EDN1", "EDNRA", "EDNRB", "ECE1", "ECE2", "VEGFA", "KDR", "FLT1", "PECAM1", "CDH5"],
        "description": "内皮功能调节",
        "pathway": "血管舒缩"
    },
    # 交感神经
    "Sympathetic": {
        "genes": ["ADRA1A", "ADRA1B", "ADRA1D", "ADRA2A", "ADRA2B", "ADRB1", "ADRB2", "DBH", "PNMT", "TH"],
        "description": "交感神经系统",
        "pathway": "血管收缩"
    }
}

def get_protein_structure(uniprot_id):
    """从UniProt获取蛋白结构信息"""
    try:
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data
    except:
        pass
    return None

def get_pdb_structure(gene_name):
    """搜索PDB中的蛋白结构"""
    try:
        url = "https://search.rcsb.org/rcsbsearch/v1/query"
        query = {
            "query": {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "value": gene_name
                }
            },
            "request_options": {
                "return_num": 3
            },
            "sort": [{"sort_by": "score", "direction": "desc"}]
        }
        response = requests.post(url, json=query, timeout=15)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"   PDB搜索错误: {e}")
    return None

def predict_binding_sites(gene_name):
    """
    预测蛋白的潜在结合位点
    基于文献和已知位点数据库
    """
    
    # 已知铅结合位点/功能域
    known_bindings = {
        # ACE - 金属蛋白，含锌离子结合位点
        "ACE": {
            "type": "Metalloproteinase",
            "metal_binding": "Zn2+ (HEMICINE)",
            "active_site": "HEXXH motif",
            "inhibitors": "Captopril, Lisinopril (结合Zn2+位点)",
            "pdb_available": True,
            "note": "ACE抑制剂通过竞争性结合Zn2+位点发挥作用"
        },
        # NOS - 一氧化氮合酶
        "NOS3": {
            "type": "Oxidoreductase", 
            "metal_binding": "Zn2+, Fe2+ (heme)",
            "cofactor": "BH4 (四氢生物蝶呤)",
            "inhibitors": "L-NAME, L-NMMA",
            "pdb_available": True,
            "note": "铅可能取代Zn2+或Fe2+，干扰NO合成"
        },
        # 肾素
        "REN": {
            "type": "Aspartic protease",
            "active_site": "Aspartyl residue",
            "inhibitors": "Aliskiren",
            "pdb_available": True,
            "note": "直接肾素抑制剂"
        },
        # AGT (血管紧张素原)
        "AGT": {
            "type": "Serpin family",
            "cleavage_sites": ["Renin site", "ACE site"],
            "pdb_available": True,
            "note": "是肾素和ACE的底物"
        },
        # AGTR1 (血管紧张素II受体)
        "AGTR1": {
            "type": "GPCR (7TM)",
            "signal": "Gq/11 protein",
            "blockers": "Losartan, Valsartan (ARB类药物)",
            "pdb_available": True,
            "note": "AT1受体拮抗剂(沙坦类)是常用降压药"
        },
        # 炎症因子
        "IL1B": {
            "type": "Cytokine",
            "receptor": "IL1R1/IL1R2",
            "inhibitors": "Anakinra (IL-1受体拮抗剂)",
            "pdb_available": True,
            "note": "IL-1β阻断剂用于炎症治疗"
        },
        # 肿瘤坏死因子
        "TNF": {
            "type": "Cytokine",
            "inhibitors": "Etanercept, Infliximab (TNF-α抑制剂)",
            "pdb_available": True,
            "note": "单克隆抗体用于自身免疫疾病"
        },
        # NFKB
        "NFKB1": {
            "type": "Transcription factor",
            "inhibitors": "BAY 11-7082, IKK inhibitor",
            "pdb_available": True,
            "note": "NF-κB是炎症信号核心转录因子"
        },
        # SOD1
        "SOD1": {
            "type": "Oxidoreductase",
            "metal_binding": "Cu+, Zn2+",
            "mutations": "与肌萎缩侧索硬化相关",
            "pdb_available": True,
            "note": "铅可能取代Zn2+，导致SOD失活"
        },
        # CAT (过氧化氢酶)
        "CAT": {
            "type": "Oxidoreductase", 
            "metal_binding": "Heme (Fe)",
            "pdb_available": True,
            "note": "铅可能干扰血红素合成"
        }
    }
    
    return known_bindings.get(gene_name, {
        "type": "Unknown",
        "pdb_available": None,
        "note": "需要进一步研究"
    })

def analyze_key_targets():
    """分析关键靶点"""
    
    print("="*60)
    print("🔬 铅诱导高血压关键靶点分析")
    print("="*60)
    
    # 收集所有靶点
    all_targets = []
    for pathway, info in BLOOD_PRESSURE_TARGETS.items():
        for gene in info["genes"]:
            all_targets.append({
                "Gene": gene,
                "Pathway": pathway,
                "Pathway_Description": info["description"],
                "Function": info["pathway"]
            })
    
    # 去重
    df_targets = pd.DataFrame(all_targets).drop_duplicates(subset=['Gene'])
    print(f"\n📊 识别到 {len(df_targets)} 个血压调控相关基因")
    
    # 预测结合位点
    binding_info = []
    for gene in df_targets["Gene"]:
        info = predict_binding_sites(gene)
        info["Gene"] = gene
        binding_info.append(info)
    
    df_binding = pd.DataFrame(binding_info)
    df_full = df_targets.merge(df_binding, on="Gene")
    
    # 保存结果
    df_full.to_csv(f"{OUTPUT_DIR}/lead_bp_key_targets.csv", index=False)
    
    # 打印核心靶点
    print("\n" + "="*60)
    print("🎯 核心靶点及干预线索")
    print("="*60)
    
    priority_targets = ["ACE", "NOS3", "REN", "AGT", "AGTR1", "SOD1", "CAT", "IL1B", "TNF", "NFKB1"]
    
    for gene in priority_targets:
        row = df_full[df_full["Gene"] == gene]
        if len(row) > 0:
            row = row.iloc[0]
            print(f"\n🔴 {gene} ({row['Pathway']})")
            print(f"   类型: {row.get('type', 'N/A')}")
            if pd.notna(row.get('metal_binding')):
                print(f"   金属结合位点: {row['metal_binding']}")
            if pd.notna(row.get('inhibitors')):
                print(f"   现有抑制剂: {row['inhibitors']}")
            if pd.notna(row.get('note')):
                print(f"   备注: {row['note']}")
    
    return df_full

def search_pdb_structures():
    """搜索PDB结构"""
    
    print("\n" + "="*60)
    print("🔍 PDB蛋白结构搜索")
    print("="*60)
    
    priority_genes = ["ACE", "NOS3", "REN", "AGT", "AGTR1", "IL1B", "SOD1", "CAT"]
    pdb_results = []
    
    for gene in priority_genes:
        print(f"\n搜索 {gene}...")
        result = get_pdb_structure(gene)
        if result and "result_set" in result:
            structures = result.get("result_set", {}).get("results", [])
            if structures:
                for s in structures[:2]:  # 取前2个
                    pdb_results.append({
                        "Gene": gene,
                        "PDB_ID": s.get("rcsb_id"),
                        "Title": s.get("title", "")[:100]
                    })
                    print(f"   ✅ {s.get('rcsb_id')}: {s.get('title', '')[:50]}")
            else:
                print(f"   ❌ 无PDB结构")
    
    if pdb_results:
        df_pdb = pd.DataFrame(pdb_results)
        df_pdb.to_csv(f"{OUTPUT_DIR}/pdb_structures.csv", index=False)
        print(f"\n✅ 已保存到 {OUTPUT_DIR}/pdb_structures.csv")
    
    return pdb_results

def generate_intervention_summary(df_targets):
    """生成干预建议摘要"""
    
    print("\n" + "="*60)
    print("💊 铅诱导高血压的干预策略")
    print("="*60)
    
    strategies = """
## 一、基于靶点的干预策略

### 1. 肾素-血管紧张素系统 (RAS) 抑制
| 靶点 | 策略 | 现有药物 | 备注 |
|------|------|----------|------|
| ACE | ACE抑制剂 | 卡托普利、赖诺普利 | 经典降压药 |
| AGTR1 | ARB受体拮抗剂 | 氯沙坦、缬沙坦 | 沙坦类 |
| REN | 直接肾素抑制剂 | 阿利吉仑 | 较新 |

### 2. 抗氧化治疗
| 靶点 | 策略 | 候选化合物 | 备注 |
|------|------|----------|------|
| NOS3 | 恢复NO合成 | L-精氨酸、BH4 | 补充底物 |
| SOD1/CAT | 抗氧化剂 | NAC、SOD模拟物 | 研究阶段 |
| 整体 | 抗氧化治疗 | 维生素C/E、辅酶Q10 | 辅助治疗 |

### 3. 抗炎治疗
| 靶点 | 策略 | 现有药物 | 备注 |
|------|------|----------|------|
| IL1B | IL-1阻断 | 阿那白滞素 | 昂贵 |
| TNF | TNF-α抑制剂 | 依那西普、英夫利昔 | 自身免疫 |
| NFKB | NF-κB抑制剂 | 姜黄素、白藜芦醇 | 天然产物 |

### 4. 钙通道调节
| 靶点 | 策略 | 现有药物 | 备注 |
|------|------|----------|------|
| 钙通道 | CCB降压药 | 氨氯地平、硝苯地平 | 常用降压药 |

## 二、小分子化合物设计线索

### 基于金属结合位点
1. **ACE Zn2+位点**: 设计金属螯合剂
2. **NOS BH4位点**: 恢复四氢生物蝶呤
3. **SOD/CAT金属位点**: 金属替代疗法

### 基于结构优化
1. **现有ARB类**: 优化与AGTR1的结合
2. **天然产物**: 姜黄素、白藜芦醇结构改造

## 三、VCell模拟建议

### 通路1: RAS系统
```
铅 → ACE激活 → Ang II → AGTR1 → 血管收缩 → 血压升高
```

### 通路2: 氧化应激
```
铅 → ROS增加 → NOS失活 → NO减少 → 血管舒张障碍 → 血压升高
```

### 通路3: 炎症
```
铅 → NF-κB激活 → IL-1β/TNF → 炎症 → 内皮功能障碍 → 血压升高
```

---
*分析日期: 2026-02-20*
"""
    
    print(strategies)
    
    # 保存
    with open(f"{OUTPUT_DIR}/intervention_strategy.md", "w", encoding="utf-8") as f:
        f.write(strategies)
    
    return strategies

def main():
    print("="*60)
    print("🔬 铅与血压/CKM关键靶点深度分析")
    print("="*60)
    
    # 1. 分析关键靶点
    df_targets = analyze_key_targets()
    
    # 2. 搜索PDB结构
    pdb_results = search_pdb_structures()
    
    # 3. 生成干预建议
    strategies = generate_intervention_summary(df_targets)
    
    print("\n" + "="*60)
    print("✅ 分析完成!")
    print("="*60)
    print(f"输出文件:")
    print(f"  - {OUTPUT_DIR}/lead_bp_key_targets.csv")
    print(f"  - {OUTPUT_DIR}/pdb_structures.csv") 
    print(f"  - {OUTPUT_DIR}/intervention_strategy.md")

if __name__ == "__main__":
    main()
