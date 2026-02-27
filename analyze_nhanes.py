#!/usr/bin/env python3
"""
NHANES 数据读取和分析
分析 PBCD_L (血铅、血镉、血汞等) 数据
"""

import pandas as pd
import os

DATA_DIR = "nhanes_data"

def load_nhanes_data():
    """加载NHANES数据"""
    
    # 加载血重金属数据 (最重要!)
    print("📂 加载血重金属数据 (PBCD_L)...")
    pbbcd = pd.read_sas(f"{DATA_DIR}/PBCD_L.xpt")
    print(f"   样本数: {len(pbbcd)}, 变量数: {len(pbbcd.columns)}")
    print(f"   列名: {list(pbbcd.columns)}")
    
    # 加载人口统计数据
    print("\n📂 加载人口统计数据 (DEMO_L)...")
    demo = pd.read_sas(f"{DATA_DIR}/DEMO_L.xpt")
    print(f"   样本数: {len(demo)}, 变量数: {len(demo.columns)}")
    
    # 加载健康问卷
    print("\n📂 加载健康问卷 (MCQ_L)...")
    mcq = pd.read_sas(f"{DATA_DIR}/MCQ_L.xpt")
    print(f"   样本数: {len(mcq)}, 变量数: {len(mcq.columns)}")
    
    return pbbcd, demo, mcq

def analyze_lead_data(pbbcd, demo):
    """分析血铅数据"""
    print("\n" + "="*60)
    print("🔬 血铅数据分析")
    print("="*60)
    
    # 查找铅相关列
    lead_cols = [c for c in pbbcd.columns if 'LBX' in c.upper() or 'LPB' in c.upper()]
    print(f"\n铅/重金属相关列: {lead_cols}")
    
    # 显示数据描述
    print("\n数据统计:")
    print(pbbcd[lead_cols].describe())
    
    return pbbcd

def main():
    print("="*60)
    print("📊 NHANES 2021-2023 数据探索")
    print("="*60)
    
    pbbcd, demo, mcq = load_nhanes_data()
    analyze_lead_data(pbbcd, demo)
    
    # 保存血铅数据为CSV
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    pbbcd.to_csv(f"{output_dir}/nhanes_lead_blood.csv", index=False)
    print(f"\n✅ 血铅数据已保存到: {output_dir}/nhanes_lead_blood.csv")

if __name__ == "__main__":
    main()
