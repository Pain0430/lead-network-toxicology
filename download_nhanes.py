#!/usr/bin/env python3
"""
NHANES 2021-2023 数据下载工具
Download NHANES 2021-2023 Data for Lead/Heavy Metals Analysis

NHANES 2021-2023 周期数据下载
"""

import os
import requests
from urllib.parse import urljoin

# 配置
OUTPUT_DIR = "nhanes_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# NHANES 2021-2023 数据文件列表
# 注意: 2021-2023周期的数据在文件中标记为 "2021"，后缀为 "_L"
NHANES_FILES = {
    # 实验室数据 - 重金属 (最重要!)
    "laboratory": {
        "PBCD_L - Blood Lead, Cadmium, Mercury, Selenium, Manganese": {
            "url": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/PBCD_L.xpt",
            "description": "血铅、血镉、血汞、硒、锰 - 核心数据!",
            "file": "PBCD_L.xpt"
        },
        "IHGEM_L - Mercury (Inorganic, Ethyl, Methyl)": {
            "url": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/IHGEM_L.xpt",
            "description": "血汞形态分析",
            "file": "IHGEM_L.xpt"
        },
    },
    # 实验室数据 - 生化指标
    "laboratory_biochemistry": {
        "CBC_L - Complete Blood Count": {
            "url": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/CBC_L.xpt",
            "description": "血常规 (1.5 MB)",
            "file": "CBC_L.xpt"
        },
        "GHB_L - Glycohemoglobin": {
            "url": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/GHB_L.xpt",
            "description": "糖化血红蛋白",
            "file": "GHB_L.xpt"
        },
        "HDL_L - High-Density Lipoprotein": {
            "url": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/HDL_L.xpt",
            "description": "高密度脂蛋白",
            "file": "HDL_L.xpt"
        },
        "TRIGLY_L - LDL & Triglycerides": {
            "url": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/TRIGLY_L.xpt",
            "description": "低密度脂蛋白和甘油三酯",
            "file": "TRIGLY_L.xpt"
        },
    },
    # 体检数据
    "examination": {
        "BPX_L - Blood Pressure": {
            "url": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/BPX_L.xpt",
            "description": "血压检查数据",
            "file": "BPX_L.xpt"
        },
    },
    # 问卷数据
    "questionnaire": {
        "DEMO_L - Demographics": {
            "url": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DEMO_L.xpt",
            "description": "人口统计学数据 (2.5 MB)",
            "file": "DEMO_L.xpt"
        },
        "MCQ_L - Medical Conditions": {
            "url": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/MCQ_L.xpt",
            "description": "健康状况问卷",
            "file": "MCQ_L.xpt"
        },
    }
}

def download_file(url, filepath):
    """下载单个文件"""
    print(f"📥 下载: {url}")
    print(f"   -> {filepath}")
    
    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        size = os.path.getsize(filepath) / 1024  # KB
        print(f"   ✅ 完成 ({size:.1f} KB)")
        return True
        
    except Exception as e:
        print(f"   ❌ 失败: {e}")
        return False

def main():
    print("=" * 60)
    print("📊 NHANES 2021-2023 数据下载工具")
    print("=" * 60)
    
    success_count = 0
    fail_count = 0
    
    # 遍历所有数据类别
    for category, datasets in NHANES_FILES.items():
        print(f"\n📂 {category.upper()}")
        print("-" * 40)
        
        for name, info in datasets.items():
            filepath = os.path.join(OUTPUT_DIR, info["file"])
            
            # 检查文件是否已存在
            if os.path.exists(filepath):
                size = os.path.getsize(filepath) / 1024
                print(f"⏭️  跳过 (已存在): {info['file']} ({size:.1f} KB)")
                success_count += 1
                continue
            
            if download_file(info["url"], filepath):
                success_count += 1
            else:
                fail_count += 1
    
    print("\n" + "=" * 60)
    print(f"✅ 下载完成!")
    print(f"   成功: {success_count}")
    print(f"   失败: {fail_count}")
    print(f"   保存位置: {OUTPUT_DIR}/")
    print("=" * 60)
    
    # 列出下载的文件
    print("\n📁 已下载的文件:")
    for f in os.listdir(OUTPUT_DIR):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024
        print(f"   - {f} ({size:.1f} KB)")

if __name__ == "__main__":
    main()
