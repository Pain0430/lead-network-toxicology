# 🔬 铅 (Lead) 环境毒理学研究
## Network Toxicology + Virtual Cell + NHANES

基于网络毒理学、虚拟细胞模拟和NHANES真实世界数据的铅毒性机制研究。

---

## 📊 项目概述

```
污染物选取 → 网络毒理学预测 → 虚拟细胞模拟 → NHANES人群验证 → AOP构建
```

### 研究流程

1. **网络毒理学分析** - 预测铅的毒性靶点和通路
2. **VCell虚拟细胞模拟** - 动态验证剂量-效应关系
3. **NHANES数据分析** - 人群暴露-健康关联分析

---

## 📁 文件结构

```
lead-network-toxicology/
├── README.md                    # 本文件
├── lead_network_toxicology.py  # 网络毒理学分析主程序
├── download_nhanes.py          # NHANES数据下载工具
├── analyze_nhanes.py           # NHANES数据分析
├── VCELL_TUTORIAL.md           # VCell使用教程
├── output/                     # 分析结果
│   ├── lead_target_genes.txt   # 靶点基因列表
│   ├── lead_pathways.csv       # 通路富集结果
│   ├── lead_network_toxicology.html  # 交互式网络图
│   └── nhanes_lead_blood.csv  # NHANES血铅数据
├── nhanes_data/               # NHANES原始数据
│   ├── PBCD_L.xpt             # 血重金属 (铅、镉、汞、硒、锰)
│   ├── DEMO_L.xpt             # 人口统计数据
│   ├── MCQ_L.xpt              # 健康问卷
│   └── ...
└── requirements.txt            # Python依赖
```

---

## 🔬 网络毒理学结果

### 靶点基因: 96个

### KEGG通路富集 (Top 10)

| 通路 | 重叠基因 | p-value |
|------|---------|---------|
| 氧化应激反应 | 9/9 | 1e-15 |
| 血红素合成 | 4/4 | 1e-14 |
| 金属转运 | 5/5 | 1e-13 |
| 炎症反应 | 6/6 | 1e-12 |
| 细胞凋亡 | 6/6 | 1e-11 |
| 神经毒性 | 7/7 | 1e-10 |
| DNA损伤修复 | 6/6 | 1e-09 |
| 肾毒性 | 6/6 | 1e-08 |
| MAPK信号 | 4/6 | 1e-08 |
| 心血管疾病 | 6/6 | 1e-07 |

---

## 📊 NHANES 数据

### 已下载数据 (2021-2023)

| 文件 | 描述 | 大小 |
|------|------|------|
| PBCD_L.xpt | 血铅、血镉、血汞、硒、锰 | 1.1 MB |
| DEMO_L.xpt | 人口统计学 | 2.5 MB |
| MCQ_L.xpt | 健康状况问卷 | 3.2 MB |
| CBC_L.xpt | 血常规 | 1.5 MB |
| HDL_L.xpt | 血脂 | 253 KB |
| TRIGLY_L.xpt | 甘油三酯 | 314 KB |
| GHB_L.xpt | 糖化血红蛋白 | 170 KB |
| IHGEM_L.xpt | 血汞形态 | 752 KB |

### 血铅统计 (NHANES 2021-2023)

| 指标 | 值 |
|------|-----|
| 样本数 | 7,586 |
| 均值 | 0.87 μg/dL |
| 中位数 | 0.64 μg/dL |
| 最小值 | 0.085 μg/dL |
| 最大值 | 48.07 μg/dL |

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install pandas numpy requests scipy
pip install pyreadstat  # 读取SAS格式
pip install xgboost shap  # 机器学习
```

### 2. 运行网络毒理学分析

```bash
python lead_network_toxicology.py
```

### 3. 下载NHANES数据

```bash
python download_nhanes.py
```

### 4. 分析NHANES数据

```bash
python analyze_nhanes.py
```

---

## 🔧 工具与数据库

### 网络毒理学

| 工具 | 用途 |
|------|------|
| CTD | 毒物基因组学数据库 |
| SwissTargetPrediction | 靶点预测 |
| STRING | 蛋白互作网络 |
| KEGG/Reactome | 通路富集 |
| Cytoscape | 网络可视化 |

### 虚拟细胞

| 工具 | 用途 |
|------|------|
| VCell | 虚拟细胞建模平台 |
| COPASI | 细胞通路模拟 |
| tellurium | Python细胞建模 |

### 数据分析

| 工具 | 用途 |
|------|------|
| Python/Pandas | 数据处理 |
| R/NHANES | 流行病学分析 |
| XGBoost/SHAP | 机器学习建模 |

---

## 📚 参考文献

1. 网络毒理学及其在外源性化学物毒性研究的应用概况. JEOM, 2025
2. The Virtual Cell Modeling and Simulation Software Environment. PMC, 2009
3. Construction of environmental risk score using ML: metal mixtures, oxidative stress and CVD in NHANES. Environmental Health, 2017
4. The Adverse Outcome Pathway: A Multifaceted Framework. PMC, 2018

---

## 📅 更新日志

- **2026-02-20**: 初始版本
  - 完成网络毒理学分析 (96个靶点, 10条通路)
  - 下载NHANES 2021-2023数据 (8个文件)
  - 创建VCell教程文档

---

## 📧 联系

如有问题，请提交 Issue 或联系作者。

---

*本项目用于科研目的，数据来源为公开的NHANES数据库*
