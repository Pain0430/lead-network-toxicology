# Lead Network Toxicology - 铅网络毒理学分析

## 项目概述

本项目采用网络毒理学方法，系统分析铅暴露与心血管-肾脏-代谢(CKM)综合征之间的分子关联。

## 分析模块

### 1. 剂量-反应分析 (`dose_response_analysis.py`)
- 血铅剂量-反应曲线
- 阈值效应分析
- 限制性立方样条建模

### 2. 发表偏倚分析 (`publication_bias_analysis.py`)
- 漏斗图
- Egger's检验
- Trim-and-fill分析

### 3. 生物标志物网络分析 (`biomarker_network_analysis.py`)
- 相关性网络构建
- 社区检测算法
- 中心性分析

### 4. 通路富集分析 (`pathway_enrichment_analysis.py`)
- KEGG通路富集
- 通路相互作用网络
- 毒性机制热力图

### 5. 中介效应分析 (`mediation_analysis.py`)
- Baron-Kenny方法
- Bootstrap置信区间
- 多中介效应比较

### 6. 可视化增强 (`enhanced_visualization.py`)
- 增强版ROC/PR曲线
- 决策曲线分析(DCA)
- 校准曲线

### 7. 交互式可视化
- 交互式森林图
- 交互式列线图
- 交互式网络图

## 生成文件

### 核心分析结果
- `output/comprehensive_report.txt` - 综合分析报告
- `output/dose_response_report.txt` - 剂量-反应分析报告
- `output/publication_bias_report.txt` - 发表偏倚分析报告
- `output/network_analysis_report.txt` - 网络分析报告
- `output/pathway_analysis_report.txt` - 通路分析报告
- `output/mediation_report.txt` - 中介效应报告

### 可视化图表
详见 `output/` 目录

## 使用方法

```bash
# 运行所有分析
python enhanced_visualization.py
python dose_response_analysis.py
python publication_bias_analysis.py
python biomarker_network_analysis.py
python pathway_enrichment_analysis.py
python mediation_analysis.py

# 生成综合报告
python generate_report.py
```

## 主要发现

1. **血铅阈值**: 6.4 μg/dL 是CKM综合征的临界点
2. **核心风险因素**: 职业暴露(OR=2.57), 血铅(OR=2.17)
3. **关键通路**: 氧化应激 > NF-κB炎症 > 肠-肝轴
4. **预测性能**: AUC = 0.944

## 作者

**Pain** - 重庆医科大学 公共卫生学院 副教授

---
*生成日期: 2026-02-27*
