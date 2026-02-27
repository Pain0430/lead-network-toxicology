# 因果推断分析模块规范 (Causal Inference Module)

## 模块名称
`causal_inference.py` - 因果推断分析模块

## 目标
为铅暴露与CKM综合征研究提供严格的因果推断方法，包括DAG构建、倾向评分匹配、逆概率加权等

## 核心功能

### 1. DAG有向无环图构建
- 基于领域知识构建因果有向无环图
- 使用DAGitty格式保存和可视化
- 识别最小调整集

### 2. 倾向评分方法
- **倾向评分匹配 (PSM)**: 1:n最近邻匹配
- **逆概率加权 (IPTW)**: 稳定权重计算
- **双重稳健估计 (AIPW)**: 结合倾向评分和结果回归

### 3. 协变量平衡评估
- 标准化均值差 (SMD)
- 方差比 (VR)
- 平衡可视化

### 4. 因果效应处理效应 (ATE估计
- 平均)
- 处理组平均效应 (ATT)
- 95%置信区间 (Bootstrap)

### 5. 敏感性分析
- E-value计算 (未测量混杂)
- 残差 confounding 评估

## 输入
- NHANES数据集 (Blood Lead, CKM相关指标)
- 协变量列表

## 输出
- `output/causal_dag.png` - DAG可视化
- `output/psm_balance.png` - 匹配前后平衡对比
- `output/iptw_weights.png` - IPTW权重分布
- `output/causal_effects.csv` - 因果效应结果
- `output/sensitivity_analysis.csv` - 敏感性分析结果

## 技术栈
- pandas, numpy
- dowhy (因果推断框架)
- econml ( doubly robust estimation)
- matplotlib, seaborn

## 预期效果
- 提供因果效应估计而非相关性
- 增强研究结果的科学可信度
- 支持Q1期刊投稿要求
