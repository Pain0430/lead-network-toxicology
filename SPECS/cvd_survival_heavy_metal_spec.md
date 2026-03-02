# CVD Survival + Heavy Metal ML Prediction Specification

## Project Overview
- **名称**: CVD Survival Heavy Metal Prediction
- **类型**: Machine Learning Survival Analysis
- **核心功能**: 基于血清和尿液重金属水平预测心血管疾病患者生存率
- **目标**: 结合最新Frontiers in Public Health (2025)方法，提升CKM研究

## Data Sources
- NHANES 数据 (2003-2018)
- 重金属: Pb, As, Cd, Hg, Ba (血清/尿液)
- 心血管疾病: CHD, Stroke, Heart Failure

## Methods
1. 多模型对比: Random Forest, XGBoost, LightGBM, Neural Network
2. Cox比例风险模型
3. SHAP可解释性分析
4. 生存曲线 (Kaplan-Meier)
5. 校准曲线

## Features
- 特征重要性排序
- SHAP Summary Plot
- 风险分层分析
- 时间依赖ROC

## Output
- cvd_survival_ml.py: 主分析脚本
- output/cvd_*: 可视化结果

## Reference
- Jin H et al. (2025) Frontiers in Public Health. DOI: 10.3389/fpubh.2025.1582779
