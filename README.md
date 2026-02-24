# Lead Network Toxicology - Visualization Enhancement

## 项目简介

铅网络毒理学可视化增强模块，提供高质量的ROC曲线、PR曲线、决策曲线分析(DCA)、校准曲线等可视化功能。

## 作者

- **Pain** (重庆医科大学 公共卫生学院 副教授)

## 功能特性

### 静态可视化 (Matplotlib/Seaborn)
- ✅ 增强版ROC曲线（带最优点标注）
- ✅ 增强版PR曲线
- ✅ 决策曲线分析 (DCA)
- ✅ 校准曲线
- ✅ 混淆矩阵热力图
- ✅ 特征重要性对比图
- ✅ 模型性能雷达图
- ✅ 风险分层图

### 交互式可视化 (Plotly)
- ✅ 交互式ROC曲线
- ✅ 交互式PR曲线
- ✅ 交互式决策曲线
- ✅ 交互式特征重要性
- ✅ 交互式风险热力图
- ✅ 交互式校准曲线

## 依赖

```
numpy
pandas
matplotlib
seaborn
scikit-learn
plotly
kaleido
```

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

### 静态可视化
```bash
python enhanced_visualization.py
```

### 交互式可视化
```bash
python interactive_visualization.py
```

## 输出文件

### 静态图表 (output/)
- `enhanced_roc_curves.png/pdf` - ROC曲线
- `enhanced_pr_curves.png/pdf` - PR曲线
- `decision_curve_analysis.png/pdf` - DCA曲线
- `calibration_curves.png/pdf` - 校准曲线
- `confusion_matrices.png` - 混淆矩阵
- `feature_importance_comparison.png` - 特征重要性
- `model_performance_radar.png` - 性能雷达图
- `risk_stratification.png` - 风险分层

### 交互式图表 (output/)
- `interactive_roc.html` - 交互式ROC
- `interactive_pr.html` - 交互式PR
- `interactive_dca.html` - 交互式DCA
- `interactive_feature_importance.html` - 交互式特征重要性
- `interactive_risk_heatmap.html` - 交互式风险热力图
- `interactive_calibration.html` - 交互式校准曲线

## 数据说明

- 样本量: 2000
- 特征数: 28
- 目标: 铅毒性风险预测 (二分类)
- 特征类别:
  - 人口学: Age, Sex, BMI
  - 铅暴露: Blood_Lead, Urine_Lead, Hair_Lead
  - 氧化应激: SOD, GSH, MDA, 8-OHdG
  - 炎症: CRP, IL6, TNF-α
  - 肝肾功能: ALT, AST, Creatinine, BUN
  - 心血管: BP, HbA1c, Cholesterol
  - 胆汁酸: DCA, LCA, CA, UDCA
  - 肠道屏障: Calprotectin, Zonulin, LBP

## 模型性能

| 模型 | ROC-AUC | PR-AUC | F1 | Brier Score |
|------|---------|--------|-----|-------------|
| Logistic Regression | 0.944 | 0.032 | 0.069 | 0.045 |
| Random Forest | 0.911 | 0.023 | 0.000 | 0.004 |
| Gradient Boosting | 0.504 | 0.021 | 0.000 | 0.006 |

## 关键发现

1. **血铅水平**是最重要的预测因子
2. 氧化应激指标(MDA)与铅毒性密切相关
3. 肠道屏障标志物(钙卫蛋白)对预测有重要贡献
4. 模型在广泛阈值范围内具有正的净收益(决策曲线分析)

## 许可证

仅供研究使用

---
更新日期: 2026-02-25
