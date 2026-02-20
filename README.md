# 🔬 铅 (Lead) 环境毒理学研究 - 创新方向
## Network Toxicology + CKM Syndrome + NHANES

**创新方向**: 铅与CKM (Cardiovascular-Kidney-Metabolic) 综合征的关联研究

---

## 📊 研究创新点

### 背景突破
- **传统研究**: 儿童铅中毒 → 神经发育；成人铅中毒 → 神经退行性疾病
- **本研究创新**: 聚焦**代谢性疾病**和**CKM综合征**

### CKM综合征 (2024年AHA新概念)
- **C**ardiovascular: 心血管疾病
- **K**idney: 慢性肾脏病
- **M**etabolic: 代谢综合征(肥胖、糖尿病)

### 综合指标创新
1. **CKM风险评分**: 整合高血压、糖尿病、心脏病、肾病、代谢综合征
2. **TyG指数**: 甘油三酯-葡萄糖指数 (胰岛素抵抗指标)
3. **中介效应分析**: 铅 → 代谢指标 → CKM

---

## 📊 初步分析结果 (NHANES 2021-2023)

### 样本量: 7,586人

### 血铅分布
| 指标 | 值 |
|------|-----|
| 均值 | 0.87 μg/dL |
| 中位数 | 0.64 μg/dL |
| P95 | 2.14 μg/dL |
| P99 | 4.25 μg/dL |

### 铅与CKM指标相关性 (Spearman)

| CKM指标 | r值 | p值 | 显著性 |
|---------|-----|-----|--------|
| CKM综合风险评分 | **0.183** | <0.001 | *** |
| 糖化血红蛋白 | **0.205** | <0.001 | *** |
| 代谢综合征评分 | 0.094 | <0.001 | *** |
| 甘油三酯 | 0.080 | <0.001 | *** |

### 回归分析
- β = 0.0801, p < 0.001
- 血铅每升高1 μg/dL，CKM风险评分增加0.08分

---

## 📁 项目文件

```
lead-network-toxicology/
├── README.md                        # 项目说明
├── lead_network_toxicology.py       # 网络毒理学分析
├── lead_ckm_analysis.py             # CKM综合征分析 (创新!)
├── download_nhanes.py               # 数据下载
├── VCELL_TUTORIAL.md               # VCell教程
├── nhanes_data/                    # NHANES原始数据
└── output/                         # 分析结果
```

---

## 🔬 下一步计划

1. **完善数据**: 获取血压、腰围数据计算更完整的CKM指标
2. **中介效应**: 构建结构方程模型 (铅→TyG→CKM)
3. **网络毒理学整合**: 将CKM相关靶点与网络预测对比
4. **VCell模拟**: 验证铅对代谢通路的动态影响

---

## 📚 参考文献

1. CKM Syndrome - AHA Presidential Advisory (2024)
2. Heavy metals and CKM syndrome - Frontiers in Nutrition (2025)
3. Network toxicology and its application - JEOM (2025)

---

*更新: 2026-02-20*
