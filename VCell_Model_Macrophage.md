# 铅诱导炎症 - VCell通路模型 (单核巨噬细胞)
# Lead-induced Inflammation - VCell Pathway Model (Monocyte/Macrophage)
# 
# 细胞类型: 单核细胞/巨噬细胞 (Monocyte/Macrophage)
# 模拟时间: 0-24小时
#
# 用途: 模拟铅暴露后巨噬细胞的炎症反应

## Species (物种)

### 外部刺激
| 物种 | 初始浓度 | 单位 | 描述 |
|------|---------|------|------|
| Lead | 0 | μM | 细胞外铅浓度 |

### 氧化应激
| 物种 | 初始浓度 | 单位 | 描述 |
|------|---------|------|------|
| ROS | 1 | a.u. | 活性氧 |
| SOD | 100 | a.u. | 超氧化物歧化酶 |
| GSH | 100 | a.u. | 谷胱甘肽 |

### 炎症信号
| 物种 | 初始浓度 | 单位 | 描述 |
|------|---------|------|------|
| NFkB | 10 | a.u. | NF-κB (细胞质，非激活) |
| NFkB_nuc | 1 | a.u. | NF-κB (细胞核，激活) |
| IKK | 10 | a.u. | IκB激酶 |
| IkB | 50 | a.u. | IκB抑制剂 |
| IL1b | 1 | a.u. | 白介素-1β |
| IL6 | 1 | a.u. | 白介素-6 |
| TNFa | 1 | a.u. | 肿瘤坏死因子-α |

### 细胞状态
| 物种 | 初始浓度 | 单位 | 描述 |
|------|---------|------|------|
| Macrophage | 100 | a.u. | 巨噬细胞活化状态 |
| Apoptosis | 1 | a.u. | 凋亡水平 |

## Reactions (反应)

### 铅诱导ROS
| 反应 | 速率方程 | 参数 |
|------|---------|------|
| Lead → ROS | k1 * Lead | k1 = 0.1 |
| ROS + GSH → GSSG | k2 * ROS * GSH | k2 = 0.05 |
| ROS + SOD → | k3 * ROS * SOD | k3 = 0.01 |

### NF-κB炎症通路
| 反应 | 速率方程 | 参数 |
|------|---------|------|
| Lead + IKK → IKK_active | k4 * Lead * IKK | k4 = 0.05 |
| IKK_active + IkB → pIkB | k5 * IKK_active * IkB | k5 = 0.1 |
| pIkB → degraded | k6 * pIkB | k6 = 0.1 |
| NFkB + IkB → Complex | k7 * NFkB * IkB | k7 = 0.1 |
| Complex → degraded | k8 * Complex | k8 = 0.05 |
| NFkB → NFkB_nuc | k9 * NFkB | k9 = 0.1 |
| NFkB_nuc → IL1b | k10 * NFkB_nuc | k10 = 0.1 |
| NFkB_nuc → IL6 | k11 * NFkB_nuc | k11 = 0.1 |
| NFkB_nuc → TNFa | k12 * NFkB_nuc | k12 = 0.1 |

### 炎症反馈与细胞损伤
| 反应 | 速率方程 | 参数 |
|------|---------|------|
| IL1b + Macrophage → activated | k13 * IL1b * Macrophage | k13 = 0.05 |
| TNFa + Macrophage → activated | k14 * TNFa * Macrophage | k14 = 0.05 |
| ROS + Macrophage → damage | k15 * ROS * Macrophage | k15 = 0.02 |
| damage → Apoptosis | k16 * damage | k16 = 0.01 |

## 参数表

| 参数 | 值 | 描述 |
|------|-----|------|
| k1 | 0.1 | 铅诱导ROS产生 |
| k2 | 0.05 | ROS消耗GSH |
| k3 | 0.01 | ROS与SOD反应 |
| k4 | 0.05 | 铅激活IKK |
| k5 | 0.1 | IKK磷酸化IκB |
| k6 | 0.1 | 磷酸化IκB降解 |
| k7 | 0.1 | IκB结合NF-κB |
| k8 | 0.05 | 复合物降解 |
| k9 | 0.1 | NF-κB核转位 |
| k10 | 0.1 | NF-κB激活IL-1β |
| k11 | 0.1 | NF-κB激活IL-6 |
| k12 | 0.1 | NF-κB激活TNF-α |
| k13 | 0.05 | IL-1β激活巨噬细胞 |
| k14 | 0.05 | TNF-α激活巨噬细胞 |
| k15 | 0.02 | ROS损伤巨噬细胞 |
| k16 | 0.01 | 损伤导致凋亡 |

## 模拟设置

- **细胞类型**: 单核细胞/巨噬细胞
- **铅浓度梯度**: 0, 1, 5, 10 μM
- **模拟时间**: 0-24小时
- **输出**: 炎症因子(IL-1β, IL-6, TNF-α)随时间变化

## 预期结果

铅浓度升高时:
1. ROS产生增加
2. GSH耗竭
3. IKK激活 → IκB降解
4. NF-κB核转位
5. 炎症因子(IL-1β, IL-6, TNF-α)分泌增加
6. 巨噬细胞持续激活
7. 可能诱导凋亡

这与铅诱导的系统性炎症和CKM综合征进展相关。
