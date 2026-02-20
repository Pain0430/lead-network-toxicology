# 铅诱导高血压 - VCell通路模型
# Lead-induced Hypertension - VCell Pathway Model
# 
# 细胞类型: 血管内皮细胞 (Vascular Endothelial Cells)
# 模拟时间: 0-24小时
# 
# VCell模型可在 https://vcell.org/ 导入使用

# 模型说明:
# 本模型模拟铅暴露后血管内皮细胞中的关键信号通路

## Species (物种/分子)

### 外部刺激
| 物种 | 初始浓度 | 单位 | 描述 |
|------|---------|------|------|
| Lead | 0 | μM | 细胞外铅浓度 |

### 氧化应激通路
| 物种 | 初始浓度 | 单位 | 描述 |
|------|---------|------|------|
| ROS | 1 | a.u. | 活性氧水平 |
| SOD | 100 | a.u. | 超氧化物歧化酶活性 |
| CAT | 100 | a.u. | 过氧化氢酶活性 |
| NOS3 | 100 | a.u. | eNOS活性 |
| NO | 10 | a.u. | 一氧化氮 |

### RAS系统
| 物种 | 初始浓度 | 单位 | 描述 |
|------|---------|------|------|
| ACE | 50 | a.u. | 血管紧张素转换酶 |
| AngI | 10 | a.u. | 血管紧张素I |
| AngII | 1 | a.u. | 血管紧张素II |
| AT1R | 50 | a.u. | AT1受体激活状态 |
| VascularTone | 10 | a.u. | 血管张力 |

### 炎症通路
| 物种 | 初始浓度 | 单位 | 描述 |
|------|---------|------|------|
| NFkB | 10 | a.u. | NF-κB激活 |
| IL1b | 1 | a.u. | 白介素1β |
| TNFa | 1 | a.u. | 肿瘤坏死因子α |
| EndothelialFunc | 100 | a.u. | 内皮功能 |

## Reactions (反应)

### 氧化应激反应
| 反应 | 速率方程 | 参数 |
|------|---------|------|
| Lead → ROS | k_lead_ros * Lead | k_lead_ros = 0.1 |
| ROS + SOD → inactive | k_ros_sod * ROS * SOD | k_ros_sod = 0.01 |
| ROS + CAT → inactive | k_ros_cat * ROS * CAT | k_ros_cat = 0.01 |
| NOS3 + ROS → inactive | k_nos_ros * NOS3 * ROS | k_nos_ros = 0.05 |
| NOS3 → NO | k_nos * NOS3 | k_nos = 0.1 |

### RAS系统反应
| 反应 | 速率方程 | 参数 |
|------|---------|------|
| Lead + ACE → ACE_active | k_lead_ace * Lead * ACE | k_lead_ace = 0.05 |
| ACE_active + AngI → AngII | k_ace * ACE_active * AngI | k_ace = 0.1 |
| AngII + AT1R → AT1R_active | k_ang_at1r * AngII * AT1R | k_ang_at1r = 0.1 |
| AT1R_active → VascularTone | k_at1r_tone * AT1R_active | k_at1r_tone = 0.1 |

### 炎症反应
| 反应 | 速率方程 | 参数 |
|------|---------|------|
| Lead + NFkB → NFkB_active | k_lead_nfk * Lead * NFkB | k_lead_nfk = 0.03 |
| NFkB_active → IL1b | k_nfk_il1 * NFkB_active | k_nfk_il1 = 0.1 |
| NFkB_active → TNFa | k_nfk_tnf * NFkB_active | k_nfk_tnf = 0.1 |
| IL1b + EndothelialFunc → dysfunction | k_il1_endo * IL1b * EndothelialFunc | k_il1_endo = 0.01 |

### 血压相关综合反应
| 反应 | 速率方程 | 参数 |
|------|---------|------|
| BloodPressure = f(VascularTone, NO, EndothelialFunc) | (VascularTone * 10) / (NO + 1) | - |

## 参数

| 参数 | 值 | 单位 | 描述 |
|------|-----|------|------|
| k_lead_ros | 0.1 | /h | 铅诱导ROS产生 |
| k_ros_sod | 0.01 | /h | ROS与SOD反应 |
| k_ros_cat | 0.01 | /h | ROS与CAT反应 |
| k_nos_ros | 0.05 | /h | ROS抑制NOS |
| k_nos | 0.1 | /h | NOS产生NO |
| k_lead_ace | 0.05 | /h | 铅激活ACE |
| k_ace | 0.1 | /h | ACE转化AngI→AngII |
| k_ang_at1r | 0.1 | /h | AngII激活AT1R |
| k_at1r_tone | 0.1 | /h | AT1R增加血管张力 |
| k_lead_nfk | 0.03 | /h | 铅激活NF-κB |
| k_nfk_il1 | 0.1 | /h | NF-κB产生IL-1β |
| k_nfk_tnf | 0.1 | /h | NF-κB产生TNF-α |
| k_il1_endo | 0.01 | /h | IL-1β损伤内皮 |

## 模拟设置

- **细胞类型**: 血管内皮细胞
- **铅浓度梯度**: 0, 1, 5, 10 μM
- **模拟时间**: 0-24小时
- **时间步长**: 0.1小时
- **输出**: 血压随时间变化曲线

## 使用说明

1. 访问 https://vcell.org/
2. 创建新模型 (File → New BioModel)
3. 导入本VCML文件或在VCell界面手动创建
4. 设置铅浓度参数
5. 运行模拟
6. 分析血压(VascularTone)和NO变化

## 预期结果

当铅浓度升高时:
1. ROS产生增加
2. NOS活性下降 → NO减少
3. ACE激活 → AngII增加
4. 血管张力增加
5. 血压升高

这与NHANES数据分析发现的"收缩压中介效应88.6%"一致。
