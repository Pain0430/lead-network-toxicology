# VCell 虚拟细胞建模教程

VCell (Virtual Cell) 是一个免费的细胞建模平台，用于模拟细胞内复杂的生物过程。

---

## 访问 VCell

**官网**: https://vcell.org/

**快速入门指南**: https://vcell.org/webstart/VCell_Tutorials/VCell_Quickstart_7_Biomodel.pdf

---

## VCell 建模流程

### 1. 注册与登录

1. 访问 https://vcell.org/
2. 点击 "Login" 或 "Register"
3. 使用邮箱注册账号（免费）

### 2. 创建新模型

#### 方式A: 从头创建
```
File → New BioModel
```

#### 方式B: 基于现有模型修改
```
File → Open → Shared Models → 搜索 "oxidative stress" 或 "cell death"
```

### 3. 模型构建步骤

#### Step 1: 定义生物网络 (Molecular Species)
- 点击 "Species" 添加分子
- 例如: Lead, ROS, SOD, CAT, GPx
- 设置初始浓度

#### Step 2: 定义反应 (Reactions)
- 点击 "Reactions" 添加反应
- 格式: A + B → C (反应速率)

#### Step 3: 设置参数 (Parameters)
- 点击 "Parameters" 
- 设置反应速率常数 (k)
- 可从文献获取或估计

#### Step 4: 定义初始条件 (Initial Conditions)
- 设置各分子的初始值

#### Step 5: 运行模拟 (Simulation)

1. 点击 "New Simulation"
2. 设置时间范围 (如 0-24 小时)
3. 点击 "Run"

### 4. 分析结果

- **时间曲线**: 查看各分子浓度随时间变化
- **相图**: 查看两个变量之间的关系
- **敏感性分析**: 识别关键参数

---

## 铅毒性建模示例

### 核心通路

```
铅(Pb) 
  ↓
ROS升高 (活性氧)
  ↓
抗氧化系统受损 (SOD, CAT, GPx活性下降)
  ↓
氧化应激 → 细胞损伤
  ↓
线粒体功能障碍 → 凋亡
```

### VCell 模型组件

| 组件 | 描述 |
|------|------|
| Species | Pb, ROS, SOD, CAT, GPx, Mitochondria, Apoptosis |
| Reactions | Pb → ROS, ROS + SOD → ... |
| Parameters | k_ROS_prod, k_SOD, k_CAT, k_apoptosis |

### 铅浓度设置参考

| 血铅水平 (μg/dL) | 风险级别 |
|-----------------|---------|
| < 5 | 安全 |
| 5-10 | 边缘 |
| 10-20 | 轻度升高 |
| 20-45 | 中度升高 |
| > 45 | 严重 |

---

## 替代方案: 本地建模

如果不想用 VCell web 界面，可以使用本地工具：

### COPASI (免费开源)
- 官网: https://copasi.org/
- 支持: Windows, Mac, Linux
- 功能: 稳态分析、敏感性分析、时间course模拟

### Python + tellurium
```python
# Python 细胞建模
import tellurium as te

model = """
model oxidative_stress()
    // 物种
    Pb = 10;  // 铅浓度
    ROS = 1;  // 活性氧
    SOD = 100;  // 超氧化物歧化酶
    CAT = 100;  // 过氧化氢酶
    
    // 反应
    Pb -> ROS; k1 * Pb;
    ROS + SOD -> ; k2 * ROS * SOD;
    ROS + CAT -> ; k3 * ROS * CAT;
    
    // 参数
    k1 = 0.1;
    k2 = 0.01;
    k3 = 0.01;
end
"""

rr = te.loadAntimonyString(model)
result = rr.simulate(0, 24, 100)
rr.plot(result)
```

---

## 教程资源

1. **VCell 官方教程**: https://vcell.org/webstart/VCell_Tutorials/
2. **YouTube 教程**: https://www.youtube.com/@compcellbiol8898
3. **VCell 论文**: https://pmc.ncbi.nlm.nih.gov/articles/PMC4119324/

---

## 本项目中的应用

### 网络毒理学 → VCell 验证

1. **网络毒理学输出**: 关键靶点基因和通路
2. **VCell 建模**: 
   - 将关键通路转化为数学模型
   - 输入铅浓度
   - 模拟 ROS、氧化应激标志物的动态变化
3. **验证**: 与NHANES人群数据对比

---

*更新: 2026-02-20*
