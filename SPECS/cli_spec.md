# SPEC: Lead Network Toxicology CLI

## 项目概述
为 lead-network-toxicology 项目创建一个统一的命令行界面(CLI)，方便用户运行各种分析模块。

## 功能需求

### 1. CLI 入口 (`cli.py`)
- 使用 argparse 构建命令行工具
- 支持子命令模式，每个分析模块作为子命令

### 2. 子命令列表

```
lead-tox分析 <子命令>

可用子命令:
  causal          运行因果推断分析
  dose-response   运行剂量-反应分析
  network         运行生物标志物网络分析
  ml              运行机器学习预测
  visualize       生成可视化图表
  report          生成综合报告
  all             运行完整分析流程
```

### 3. 通用参数
- `--input/-i`: 输入数据文件 (默认: data/nhanes_lead_data.csv)
- `--output/-o`: 输出目录 (默认: output/)
- `--config/-c`: 配置文件路径
- `--verbose/-v`: 输出详细信息

### 4. 特定参数
每个子命令可以有自己的特定参数

## 验收标准
1. `python cli.py --help` 显示帮助信息
2. `python cli.py causal --help` 显示因果推断模块帮助
3. `python cli.py all` 运行完整分析流程
4. 代码遵循现有项目风格
