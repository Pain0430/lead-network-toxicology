"""
中文配置模块 - 解决 matplotlib 中文显示问题
"""

import matplotlib.pyplot as plt
import matplotlib
import os

def setup_chinese_font():
    """自动检测并设置中文字体"""
    
    # 尝试的字体列表（按优先级）
    fonts = [
        # macOS
        'PingFang SC',        # 苹方
        'Hiragino Sans GB',  # 冬青黑体
        'STHeiti',            # 华文黑体
        'WenQuanYi Micro Hei', # 文泉驿
        # Windows
        'Microsoft YaHei',    # 微软雅黑
        'SimHei',             # 黑体
        # Linux
        'WenQuanYi Zen Hei', # 文泉驿正黑
        'Noto Sans CJK SC',  # Google Noto
        # 通用
        'DejaVu Sans',
        'Arial Unicode MS',
    ]
    
    # 检查可用字体
    from matplotlib import font_manager
    available_fonts = set([f.name for f in font_manager.fontManager.ttflist])
    
    # 找到第一个可用的中文字体
    chosen_font = None
    for font in fonts:
        if font in available_fonts:
            chosen_font = font
            break
    
    if chosen_font:
        plt.rcParams['font.sans-serif'] = [chosen_font] + plt.rcParams['font.sans-serif']
        print(f"✓ 使用中文字体: {chosen_font}")
    else:
        print("⚠ 未找到中文字体，尝试使用系统默认")
    
    # 设置负号显示
    plt.rcParams['axes.unicode_minus'] = False
    
    return chosen_font


def get_chinese_label(english, chinese):
    """获取中英文标签"""
    # 如果有中文字体，返回中文；否则返回英文
    from matplotlib import font_manager
    fonts = [f.name for f in font_manager.fontManager.ttflist]
    
    # 检查是否有中文字体
    chinese_fonts = ['PingFang', 'Hiragino', 'STHeiti', 'Microsoft', 'WenQuanYi', 'Noto']
    has_chinese = any(any(cf in f for cf in chinese_fonts) for f in fonts)
    
    return chinese if has_chinese else english


# 研究相关中英文术语对照
TERMS = {
    # 统计指标
    'AUC': 'AUC (曲线下面积)',
    'ROC': 'ROC曲线',
    'PR': 'PR曲线 (精确率-召回率)',
    'DCA': '决策曲线分析',
    'OR': '比值比 (OR)',
    'CI': '置信区间',
    'p-value': 'P值',
    
    # 研究相关
    'Lead': '铅',
    'Blood Lead': '血铅',
    'Urine Lead': '尿铅',
    'Lead Colic': '铅绞痛',
    'Risk Factor': '风险因素',
    'Biomarker': '生物标志物',
    
    # 分析类型
    'Univariate': '单因素分析',
    'Multivariate': '多因素分析',
    'Subgroup': '亚组分析',
    'Calibration': '校准曲线',
    'SHAP': 'SHAP分析',
    'Feature Importance': '特征重要性',
    
    # 输出标签
    'Sensitivity': '灵敏度',
    'Specificity': '特异度',
    'PPV': '阳性预测值',
    'NPV': '阴性预测值',
    'Accuracy': '准确率',
    'Precision': '精确率',
    'Recall': '召回率',
    'F1-Score': 'F1分数',
    
    # 图表标题
    'ROC Curve Comparison': 'ROC曲线对比',
    'PR Curve Comparison': 'PR曲线对比',
    'Decision Curve Analysis': '决策曲线分析',
    'Calibration Curve': '校准曲线',
    'Confusion Matrix': '混淆矩阵',
    'Feature Importance': '特征重要性',
    'Model Performance': '模型性能',
    'Risk Distribution': '风险分布',
    'Forest Plot': '森林图',
    
    # 研究领域
    'Network Toxicology': '网络毒理学',
    'Heavy Metal': '重金属',
    'Neurotoxicity': '神经毒性',
    'Gut-Brain Axis': '肠-脑轴',
    'CKM Syndrome': 'CKM综合征',
}


def get_term(key):
    """获取术语的中文翻译"""
    return TERMS.get(key, key)


if __name__ == "__main__":
    setup_chinese_font()
    print("中文配置已加载")
    print(f"可用术语数量: {len(TERMS)}")
