#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中介效应分析模块 - 铅暴露 → 血压 → CKM综合征
Mediation Analysis Module: Lead Exposure → Blood Pressure → CKM Syndrome

方法：
1. Baron-Kenny 四步法
2. Sobel检验
3. Bootstrap置信区间法
4. 多中介变量模型
5. 交互式可视化（路径图）

理论基础：
- 总效应 (c) = 直接效应 (c') + 间接效应 (a*b)
- 中介比例 = a*b / c × 100%

作者: Pain's AI Assistant
日期: 2026-02-23
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 配置
# ============================================================================

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 变量标签
VARIABLE_LABELS = {
    'LBXBPB': 'Blood Lead (μg/dL)',
    'BPXOSY1': 'Systolic BP (mmHg)',
    'BPXODI1': 'Diastolic BP (mmHg)',
    'LBXGH': 'HbA1c (%)',
    'LBXSATSI': 'ALT (U/L)',
    'LBXSCR': 'Creatinine (mg/dL)',
    'LBXSTR': 'Triglycerides (mg/dL)',
    'LBXSUA': 'Uric Acid (mg/dL)',
    'LBDHDD': 'HDL (mg/dL)',
    'BMXBMI': 'BMI (kg/m²)',
    'ckm_stage': 'CKM Stage',
    'hypertension': 'Hypertension',
    'mets_score': 'Metabolic Score',
    'egfr': 'eGFR (mL/min/1.73m²)',
}


# ============================================================================
# OLS回归工具（不依赖statsmodels）
# ============================================================================

class OLSResult:
    """简单OLS回归结果"""
    def __init__(self, X, y):
        # 添加截距列
        n = len(y)
        X_with_const = np.column_stack([np.ones(n), X])
        
        # OLS估计: beta = (X'X)^-1 X'y
        XtX = X_with_const.T @ X_with_const
        Xty = X_with_const.T @ y
        self.beta = np.linalg.solve(XtX, Xty)
        
        # 残差和R²
        y_hat = X_with_const @ self.beta
        residuals = y - y_hat
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        self.r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        # 标准误
        p = X_with_const.shape[1]
        mse = ss_res / (n - p)
        var_beta = mse * np.linalg.inv(XtX)
        self.se = np.sqrt(np.diag(var_beta))
        
        # t统计量和p值
        self.t_values = self.beta / self.se
        self.p_values = 2 * (1 - stats.t.cdf(np.abs(self.t_values), n - p))
        
        # 95%置信区间
        t_crit = stats.t.ppf(0.975, n - p)
        self.ci_lower = self.beta - t_crit * self.se
        self.ci_upper = self.beta + t_crit * self.se
        
        self.n = n
        self.df = n - p
        
        # F检验
        if ss_tot > 0 and (p - 1) > 0:
            f_stat = (ss_tot - ss_res) / (p - 1) / (ss_res / (n - p))
            self.f_pvalue = 1 - stats.f.cdf(f_stat, p - 1, n - p)
        else:
            self.f_pvalue = 1.0
    
    @property
    def intercept(self):
        return self.beta[0]
    
    @property
    def coefficients(self):
        return self.beta[1:]
    
    @property
    def coef_pvalues(self):
        return self.p_values[1:]
    
    @property
    def coef_se(self):
        return self.se[1:]


def ols_regression(X, y):
    """执行OLS回归"""
    if isinstance(X, pd.Series):
        X = X.values.reshape(-1, 1)
    elif isinstance(X, pd.DataFrame):
        X = X.values
    elif X.ndim == 1:
        X = X.reshape(-1, 1)
    
    if isinstance(y, pd.Series):
        y = y.values
    
    return OLSResult(X, y)


# ============================================================================
# Baron-Kenny 中介效应分析
# ============================================================================

def baron_kenny_mediation(df, x_col, m_col, y_col, covariates=None):
    """
    Baron-Kenny四步法中介效应分析
    
    步骤：
    1. X → Y (总效应 c)
    2. X → M (a路径)
    3. X + M → Y (直接效应 c' 和 b路径)
    4. 间接效应 = a * b, 中介比例 = a*b/c
    
    Args:
        df: 数据框
        x_col: 自变量（暴露）
        m_col: 中介变量
        y_col: 因变量（结果）
        covariates: 协变量列表
    
    Returns:
        dict: 中介分析结果
    """
    # 准备数据
    cols = [x_col, m_col, y_col]
    if covariates:
        cols.extend(covariates)
    data = df[cols].dropna()
    
    if len(data) < 30:
        return {'error': '样本量不足 (n < 30)'}
    
    X = data[x_col].values
    M = data[m_col].values
    Y = data[y_col].values
    
    # 准备协变量矩阵
    if covariates and len(covariates) > 0:
        C = data[covariates].values
    else:
        C = None
    
    # =====================
    # Step 1: X → Y (总效应 c)
    # =====================
    if C is not None:
        X_step1 = np.column_stack([X, C])
    else:
        X_step1 = X
    
    model1 = ols_regression(X_step1, Y)
    c_total = model1.coefficients[0]
    c_se = model1.coef_se[0]
    c_p = model1.coef_pvalues[0]
    
    # =====================
    # Step 2: X → M (a路径)
    # =====================
    if C is not None:
        X_step2 = np.column_stack([X, C])
    else:
        X_step2 = X
    
    model2 = ols_regression(X_step2, M)
    a = model2.coefficients[0]
    a_se = model2.coef_se[0]
    a_p = model2.coef_pvalues[0]
    
    # =====================
    # Step 3: X + M → Y (c' 和 b)
    # =====================
    if C is not None:
        X_step3 = np.column_stack([X, M, C])
    else:
        X_step3 = np.column_stack([X, M])
    
    model3 = ols_regression(X_step3, Y)
    c_prime = model3.coefficients[0]  # 直接效应
    c_prime_se = model3.coef_se[0]
    c_prime_p = model3.coef_pvalues[0]
    b = model3.coefficients[1]  # M → Y (控制X)
    b_se = model3.coef_se[1]
    b_p = model3.coef_pvalues[1]
    
    # =====================
    # 间接效应
    # =====================
    indirect = a * b
    
    # Sobel检验
    sobel_se = np.sqrt(a**2 * b_se**2 + b**2 * a_se**2)
    sobel_z = indirect / sobel_se if sobel_se > 0 else 0
    sobel_p = 2 * (1 - stats.norm.cdf(abs(sobel_z)))
    
    # 中介比例
    mediation_pct = (indirect / c_total * 100) if abs(c_total) > 1e-10 else 0
    
    # Baron-Kenny 判断标准
    step1_sig = c_p < 0.05       # 总效应显著
    step2_sig = a_p < 0.05       # a路径显著
    step3_b_sig = b_p < 0.05     # b路径显著
    step3_c_sig = c_prime_p < 0.05  # 直接效应是否仍显著
    
    if step1_sig and step2_sig and step3_b_sig:
        if not step3_c_sig:
            mediation_type = "完全中介 (Full Mediation)"
        else:
            mediation_type = "部分中介 (Partial Mediation)"
    else:
        mediation_type = "无中介效应 (No Mediation)"
    
    return {
        # 模型信息
        'exposure': x_col,
        'mediator': m_col,
        'outcome': y_col,
        'n': len(data),
        'covariates': covariates or [],
        
        # 路径系数
        'c_total': c_total,         # 总效应
        'c_total_se': c_se,
        'c_total_p': c_p,
        'a_path': a,                # X → M
        'a_se': a_se,
        'a_p': a_p,
        'b_path': b,                # M → Y (控制X)
        'b_se': b_se,
        'b_p': b_p,
        'c_prime': c_prime,         # 直接效应
        'c_prime_se': c_prime_se,
        'c_prime_p': c_prime_p,
        
        # 间接效应
        'indirect_effect': indirect,
        'sobel_z': sobel_z,
        'sobel_p': sobel_p,
        'sobel_se': sobel_se,
        
        # 中介比例
        'mediation_pct': mediation_pct,
        'mediation_type': mediation_type,
        
        # R²
        'r2_total': model1.r_squared,
        'r2_mediator': model2.r_squared,
        'r2_full': model3.r_squared,
        
        # 判断
        'step1_sig': step1_sig,
        'step2_sig': step2_sig,
        'step3_b_sig': step3_b_sig,
        'step3_c_sig': step3_c_sig,
    }


# ============================================================================
# Bootstrap置信区间法
# ============================================================================

def bootstrap_mediation(df, x_col, m_col, y_col, covariates=None,
                        n_bootstrap=5000, ci_level=0.95, seed=42):
    """
    Bootstrap置信区间法估计间接效应
    
    Args:
        df: 数据框
        x_col: 自变量
        m_col: 中介变量
        y_col: 因变量
        covariates: 协变量列表
        n_bootstrap: Bootstrap样本数
        ci_level: 置信水平
        seed: 随机种子
    
    Returns:
        dict: Bootstrap结果
    """
    np.random.seed(seed)
    
    cols = [x_col, m_col, y_col]
    if covariates:
        cols.extend(covariates)
    data = df[cols].dropna()
    n = len(data)
    
    if n < 30:
        return {'error': '样本量不足'}
    
    indirect_effects = []
    a_boots = []
    b_boots = []
    
    for _ in range(n_bootstrap):
        # 有放回抽样
        sample = data.sample(n=n, replace=True)
        
        X = sample[x_col].values
        M = sample[m_col].values
        Y = sample[y_col].values
        
        if covariates:
            C = sample[covariates].values
            X_a = np.column_stack([X, C])
            X_b = np.column_stack([X, M, C])
        else:
            X_a = X
            X_b = np.column_stack([X, M])
        
        try:
            # a路径: X → M
            model_a = ols_regression(X_a, M)
            a = model_a.coefficients[0]
            
            # b路径: M → Y (控制X)
            model_b = ols_regression(X_b, Y)
            b = model_b.coefficients[1]
            
            indirect_effects.append(a * b)
            a_boots.append(a)
            b_boots.append(b)
        except Exception:
            continue
    
    indirect_effects = np.array(indirect_effects)
    a_boots = np.array(a_boots)
    b_boots = np.array(b_boots)
    
    # 置信区间
    alpha = 1 - ci_level
    ci_lower = np.percentile(indirect_effects, alpha / 2 * 100)
    ci_upper = np.percentile(indirect_effects, (1 - alpha / 2) * 100)
    
    # Bias-corrected CI
    z0 = stats.norm.ppf(np.mean(indirect_effects < np.mean(indirect_effects)))
    za = stats.norm.ppf(alpha / 2)
    zb = stats.norm.ppf(1 - alpha / 2)
    
    bc_lower_pct = stats.norm.cdf(2 * z0 + za) * 100
    bc_upper_pct = stats.norm.cdf(2 * z0 + zb) * 100
    bc_ci_lower = np.percentile(indirect_effects, bc_lower_pct)
    bc_ci_upper = np.percentile(indirect_effects, bc_upper_pct)
    
    # 显著性: 置信区间不包含0
    significant = (ci_lower > 0 and ci_upper > 0) or (ci_lower < 0 and ci_upper < 0)
    bc_significant = (bc_ci_lower > 0 and bc_ci_upper > 0) or (bc_ci_lower < 0 and bc_ci_upper < 0)
    
    return {
        'n_bootstrap': len(indirect_effects),
        'indirect_mean': np.mean(indirect_effects),
        'indirect_se': np.std(indirect_effects),
        'ci_level': ci_level,
        
        # Percentile CI
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'significant': significant,
        
        # Bias-corrected CI
        'bc_ci_lower': bc_ci_lower,
        'bc_ci_upper': bc_ci_upper,
        'bc_significant': bc_significant,
        
        # 分布统计
        'a_mean': np.mean(a_boots),
        'a_se': np.std(a_boots),
        'b_mean': np.mean(b_boots),
        'b_se': np.std(b_boots),
    }


# ============================================================================
# 多中介变量模型
# ============================================================================

def multi_mediator_analysis(df, x_col, mediators, y_col, covariates=None):
    """
    多中介变量并行模型
    
    Lead → [SBP, DBP, HbA1c, BMI, ...] → CKM
    
    Args:
        df: 数据框
        x_col: 自变量
        mediators: 中介变量列表
        y_col: 因变量
        covariates: 协变量列表
    
    Returns:
        dict: 各中介路径结果汇总
    """
    results = {}
    
    for m_col in mediators:
        label = VARIABLE_LABELS.get(m_col, m_col)
        
        # Baron-Kenny分析
        bk_result = baron_kenny_mediation(df, x_col, m_col, y_col, covariates)
        
        if 'error' in bk_result:
            results[m_col] = {'label': label, 'error': bk_result['error']}
            continue
        
        # Bootstrap分析
        boot_result = bootstrap_mediation(df, x_col, m_col, y_col, covariates,
                                          n_bootstrap=2000)
        
        results[m_col] = {
            'label': label,
            'baron_kenny': bk_result,
            'bootstrap': boot_result,
        }
    
    return results


# ============================================================================
# 可视化：路径图（HTML）
# ============================================================================

def generate_mediation_path_diagram(result, boot_result=None, output_dir=OUTPUT_DIR):
    """
    生成中介效应路径图（交互式HTML）
    
    Args:
        result: Baron-Kenny结果
        boot_result: Bootstrap结果
        output_dir: 输出目录
    """
    x_label = VARIABLE_LABELS.get(result['exposure'], result['exposure'])
    m_label = VARIABLE_LABELS.get(result['mediator'], result['mediator'])
    y_label = VARIABLE_LABELS.get(result['outcome'], result['outcome'])
    
    # 显著性标注
    def sig_star(p):
        if p < 0.001: return '***'
        elif p < 0.01: return '**'
        elif p < 0.05: return '*'
        return 'ns'
    
    # Bootstrap CI文本
    boot_text = ""
    if boot_result and 'error' not in boot_result:
        boot_text = f"""
        <div class="boot-section">
            <h3>🔄 Bootstrap Confidence Interval (n={boot_result['n_bootstrap']})</h3>
            <table>
                <tr><td>Indirect Effect (mean)</td><td><b>{boot_result['indirect_mean']:.4f}</b></td></tr>
                <tr><td>SE</td><td>{boot_result['indirect_se']:.4f}</td></tr>
                <tr><td>95% Percentile CI</td><td>[{boot_result['ci_lower']:.4f}, {boot_result['ci_upper']:.4f}]</td></tr>
                <tr><td>95% Bias-Corrected CI</td><td>[{boot_result['bc_ci_lower']:.4f}, {boot_result['bc_ci_upper']:.4f}]</td></tr>
                <tr><td>Significant (Percentile)</td><td>{'✅ Yes' if boot_result['significant'] else '❌ No'}</td></tr>
                <tr><td>Significant (BC)</td><td>{'✅ Yes' if boot_result['bc_significant'] else '❌ No'}</td></tr>
            </table>
        </div>
        """
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Mediation Analysis: Lead → BP → CKM</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{ 
            font-family: 'Segoe UI', Arial, sans-serif; 
            background: #f0f2f5; 
            margin: 0; padding: 30px; 
            color: #2c3e50;
        }}
        .container {{ max-width: 1100px; margin: 0 auto; }}
        h1 {{ 
            text-align: center; 
            color: #2c3e50; 
            border-bottom: 3px solid #3498db; 
            padding-bottom: 15px;
            margin-bottom: 30px;
        }}
        
        /* Path Diagram */
        .path-diagram {{
            background: white;
            border-radius: 12px;
            padding: 40px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            margin-bottom: 30px;
            position: relative;
        }}
        .path-svg {{ width: 100%; height: 280px; }}
        
        /* Results Cards */
        .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .card {{
            background: white;
            border-radius: 10px;
            padding: 25px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }}
        .card h3 {{ color: #3498db; margin-top: 0; }}
        .card table {{ width: 100%; border-collapse: collapse; }}
        .card td {{ padding: 8px 4px; border-bottom: 1px solid #eee; }}
        .card td:last-child {{ text-align: right; font-weight: 500; }}
        
        .highlight {{ 
            background: linear-gradient(135deg, #3498db15, #2ecc7115);
            border-left: 4px solid #3498db;
            padding: 20px;
            border-radius: 0 8px 8px 0;
            margin: 20px 0;
        }}
        .highlight .big {{ font-size: 2em; font-weight: bold; color: #2c3e50; }}
        
        .mediation-type {{
            text-align: center;
            font-size: 1.3em;
            font-weight: bold;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .mediation-type.full {{ background: #27ae6020; color: #27ae60; border: 2px solid #27ae60; }}
        .mediation-type.partial {{ background: #f39c1220; color: #e67e22; border: 2px solid #e67e22; }}
        .mediation-type.none {{ background: #e74c3c20; color: #e74c3c; border: 2px solid #e74c3c; }}
        
        .boot-section {{
            background: white;
            border-radius: 10px;
            padding: 25px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            margin-bottom: 30px;
        }}
        .boot-section h3 {{ color: #8e44ad; margin-top: 0; }}
        .boot-section table {{ width: 100%; border-collapse: collapse; }}
        .boot-section td {{ padding: 8px 4px; border-bottom: 1px solid #eee; }}
        .boot-section td:last-child {{ text-align: right; font-weight: 500; }}
        
        .steps {{ margin: 20px 0; }}
        .step {{
            display: flex; align-items: center; gap: 12px;
            padding: 10px 15px; margin: 8px 0;
            border-radius: 8px; background: #f9f9f9;
        }}
        .step-icon {{ 
            width: 30px; height: 30px; border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            font-weight: bold; color: white; font-size: 14px;
        }}
        .step-pass {{ background: #27ae60; }}
        .step-fail {{ background: #e74c3c; }}
        
        .footer {{ text-align: center; color: #95a5a6; font-size: 0.85em; margin-top: 30px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 中介效应分析 Mediation Analysis</h1>
        
        <!-- Path Diagram -->
        <div class="path-diagram">
            <svg class="path-svg" viewBox="0 0 800 280">
                <!-- X box -->
                <rect x="20" y="180" width="180" height="60" rx="10" fill="#3498db" opacity="0.15" stroke="#3498db" stroke-width="2"/>
                <text x="110" y="215" text-anchor="middle" font-size="13" font-weight="bold" fill="#2c3e50">{x_label}</text>
                
                <!-- M box -->
                <rect x="310" y="20" width="180" height="60" rx="10" fill="#e67e22" opacity="0.15" stroke="#e67e22" stroke-width="2"/>
                <text x="400" y="55" text-anchor="middle" font-size="13" font-weight="bold" fill="#2c3e50">{m_label}</text>
                
                <!-- Y box -->
                <rect x="600" y="180" width="180" height="60" rx="10" fill="#27ae60" opacity="0.15" stroke="#27ae60" stroke-width="2"/>
                <text x="690" y="215" text-anchor="middle" font-size="13" font-weight="bold" fill="#2c3e50">{y_label}</text>
                
                <!-- a path: X → M -->
                <line x1="200" y1="190" x2="310" y2="75" stroke="#e67e22" stroke-width="2.5" marker-end="url(#arrow-a)"/>
                <text x="230" y="120" font-size="13" fill="#e67e22" font-weight="bold">
                    a = {result['a_path']:.4f}{sig_star(result['a_p'])}
                </text>
                
                <!-- b path: M → Y -->
                <line x1="490" y1="75" x2="600" y2="190" stroke="#e67e22" stroke-width="2.5" marker-end="url(#arrow-b)"/>
                <text x="530" y="120" font-size="13" fill="#e67e22" font-weight="bold">
                    b = {result['b_path']:.4f}{sig_star(result['b_p'])}
                </text>
                
                <!-- c' path: X → Y (direct) -->
                <line x1="200" y1="210" x2="600" y2="210" stroke="#3498db" stroke-width="2.5" stroke-dasharray="8,4" marker-end="url(#arrow-c)"/>
                <text x="400" y="250" text-anchor="middle" font-size="13" fill="#3498db" font-weight="bold">
                    c' = {result['c_prime']:.4f}{sig_star(result['c_prime_p'])} (direct)
                </text>
                
                <!-- c total -->
                <text x="400" y="275" text-anchor="middle" font-size="11" fill="#95a5a6">
                    c (total) = {result['c_total']:.4f}{sig_star(result['c_total_p'])}
                </text>
                
                <!-- Arrow markers -->
                <defs>
                    <marker id="arrow-a" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="#e67e22"/></marker>
                    <marker id="arrow-b" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="#e67e22"/></marker>
                    <marker id="arrow-c" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="#3498db"/></marker>
                </defs>
            </svg>
        </div>
        
        <!-- Mediation Type -->
        <div class="mediation-type {'full' if 'Full' in result['mediation_type'] else 'partial' if 'Partial' in result['mediation_type'] else 'none'}">
            {result['mediation_type']}
        </div>
        
        <!-- Key Result -->
        <div class="highlight">
            <div class="big">{abs(result['mediation_pct']):.1f}%</div>
            <div>中介效应占总效应的比例 (Proportion Mediated)</div>
            <div style="margin-top:8px; font-size:0.9em; color:#666;">
                间接效应 a×b = {result['indirect_effect']:.4f} | 
                总效应 c = {result['c_total']:.4f} | 
                直接效应 c' = {result['c_prime']:.4f}
            </div>
        </div>
        
        <!-- Baron-Kenny Steps -->
        <div class="card">
            <h3>📋 Baron-Kenny Four Steps</h3>
            <div class="steps">
                <div class="step">
                    <div class="step-icon {'step-pass' if result['step1_sig'] else 'step-fail'}">1</div>
                    <div>X → Y 总效应显著 (c = {result['c_total']:.4f}, p = {result['c_total_p']:.2e}) {'✅' if result['step1_sig'] else '❌'}</div>
                </div>
                <div class="step">
                    <div class="step-icon {'step-pass' if result['step2_sig'] else 'step-fail'}">2</div>
                    <div>X → M a路径显著 (a = {result['a_path']:.4f}, p = {result['a_p']:.2e}) {'✅' if result['step2_sig'] else '❌'}</div>
                </div>
                <div class="step">
                    <div class="step-icon {'step-pass' if result['step3_b_sig'] else 'step-fail'}">3</div>
                    <div>M → Y b路径显著 (b = {result['b_path']:.4f}, p = {result['b_p']:.2e}) {'✅' if result['step3_b_sig'] else '❌'}</div>
                </div>
                <div class="step">
                    <div class="step-icon {'step-pass' if not result['step3_c_sig'] else 'step-fail'}">4</div>
                    <div>直接效应c'减弱或不显著 (c' = {result['c_prime']:.4f}, p = {result['c_prime_p']:.2e}) {'✅ Full' if not result['step3_c_sig'] else '⚠️ Partial'}</div>
                </div>
            </div>
        </div>
        
        <!-- Sobel Test -->
        <div class="cards">
            <div class="card">
                <h3>📊 Sobel Test</h3>
                <table>
                    <tr><td>Indirect Effect (a×b)</td><td>{result['indirect_effect']:.4f}</td></tr>
                    <tr><td>Sobel Z</td><td>{result['sobel_z']:.3f}</td></tr>
                    <tr><td>Sobel p-value</td><td>{result['sobel_p']:.2e}</td></tr>
                    <tr><td>Significant</td><td>{'✅ Yes' if result['sobel_p'] < 0.05 else '❌ No'}</td></tr>
                </table>
            </div>
            <div class="card">
                <h3>📈 Model Fit (R²)</h3>
                <table>
                    <tr><td>X → Y (Total)</td><td>{result['r2_total']:.4f}</td></tr>
                    <tr><td>X → M</td><td>{result['r2_mediator']:.4f}</td></tr>
                    <tr><td>X + M → Y (Full)</td><td>{result['r2_full']:.4f}</td></tr>
                    <tr><td>R² Improvement</td><td>+{(result['r2_full'] - result['r2_total']):.4f}</td></tr>
                </table>
            </div>
        </div>
        
        {boot_text}
        
        <div class="footer">
            <p>Sample size: n = {result['n']} | 
               Covariates: {', '.join(result['covariates']) if result['covariates'] else 'None'}</p>
            <p>Generated by Mediation Analysis Module | Lead Network Toxicology Project</p>
        </div>
    </div>
</body>
</html>"""
    
    filename = os.path.join(output_dir, 'mediation_path_diagram.html')
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Saved: {filename}")
    return filename


# ============================================================================
# 多中介变量对比图
# ============================================================================

def generate_multi_mediator_chart(multi_results, output_dir=OUTPUT_DIR):
    """
    生成多中介变量对比森林图（HTML）
    """
    rows_html = ""
    valid_results = []
    
    for m_col, res in multi_results.items():
        if 'error' in res:
            continue
        bk = res['baron_kenny']
        boot = res.get('bootstrap', {})
        
        label = res['label']
        indirect = bk['indirect_effect']
        pct = bk['mediation_pct']
        sobel_p = bk['sobel_p']
        med_type = bk['mediation_type']
        
        ci_lo = boot.get('ci_lower', 0)
        ci_hi = boot.get('ci_upper', 0)
        sig = boot.get('significant', False)
        
        valid_results.append({
            'label': label,
            'indirect': indirect,
            'ci_lo': ci_lo,
            'ci_hi': ci_hi,
            'pct': pct,
            'sig': sig,
            'sobel_p': sobel_p,
            'type': med_type,
        })
        
        sig_badge = '<span style="color:#27ae60;font-weight:bold">✅</span>' if sig else '<span style="color:#e74c3c">❌</span>'
        
        rows_html += f"""
        <tr>
            <td>{label}</td>
            <td>{indirect:.4f}</td>
            <td>[{ci_lo:.4f}, {ci_hi:.4f}]</td>
            <td>{abs(pct):.1f}%</td>
            <td>{sobel_p:.2e}</td>
            <td>{sig_badge}</td>
            <td>{med_type.split('(')[0].strip()}</td>
        </tr>"""
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Multi-Mediator Comparison</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f0f2f5; margin: 0; padding: 30px; }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #8e44ad; padding-bottom: 10px; }}
        table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        th {{ background: #34495e; color: white; padding: 12px; text-align: left; }}
        td {{ padding: 12px; border-bottom: 1px solid #eee; }}
        tr:hover {{ background: #f5f6fa; }}
        .note {{ color: #666; font-size: 0.85em; margin-top: 15px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 多中介变量对比 Multi-Mediator Comparison</h1>
        <p>暴露变量: Blood Lead (Pb) → 结果变量: CKM Stage</p>
        <table>
            <tr>
                <th>中介变量 Mediator</th>
                <th>间接效应 a×b</th>
                <th>95% Bootstrap CI</th>
                <th>中介比例</th>
                <th>Sobel p</th>
                <th>显著</th>
                <th>类型</th>
            </tr>
            {rows_html}
        </table>
        <p class="note">
            * Bootstrap n=2000 | 显著: 95% CI不含0<br>
            * 中介比例 = 间接效应/总效应 × 100%
        </p>
    </div>
</body>
</html>"""
    
    filename = os.path.join(output_dir, 'multi_mediator_comparison.html')
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Saved: {filename}")
    return filename


# ============================================================================
# 主函数
# ============================================================================

def main():
    """使用模拟数据演示中介效应分析"""
    print("=" * 60)
    print("🔬 中介效应分析 - 铅暴露 → 血压 → CKM综合征")
    print("=" * 60)
    
    # 生成模拟数据
    np.random.seed(42)
    n = 2000
    
    # 人口学变量
    age = np.random.normal(50, 15, n).clip(18, 85)
    sex = np.random.binomial(1, 0.5, n)  # 0=female, 1=male
    bmi = np.random.normal(28, 5, n).clip(15, 50)
    
    # 铅暴露 (对数正态分布)
    log_lead = -0.2 + 0.02 * age + 0.3 * sex + np.random.normal(0, 0.7, n)
    lead = np.exp(log_lead).clip(0.1, 20)
    
    # 收缩压 (受铅影响)
    sbp = (100 + 0.5 * age + 5 * sex + 0.3 * bmi
           + 3.5 * np.log(lead)  # a路径: 铅 → 血压
           + np.random.normal(0, 12, n))
    
    # CKM评分 (受铅和血压影响)
    ckm_score = (0.5 + 0.01 * age + 0.1 * sex
                 + 0.05 * bmi
                 + 0.08 * np.log(lead)    # c'直接效应
                 + 0.02 * (sbp - 120)     # b路径: 血压 → CKM
                 + np.random.normal(0, 0.5, n))
    ckm_stage = np.clip(np.round(ckm_score), 0, 4).astype(int)
    
    # HbA1c (受铅轻微影响)
    hba1c = 5.2 + 0.01 * age + 0.1 * np.log(lead) + np.random.normal(0, 0.4, n)
    
    # eGFR (受铅影响)
    egfr = 110 - 0.8 * age - 2 * np.log(lead) + np.random.normal(0, 10, n)
    
    df = pd.DataFrame({
        'LBXBPB': lead,
        'BPXOSY1': sbp,
        'BPXODI1': sbp - 40 + np.random.normal(0, 8, n),  # DBP
        'LBXGH': hba1c,
        'BMXBMI': bmi,
        'ckm_stage': ckm_stage,
        'egfr': egfr,
        'age': age,
        'sex': sex,
    })
    
    # 对铅取对数
    df['log_lead'] = np.log(df['LBXBPB'])
    
    covariates = ['age', 'sex', 'BMXBMI']
    
    # ==============================
    # 1. Baron-Kenny 分析
    # ==============================
    print("\n📊 Baron-Kenny 四步法分析...")
    print("-" * 40)
    
    bk_result = baron_kenny_mediation(
        df, 'log_lead', 'BPXOSY1', 'ckm_stage', covariates
    )
    
    print(f"  样本量: n = {bk_result['n']}")
    print(f"  总效应 (c): {bk_result['c_total']:.4f} (p = {bk_result['c_total_p']:.2e})")
    print(f"  a路径 (X→M): {bk_result['a_path']:.4f} (p = {bk_result['a_p']:.2e})")
    print(f"  b路径 (M→Y): {bk_result['b_path']:.4f} (p = {bk_result['b_p']:.2e})")
    print(f"  直接效应 (c'): {bk_result['c_prime']:.4f} (p = {bk_result['c_prime_p']:.2e})")
    print(f"  间接效应 (a×b): {bk_result['indirect_effect']:.4f}")
    print(f"  Sobel Z: {bk_result['sobel_z']:.3f} (p = {bk_result['sobel_p']:.2e})")
    print(f"  中介比例: {bk_result['mediation_pct']:.1f}%")
    print(f"  结论: {bk_result['mediation_type']}")
    
    # ==============================
    # 2. Bootstrap 分析
    # ==============================
    print("\n🔄 Bootstrap 置信区间分析 (n=2000)...")
    print("-" * 40)
    
    boot_result = bootstrap_mediation(
        df, 'log_lead', 'BPXOSY1', 'ckm_stage', covariates,
        n_bootstrap=2000
    )
    
    print(f"  间接效应均值: {boot_result['indirect_mean']:.4f}")
    print(f"  95% CI (Percentile): [{boot_result['ci_lower']:.4f}, {boot_result['ci_upper']:.4f}]")
    print(f"  95% CI (Bias-Corrected): [{boot_result['bc_ci_lower']:.4f}, {boot_result['bc_ci_upper']:.4f}]")
    print(f"  显著性: {'✅ Yes' if boot_result['significant'] else '❌ No'}")
    
    # ==============================
    # 3. 路径图
    # ==============================
    print("\n📈 生成路径图...")
    generate_mediation_path_diagram(bk_result, boot_result)
    
    # ==============================
    # 4. 多中介变量对比
    # ==============================
    print("\n🔬 多中介变量对比分析...")
    print("-" * 40)
    
    mediators = ['BPXOSY1', 'BPXODI1', 'LBXGH', 'BMXBMI', 'egfr']
    multi_results = multi_mediator_analysis(
        df, 'log_lead', mediators, 'ckm_stage', ['age', 'sex']
    )
    
    for m_col, res in multi_results.items():
        if 'error' in res:
            print(f"  {res['label']}: {res['error']}")
            continue
        bk = res['baron_kenny']
        print(f"  {res['label']}: 间接效应={bk['indirect_effect']:.4f}, "
              f"中介比例={bk['mediation_pct']:.1f}%, "
              f"{bk['mediation_type']}")
    
    generate_multi_mediator_chart(multi_results)
    
    print("\n" + "=" * 60)
    print("✅ 中介效应分析完成!")
    print("=" * 60)
    print(f"输出文件:")
    print(f"  - output/mediation_path_diagram.html (路径图)")
    print(f"  - output/multi_mediator_comparison.html (多中介对比)")


if __name__ == "__main__":
    main()
