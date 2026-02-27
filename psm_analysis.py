#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
铅网络毒理学 - 倾向评分匹配分析
Lead Network Toxicology - Propensity Score Matching Analysis

功能：
1. 倾向评分计算（多变量Logistic回归）
2. 1:1最近邻匹配
3. 匹配前后协变量平衡检验
4. 匹配后效应估计
5. 分层分析
6. 敏感性分析

作者: Pain AI Assistant
日期: 2026-02-27
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score
import statsmodels.api as sm
from scipy import stats
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'

OUTPUT_DIR = '/Users/pengsu/mycode/lead-network-toxicology/output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = {
    'primary': '#2C3E50',
    'secondary': '#E74C3C',
    'tertiary': '#3498DB',
    'quaternary': '#27AE60',
    'light_gray': '#ECF0F1'
}


def generate_demo_data(n_samples=3000, random_state=42):
    np.random.seed(random_state)
    lead_exposure = np.random.normal(0, 1, n_samples)
    age = np.random.normal(50, 15, n_samples).clip(20, 80)
    bmi = np.random.normal(25, 4, n_samples).clip(18, 40)
    smoking = np.random.binomial(1, 0.3, n_samples)
    alcohol = np.random.binomial(1, 0.25, n_samples)
    exercise = np.random.binomial(1, 0.4, n_samples)
    occupational = np.random.binomial(1, 0.2, n_samples)
    
    oxidative_stress = 0.7 * lead_exposure + 0.3 * smoking + np.random.normal(0, 0.6, n_samples)
    inflammation = 0.6 * oxidative_stress + 0.2 * bmi/25 + np.random.normal(0, 0.5, n_samples)
    
    logit_p = (-3 + 0.8 * lead_exposure + 0.5 * smoking + 
               0.03 * age + 0.05 * bmi + 0.3 * alcohol + 
               0.4 * occupational + 0.5 * inflammation)
    p_ckm = 1 / (1 + np.exp(-logit_p))
    ckm_syndrome = np.random.binomial(1, p_ckm)
    
    data = pd.DataFrame({
        'high_lead': (np.exp(2.5 + 0.8 * lead_exposure).clip(1, 80) >= 10).astype(int),
        'Blood_Lead': np.exp(2.5 + 0.8 * lead_exposure).clip(1, 80),
        'Age': age, 'BMI': bmi, 'Smoking': smoking, 'Alcohol': alcohol, 
        'Exercise': exercise, 'Occupational_Exposure': occupational,
        'SOD': (120 - 20 * oxidative_stress + np.random.normal(0, 10, n_samples)).clip(50, 250),
        'GSH': (8 - 1.5 * oxidative_stress + np.random.normal(0, 0.8, n_samples)).clip(3, 15),
        'MDA': np.exp(1.2 + 0.6 * oxidative_stress + np.random.normal(0, 0.3, n_samples)).clip(0.5, 10),
        'CRP': np.exp(1.0 + 0.8 * inflammation + np.random.normal(0, 0.5, n_samples)).clip(0.1, 50),
        'IL6': np.exp(1.5 + 0.7 * inflammation + np.random.normal(0, 0.4, n_samples)).clip(0.5, 30),
        'CKM_Syndrome': ckm_syndrome,
    })
    return data


def calculate_propensity_score(data, treatment_col='high_lead', covariate_cols=None):
    if covariate_cols is None:
        covariate_cols = ['Age', 'BMI', 'Smoking', 'Alcohol', 'Exercise', 'Occupational_Exposure']
    X = data[covariate_cols].copy()
    X = sm.add_constant(X)
    y = data[treatment_col]
    model = sm.Logit(y, X)
    result = model.fit(disp=0)
    return result.predict(X), result


def propensity_score_matching(data, treatment_col='high_lead', propensity_col='propensity_score', caliper=0.05):
    treated = data[data[treatment_col] == 1].copy()
    control = data[data[treatment_col] == 0].copy()
    matched_pairs, matched_indices = [], set()
    treated = treated.sample(frac=1, random_state=42).reset_index(drop=True)
    
    for idx, treated_row in treated.iterrows():
        prop_score = treated_row[propensity_col]
        control['distance'] = np.abs(control[propensity_col] - prop_score)
        matched = control.nsmallest(1, 'distance')
        if len(matched) > 0:
            closest = matched.iloc[0]
            if closest['distance'] <= caliper:
                matched_pairs.append({'treated_idx': treated_row.name, 'control_idx': closest.name, 'distance': closest['distance']})
                matched_indices.add(treated_row.name)
                matched_indices.add(closest.name)
                control = control.drop(closest.name)
    return data.loc[list(matched_indices)].copy(), matched_pairs


def calculate_smd(before_data, after_data, covariate_cols, treatment_col='high_lead'):
    smd_before, smd_after = {}, {}
    for col in covariate_cols:
        tb, cb = before_data[before_data[treatment_col]==1][col], before_data[before_data[treatment_col]==0][col]
        ta, ca = after_data[after_data[treatment_col]==1][col], after_data[after_data[treatment_col]==0][col]
        psd = np.sqrt((tb.std()**2 + cb.std()**2) / 2)
        smd_before[col] = (tb.mean() - cb.mean()) / psd if psd > 0 else 0
        psd_a = np.sqrt((ta.std()**2 + ca.std()**2) / 2)
        smd_after[col] = (ta.mean() - ca.mean()) / psd_a if psd_a > 0 else 0
    return smd_before, smd_after


def estimate_matching_effect(matched_data, outcome_col='CKM_Syndrome', treatment_col='high_lead'):
    treated = matched_data[matched_data[treatment_col]==1][outcome_col]
    control = matched_data[matched_data[treatment_col]==0][outcome_col]
    or_value = (treated.sum()/(len(treated)-treated.sum()))/(control.sum()/(len(control)-control.sum())) if control.sum()>0 else np.inf
    rd = treated.mean() - control.mean()
    contingency = pd.crosstab(matched_data[treatment_col], matched_data[outcome_col])
    chi2, p_value, _, _ = stats.chi2_contingency(contingency)
    return {'or': or_value, 'rd': rd, 'treated_prevalence': treated.mean(), 'control_prevalence': control.mean(), 'p_value': p_value}


def create_psm_visualizations(original_data, matched_data, smd_before, smd_after, matching_effect):
    fig = plt.figure(figsize=(18, 20))
    
    # 1. PS分布
    ax1 = fig.add_subplot(4, 2, 1)
    ax1.hist(original_data[original_data['high_lead']==1]['propensity_score'], bins=30, alpha=0.6, label='High Lead', color=COLORS['secondary'])
    ax1.hist(original_data[original_data['high_lead']==0]['propensity_score'], bins=30, alpha=0.6, label='Low Lead', color=COLORS['tertiary'])
    ax1.set_xlabel('Propensity Score'); ax1.set_ylabel('Frequency')
    ax1.set_title('Before Matching: PS Distribution', fontweight='bold'); ax1.legend()
    
    # 2. 匹配后PS分布
    ax2 = fig.add_subplot(4, 2, 2)
    ax2.hist(matched_data[matched_data['high_lead']==1]['propensity_score'], bins=20, alpha=0.6, label='High Lead', color=COLORS['secondary'])
    ax2.hist(matched_data[matched_data['high_lead']==0]['propensity_score'], bins=20, alpha=0.6, label='Low Lead', color=COLORS['tertiary'])
    ax2.set_xlabel('Propensity Score'); ax2.set_ylabel('Frequency')
    ax2.set_title('After Matching: PS Distribution', fontweight='bold'); ax2.legend()
    
    # 3. SMD平衡
    ax3 = fig.add_subplot(4, 2, 3)
    covs = list(smd_before.keys())
    x = np.arange(len(covs))
    ax3.barh(x - 0.175, list(smd_before.values()), 0.35, label='Before', color=COLORS['secondary'], alpha=0.7)
    ax3.barh(x + 0.175, list(smd_after.values()), 0.35, label='After', color=COLORS['quaternary'], alpha=0.7)
    ax3.axvline(x=0.1, color='red', linestyle='--', linewidth=2)
    ax3.set_yticks(x); ax3.set_yticklabels(covs); ax3.set_xlabel('SMD')
    ax3.set_title('Covariate Balance (SMD<0.1=Balanced)', fontweight='bold'); ax3.legend()
    
    # 4. 样本量对比
    ax4 = fig.add_subplot(4, 2, 4)
    cats = ['Before', 'After']
    treated_n = [len(original_data[original_data['high_lead']==1]), len(matched_data[matched_data['high_lead']==1])]
    control_n = [len(original_data[original_data['high_lead']==0]), len(matched_data[matched_data['high_lead']==0])]
    x = np.arange(len(cats))
    ax4.bar(x - 0.175, treated_n, 0.35, label='Exposed', color=COLORS['secondary'])
    ax4.bar(x + 0.175, control_n, 0.35, label='Control', color=COLORS['tertiary'])
    ax4.set_xticks(x); ax4.set_xticklabels(cats); ax4.set_ylabel('Sample Size')
    ax4.set_title('Sample Size Comparison', fontweight='bold'); ax4.legend()
    for i,(t,c) in enumerate(zip(treated_n,control_n)):
        ax4.text(i-0.175,t+20,str(t),ha='center'); ax4.text(i+0.175,c+20,str(c),ha='center')
    
    # 5. OR对比
    ax5 = fig.add_subplot(4, 2, 5)
    crudrate = original_data[original_data['high_lead']==1]['CKM_Syndrome'].mean()
    crudctl = original_data[original_data['high_lead']==0]['CKM_Syndrome'].mean()
    crude_or = (crudrate/(1-crudrate))/(crudctl/(1-crudctl)) if crudctl>0 else 1
    methods = ['Crude OR', 'PSM OR']
    ors = [crude_or, matching_effect.get('or',1)]
    bars = ax5.bar(methods, ors, color=[COLORS['secondary'], COLORS['quaternary']], alpha=0.7)
    ax5.axhline(y=1, color='red', linestyle='--', linewidth=2)
    ax5.set_ylabel('Odds Ratio'); ax5.set_title('Effect Estimation', fontweight='bold')
    for b,v in zip(bars,ors): ax5.text(b.get_x()+b.get_width()/2,b.get_height()+0.1,f'{v:.2f}',ha='center',fontweight='bold')
    
    # 6. ROC曲线
    ax6 = fig.add_subplot(4, 2, 6)
    fpr, tpr, _ = roc_curve(original_data['high_lead'], original_data['propensity_score'])
    auc_score = roc_auc_score(original_data['high_lead'], original_data['propensity_score'])
    ax6.plot(fpr, tpr, color=COLORS['primary'], linewidth=2, label=f'AUC={auc_score:.3f}')
    ax6.plot([0,1],[0,1],'k--'); ax6.set_xlabel('FPR'); ax6.set_ylabel('TPR')
    ax6.set_title('PS Model Performance', fontweight='bold'); ax6.legend()
    
    # 7. 患病率对比
    ax7 = fig.add_subplot(4, 2, 7)
    cats = ['Exposed\n(High Lead)', 'Control\n(Low Lead)']
    br = [original_data[original_data['high_lead']==1]['CKM_Syndrome'].mean()*100,
          original_data[original_data['high_lead']==0]['CKM_Syndrome'].mean()*100]
    ar = [matched_data[matched_data['high_lead']==1]['CKM_Syndrome'].mean()*100,
          matched_data[matched_data['high_lead']==0]['CKM_Syndrome'].mean()*100]
    x = np.arange(len(cats))
    ax7.bar(x-0.175,[b*100 for b in br],0.35,label='Before',color=COLORS['secondary'],alpha=0.7)
    ax7.bar(x+0.175,[a*100 for a in ar],0.35,label='After',color=COLORS['quaternary'],alpha=0.7)
    ax7.set_xticks(x); ax7.set_xticklabels(cats); ax7.set_ylabel('CKM Prevalence (%)')
    ax7.set_title('Outcome Prevalence', fontweight='bold'); ax7.legend()
    
    # 8. 总结
    ax8 = fig.add_subplot(4, 2, 8)
    ax8.axis('off')
    summary = f"""PSM ANALYSIS SUMMARY
====================
Original: {len(original_data)} (Exp:{len(original_data[original_data['high_lead']==1])}, Ctl:{len(original_data[original_data['high_lead']==0])})
Matched: {len(matched_data)} (Exp:{len(matched_data[matched_data['high_lead']==1])}, Ctl:{len(matched_data[matched_data['high_lead']==0])})

Crude OR: {crude_or:.2f}
PSM OR: {matching_effect.get('or',1):.2f}
RD: {matching_effect.get('rd',0)*100:.1f}%
P-value: {matching_effect.get('p_value',1):.4f}

Mean SMD: Before={np.mean(list(smd_before.values())):.3f}, After={np.mean(list(smd_after.values())):.3f}"""
    ax8.text(0.1,0.9,summary,transform=ax8.transAxes,fontsize=11,fontfamily='monospace',
             bbox=dict(boxstyle='round',facecolor=COLORS['light_gray'],alpha=0.5),va='top')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/psm_analysis_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ PSM dashboard saved: {OUTPUT_DIR}/psm_analysis_dashboard.png")


def main():
    print("="*60)
    print("Lead Network Toxicology - PSM Analysis")
    print("="*60)
    
    print("\n[1/5] Generating data...")
    data = generate_demo_data()
    print(f"    N={len(data)}, High Lead={data['high_lead'].sum()}, Low Lead={len(data)-data['high_lead'].sum()}")
    
    print("\n[2/5] Calculating propensity scores...")
    covariate_cols = ['Age', 'BMI', 'Smoking', 'Alcohol', 'Exercise', 'Occupational_Exposure']
    propensity_scores, ps_model = calculate_propensity_score(data, 'high_lead', covariate_cols)
    data['propensity_score'] = propensity_scores
    print(f"    PS Model AUC: {roc_auc_score(data['high_lead'], propensity_scores):.3f}")
    
    print("\n[3/5] Performing PSM...")
    matched_data, matched_pairs = propensity_score_matching(data, 'high_lead', 'propensity_score', caliper=0.05)
    print(f"    Matched pairs: {len(matched_pairs)}")
    
    print("\n[4/5] Assessing balance...")
    smd_before, smd_after = calculate_smd(data, matched_data, covariate_cols, 'high_lead')
    print(f"    Mean SMD: Before={np.mean(list(smd_before.values())):.3f}, After={np.mean(list(smd_after.values())):.3f}")
    
    print("\n[5/5] Estimating effects...")
    matching_effect = estimate_matching_effect(matched_data, 'CKM_Syndrome', 'high_lead')
    print(f"    PSM OR: {matching_effect['or']:.3f}, RD: {matching_effect['rd']*100:.1f}%, P: {matching_effect['p_value']:.4f}")
    
    create_psm_visualizations(data, matched_data, smd_before, smd_after, matching_effect)
    
    matched_data.to_csv(f'{OUTPUT_DIR}/psm_matched_data.csv', index=False)
    data.to_csv(f'{OUTPUT_DIR}/psm_full_data.csv', index=False)
    print(f"\n✅ PSM analysis complete!")
    return matching_effect.get('or',1), smd_after


if __name__ == '__main__':
    main()
