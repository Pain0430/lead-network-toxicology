#!/usr/bin/env python3
"""
铅 (Lead) 与 CKM 综合征研究 - 最终版
Network Toxicology + CKM Syndrome + Mediation Analysis

发现最强中介: 收缩压 (SBP)
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "nhanes_data"
OUTPUT_DIR = "output"

def main():
    print("="*60)
    print("🔬 铅与CKM综合征研究 - 最终版")
    print("="*60)
    
    # 加载数据
    print("\n📂 加载数据...")
    pbbcd = pd.read_sas(f"{DATA_DIR}/PBCD_L.xpt")
    demo = pd.read_sas(f"{DATA_DIR}/DEMO_L.xpt")
    bpxo = pd.read_sas(f"{DATA_DIR}/BPXO_L.xpt")
    bmx = pd.read_sas(f"{DATA_DIR}/BMX_L.xpt")
    hdl = pd.read_sas(f"{DATA_DIR}/HDL_L.xpt")
    trigly = pd.read_sas(f"{DATA_DIR}/TRIGLY_L.xpt")
    ghb = pd.read_sas(f"{DATA_DIR}/GHB_L.xpt")
    mcq = pd.read_sas(f"{DATA_DIR}/MCQ_L.xpt")
    
    # 合并
    df = pbbcd[['SEQN', 'LBXBPB']].copy()
    df.columns = ['SEQN', 'Blood_Lead']
    
    demo_sub = demo[['SEQN', 'RIDAGEYR', 'RIAGENDR']].copy()
    demo_sub.columns = ['SEQN', 'Age', 'Gender']
    df = df.merge(demo_sub, on='SEQN', how='left')
    
    bpxo_sub = bpxo[['SEQN', 'BPXOSY1', 'BPXODI1']].copy()
    bpxo_sub.columns = ['SEQN', 'SBP', 'DBP']
    df = df.merge(bpxo_sub, on='SEQN', how='left')
    
    bmx_sub = bmx[['SEQN', 'BMXBMI', 'BMXWAIST']].copy()
    bmx_sub.columns = ['SEQN', 'BMI', 'Waist']
    df = df.merge(bmx_sub, on='SEQN', how='left')
    
    hdl_sub = hdl[['SEQN', 'LBDHDD']].copy()
    hdl_sub.columns = ['SEQN', 'HDL']
    df = df.merge(hdl_sub, on='SEQN', how='left')
    
    trigly_sub = trigly[['SEQN', 'LBXTLG']].copy()
    trigly_sub.columns = ['SEQN', 'TG']
    df = df.merge(trigly_sub, on='SEQN', how='left')
    
    ghb_sub = ghb[['SEQN', 'LBXGH']].copy()
    ghb_sub.columns = ['SEQN', 'HbA1c']
    df = df.merge(ghb_sub, on='SEQN', how='left')
    
    mcq_sub = mcq[['SEQN', 'MCQ010', 'MCQ160A', 'MCQ160B', 'MCQ160C', 'MCQ160D']].copy()
    mcq_sub.columns = ['SEQN', 'DM_Dx', 'HTN_Dx', 'CHD', 'CKD', 'Stroke']
    df = df.merge(mcq_sub, on='SEQN', how='left')
    
    # 计算指标
    df['High_Waist'] = np.where(df['Gender']==2, df['Waist']>80, df['Waist']>90).astype(float)
    df['High_TG'] = (df['TG'] >= 150).astype(float)
    df['Low_HDL'] = np.where(df['Gender']==2, df['HDL']<50, df['HDL']<40).astype(float)
    df['High_BP'] = ((df['SBP']>=130) | (df['DBP']>=85)).astype(float)
    df['High_HbA1c'] = (df['HbA1c']>=5.7).astype(float)
    
    df['MetS'] = df['High_Waist'].fillna(0) + df['High_TG'].fillna(0) + df['Low_HDL'].fillna(0) + df['High_BP'].fillna(0) + df['High_HbA1c'].fillna(0)
    
    df['HTN'] = df['HTN_Dx'].fillna(0).replace({2:0,7:0,9:0})
    df['DM'] = df['DM_Dx'].fillna(0).replace({2:0,7:0,9:0})
    df['CHD'] = df['CHD'].fillna(0).replace({2:0,7:0,9:0})
    df['CKD'] = df['CKD'].fillna(0).replace({2:0,7:0,9:0})
    
    df['CKM_Score'] = df['HTN'] + df['DM'] + df['CHD'] + df['CKD'] + df['MetS'].fillna(0)
    
    # 分析
    df_clean = df.dropna(subset=['Blood_Lead', 'CKM_Score'])
    print(f"样本量: {len(df_clean)}")
    
    print("\n" + "="*60)
    print("📊 核心发现: 铅与CKM指标的相关性")
    print("="*60)
    
    pairs = [
        ('SBP', '收缩压'),
        ('CKM_Score', 'CKM风险评分'),
        ('MetS', '代谢综合征'),
        ('HbA1c', '糖化血红蛋白'),
        ('Waist', '腰围'),
        ('BMI', 'BMI'),
        ('TG', '甘油三酯'),
    ]
    
    for col, name in pairs:
        data = df_clean[['Blood_Lead', col]].dropna()
        r, p = stats.spearmanr(data['Blood_Lead'], data[col])
        sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
        print(f"   铅 vs {name}: r={r:.3f}, p={p:.2e} {sig}")
    
    # 中介效应: SBP
    print("\n" + "="*60)
    print("📊 中介效应分析: 铅 → 收缩压 → CKM风险")
    print("="*60)
    
    med = df_clean[['Blood_Lead', 'SBP', 'CKM_Score']].dropna()
    X, M, Y = med['Blood_Lead'].values, med['SBP'].values, med['CKM_Score'].values
    
    # a路径
    a, _, _, pa, _ = stats.linregress(X, M)
    print(f"\n路径a (铅→收缩压): β={a:.4f}, p={pa:.4f}")
    
    # b路径  
    Xm = np.column_stack([np.ones(len(X)), X, M])
    beta = np.linalg.lstsq(Xm, Y, rcond=None)[0]
    y_pred = Xm @ beta
    n = len(Y)
    mse = np.sum((Y-y_pred)**2)/(n-3)
    cov = mse * np.linalg.inv(Xm.T @ Xm)
    se_b = np.sqrt(cov[2,2])
    b, pb = beta[2], stats.t.sf(abs(beta[2]/se_b), n-3)*2
    print(f"路径b (收缩压→CKM, 控制铅): β={b:.4f}, p={pb:.4f}")
    
    # c路径
    c, _, pc, _, _ = stats.linregress(X, Y)
    print(f"路径c (铅→CKM, 总效应): β={c:.4f}, p={pc:.4f}")
    
    # 间接效应
    indirect = a * b
    direct = beta[1]
    print(f"\n间接效应 (a×b): {indirect:.4f}")
    print(f"直接效应: {direct:.4f}")
    if c != 0:
        print(f"中介占比: {indirect/c*100:.1f}%")
    
    # 保存
    df.to_csv(f"{OUTPUT_DIR}/lead_ckm_final.csv", index=False)
    print(f"\n✅ 已保存到 {OUTPUT_DIR}/lead_ckm_final.csv")

if __name__ == "__main__":
    main()
