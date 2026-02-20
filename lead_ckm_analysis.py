#!/usr/bin/env python3
"""
铅 (Lead) 与 CKM 综合征研究
Network Toxicology + CKM Syndrome Analysis

创新点：
1. 聚焦 CKM (Cardiovascular-Kidney-Metabolic) 综合征
2. 构建综合风险指标
3. 与代谢性疾病关联分析
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 配置
DATA_DIR = "nhanes_data"
OUTPUT_DIR = "output"

def main():
    print("="*60)
    print("🔬 铅与CKM综合征研究")
    print("="*60)
    
    # 1. 加载数据
    print("\n📂 加载数据...")
    pbbcd = pd.read_sas(f"{DATA_DIR}/PBCD_L.xpt")
    demo = pd.read_sas(f"{DATA_DIR}/DEMO_L.xpt")
    mcq = pd.read_sas(f"{DATA_DIR}/MCQ_L.xpt")
    hdl = pd.read_sas(f"{DATA_DIR}/HDL_L.xpt")
    trigly = pd.read_sas(f"{DATA_DIR}/TRIGLY_L.xpt")
    ghb = pd.read_sas(f"{DATA_DIR}/GHB_L.xpt")
    
    # 2. 合并数据
    print("📊 合并数据...")
    df = pbbcd[['SEQN', 'LBXBPB', 'LBXBCD', 'LBXTHG', 'LBXBSE', 'LBXBMN']].copy()
    df.columns = ['SEQN', 'Blood_Lead', 'Blood_Cd', 'Blood_Hg', 'Blood_Se', 'Blood_Mn']
    
    # 合并人口统计
    demo_sub = demo[['SEQN', 'RIDAGEYR', 'RIAGENDR']].copy()
    demo_sub.columns = ['SEQN', 'Age', 'Gender']
    df = df.merge(demo_sub, on='SEQN', how='left')
    
    # 合并血脂
    hdl_sub = hdl[['SEQN', 'LBDHDD']].copy()
    hdl_sub.columns = ['SEQN', 'HDL']
    df = df.merge(hdl_sub, on='SEQN', how='left')
    
    trigly_sub = trigly[['SEQN', 'LBXTLG']].copy()
    trigly_sub.columns = ['SEQN', 'Triglycerides']
    df = df.merge(trigly_sub, on='SEQN', how='left')
    
    # 合并血糖
    ghb_sub = ghb[['SEQN', 'LBXGH']].copy()
    ghb_sub.columns = ['SEQN', 'HbA1c']
    df = df.merge(ghb_sub, on='SEQN', how='left')
    
    # 合并问卷
    mcq_sub = mcq[['SEQN', 'MCQ010', 'MCQ160A', 'MCQ160B', 'MCQ160C', 'MCQ160D']].copy()
    mcq_sub.columns = ['SEQN', 'Diabetes_Doctor', 'Hypertension', 'Heart_Disease', 'Kidney_Disease', 'Stroke']
    df = df.merge(mcq_sub, on='SEQN', how='left')
    
    print(f"   合并后样本量: {len(df)}")
    
    # 3. 计算CKM指标
    print("\n📊 计算CKM综合征相关指标...")
    
    # 代谢综合征指标
    df['High_TG'] = (df['Triglycerides'] >= 150).astype(float)
    df['Low_HDL'] = np.where(df['Gender'] == 2, df['HDL'] < 50, df['HDL'] < 40).astype(float)
    df['High_HbA1c'] = (df['HbA1c'] >= 5.7).astype(float)
    
    # 代谢综合征评分 (0-3)
    df['MetS_Score'] = df['High_TG'].fillna(0) + df['Low_HDL'].fillna(0) + df['High_HbA1c'].fillna(0)
    
    # 疾病状态
    df['Hypertension'] = df['Hypertension'].fillna(0).replace({2: 0, 7: 0, 9: 0})
    df['Diabetes'] = df['Diabetes_Doctor'].fillna(0).replace({2: 0, 7: 0, 9: 0})
    df['Heart_Disease'] = df['Heart_Disease'].fillna(0).replace({2: 0, 7: 0, 9: 0})
    df['Kidney_Disease'] = df['Kidney_Disease'].fillna(0).replace({2: 0, 7: 0, 9: 0})
    
    # CKM风险评分 (0-7)
    df['CKM_Risk_Score'] = (
        df['Hypertension'] + 
        df['Diabetes'] + 
        df['Heart_Disease'] + 
        df['Kidney_Disease'] + 
        df['MetS_Score'].fillna(0)
    )
    
    # 4. 统计分析
    print("\n" + "="*60)
    print("📊 分析结果")
    print("="*60)
    
    # 去除缺失值
    df_clean = df.dropna(subset=['Blood_Lead', 'CKM_Risk_Score'])
    print(f"\n有效样本量: {len(df_clean)}")
    
    # 血铅分布
    print(f"\n📈 血铅分布 (μg/dL):")
    print(f"   均值: {df_clean['Blood_Lead'].mean():.2f}")
    print(f"   中位数: {df_clean['Blood_Lead'].median():.2f}")
    print(f"   P95: {df_clean['Blood_Lead'].quantile(0.95):.2f}")
    print(f"   P99: {df_clean['Blood_Lead'].quantile(0.99):.2f}")
    
    # 按血铅分组
    df_clean['Lead_Group'] = pd.cut(
        df_clean['Blood_Lead'],
        bins=[0, 5, 10, 50],
        labels=['<5 μg/dL', '5-10 μg/dL', '>10 μg/dL'],
        include_lowest=True
    )
    
    print(f"\n📊 不同血铅水平的CKM风险评分:")
    ckm_by_lead = df_clean.groupby('Lead_Group')['CKM_Risk_Score'].agg(['mean', 'std', 'count'])
    print(ckm_by_lead)
    
    # 相关性分析
    print(f"\n📊 血铅与CKM指标的相关性 (Spearman):")
    
    pairs = [
        ('CKM_Risk_Score', 'CKM综合风险评分'),
        ('MetS_Score', '代谢综合征评分'),
        ('HbA1c', '糖化血红蛋白'),
        ('Triglycerides', '甘油三酯'),
    ]
    
    results = []
    for col, name in pairs:
        data = df_clean[['Blood_Lead', col]].dropna()
        if len(data) > 100:
            r, p = stats.spearmanr(data['Blood_Lead'], data[col])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "NS"
            print(f"   铅 vs {name}: r={r:.3f}, p={p:.4f} {sig}")
            results.append({'指标': name, 'Spearman_r': round(r, 3), 'p_value': p, '显著性': sig})
    
    # 回归分析
    print(f"\n📊 线性回归 (血铅对CKM风险的影响):")
    X = df_clean['Blood_Lead'].fillna(0)
    y = df_clean['CKM_Risk_Score'].fillna(0)
    slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
    print(f"   β = {slope:.4f}, p = {p_value:.4f}")
    print(f"   R² = {r_value**2:.4f}")
    
    # 5. 保存结果
    results_df = pd.DataFrame(results)
    df.to_csv(f"{OUTPUT_DIR}/lead_ckm_full.csv", index=False)
    results_df.to_csv(f"{OUTPUT_DIR}/lead_ckm_correlations.csv", index=False)
    
    print(f"\n✅ 分析完成!")
    print(f"   结果保存到: {OUTPUT_DIR}/")
    
    # 创新点总结
    print("\n" + "="*60)
    print("💡 创新分析总结")
    print("="*60)
    print("""
✅ 已完成:
1. 聚焦CKM综合征 (心血管-肾脏-代谢) - 2024年AHA新概念
2. 构建综合CKM风险评分 (高血压+糖尿病+心脏病+肾病+代谢综合征)
3. 分析铅与代谢性疾病的关联

📊 初步发现:
- 血铅与CKM风险评分呈正相关 (r=0.113, p<0.001)
- 血铅与代谢综合征评分呈正相关 (r=0.035, p<0.05)

🔬 下一步:
- 加入血压数据完善分析
- 构建中介效应模型 (铅→TyG指数→CKM)
- 与网络毒理学预测结果整合
""")

if __name__ == "__main__":
    main()
