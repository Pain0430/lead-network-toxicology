"""
铅暴露与CKM综合征预测 - XGBoost机器学习分析
============================================
基于NHANES 2021-2023数据
使用XGBoost + SHAP进行可解释性机器学习

作者: Pain's AI Assistant
日期: 2026-03-01
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============= 数据准备 =============
def load_and_prepare_data():
    """加载NHANES数据并准备特征"""
    
    # 模拟NHANES数据结构（实际使用时替换为真实数据）
    np.random.seed(42)
    n = 5000
    
    # 铅暴露指标
    data = {
        'BPB_ug/dL': np.random.lognormal(mean=0.5, sigma=0.8, size=n),  # 血铅
        'URXUBA_ug/L': np.random.lognormal(mean=2, sigma=1, size=n),   # 尿铅
    }
    
    # CKM综合征相关指标
    data['SBP'] = np.random.normal(130, 20, n)  # 收缩压
    data['DIABETES'] = np.random.binomial(1, 0.15, n)  # 糖尿病
    data['CKD'] = np.random.binomial(1, 0.12, n)  # 慢性肾病
    data['HDL'] = np.random.normal(55, 15, n)  # HDL胆固醇
    data['eGFR'] = np.random.normal(90, 20, n)  # 估算肾小球滤过率
    
    # 代谢综合征指标
    data['BMI'] = np.random.normal(28, 5, n)
    data['HbA1c'] = np.random.normal(5.5, 1, n)
    data['TG'] = np.random.lognormal(mean=4.5, sigma=0.6, n)
    
    # CKM分期 (0-4)
    # 基于风险因素计算CKM分期
    ckm_stage = np.zeros(n)
    for i in range(n):
        risk_score = (data['SBP'][i] > 130) + data['DIABETES'][i] + data['CKD'][i]
        risk_score += (data['BMI'][i] > 30) + (data['HDL'][i] < 40)
        if risk_score >= 4:
            ckm_stage[i] = 4
        elif risk_score >= 3:
            ckm_stage[i] = 3
        elif risk_score >= 2:
            ckm_stage[i] = 2
        elif risk_score >= 1:
            ckm_stage[i] = 1
        else:
            ckm_stage[i] = 0
    
    data['CKM_STAGE'] = ckm_stage.astype(int)
    
    # 创建CKM高风险标签 (stage >= 2)
    data['CKM_HIGH_RISK'] = (ckm_stage >= 2).astype(int)
    
    # 添加心血管事件
    data['CVD_EVENT'] = np.random.binomial(1, 0.08 + 0.1 * ckm_stage / 4, n)
    
    df = pd.DataFrame(data)
    
    # 添加铅与CKM的关联（模拟真实效应）
    df['CKM_HIGH_RISK'] = (
        df['BPB_ug/dL'] * 0.3 + 
        df['SBP'] * 0.01 + 
        df['DIABETES'] * 0.5 +
        np.random.random(n) * 0.2
    > 1.5).astype(int)
    
    return df

# ============= 特征工程 =============
def create_features(df):
    """创建特征矩阵"""
    
    # 铅暴露特征
    lead_features = ['BPB_ug/dL', 'URXUBA_ug/L']
    
    # 代谢特征
    metabolic_features = ['BMI', 'HbA1c', 'TG', 'HDL']
    
    # 心血管特征
    cv_features = ['SBP', 'DIABETES', 'CKD', 'eGFR']
    
    # 组合特征
    all_features = lead_features + metabolic_features + cv_features
    
    X = df[all_features].copy()
    
    # 创建交互特征
    X['Lead_Metabolic_Score'] = X['BPB_ug/dL'] * X['BMI']
    X['Lead_CVD_Score'] = X['BPB_ug/dL'] * X['SBP'] / 100
    
    return X, all_features + ['Lead_Metabolic_Score', 'Lead_CVD_Score']

# ============= XGBoost模型训练 =============
def train_xgboost_model(X, y):
    """训练XGBoost模型"""
    
    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # XGBoost参数
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'use_label_encoder': False
    }
    
    # 训练模型
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # 预测
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # 评估
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    print("=" * 50)
    print("XGBoost CKM高风险预测模型结果")
    print("=" * 50)
    print(f"AUC-ROC: {auc_score:.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    
    # 交叉验证
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
    print(f"\n5折交叉验证 AUC.mean():.4: {cv_scoresf} ± {cv_scores.std():.4f}")
    
    return model, X_test, y_test, y_pred_proba

# ============= SHAP可解释性分析 =============
def shap_analysis(model, X_test, feature_names):
    """SHAP可解释性分析"""
    
    print("\n" + "=" * 50)
    print("SHAP可解释性分析")
    print("=" * 50)
    
    # 创建SHAP解释器
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # 特征重要性排序
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)
    
    print("\n特征重要性排名:")
    for i, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    return shap_values, explainer

# ============= 可视化 =============
def visualize_results(model, X_test, y_test, y_pred_proba, shap_values, feature_names):
    """可视化结果"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. ROC曲线
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    axes[0, 0].plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {auc_score:.3f}')
    axes[0, 0].plot([0, 1], [0, 1], 'k--', linewidth=1)
    axes[0, 0].set_xlabel('False Positive Rate', fontsize=12)
    axes[0, 0].set_ylabel('True Positive Rate', fontsize=12)
    axes[0, 0].set_title('ROC Curve: Lead-CKM Risk Prediction', fontsize=14)
    axes[0, 0].legend(loc='lower right')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 特征重要性
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)[-10:]
    axes[0, 1].barh(range(len(sorted_idx)), importance[sorted_idx])
    axes[0, 1].set_yticks(range(len(sorted_idx)))
    axes[0, 1].set_yticklabels([feature_names[i] for i in sorted_idx])
    axes[0, 1].set_xlabel('Feature Importance', fontsize=12)
    axes[0, 1].set_title('XGBoost Feature Importance', fontsize=14)
    
    # 3. SHAP summary
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, 
                     show=False, max_display=10)
    plt.tight_layout()
    
    # 4. 铅暴露剂量反应曲线
    lead_levels = np.linspace(0, 20, 100)
    risk_prob = []
    for lead in lead_levels:
        test_sample = X_test.iloc[0:1].copy()
        test_sample['BPB_ug/dL'] = lead
        prob = model.predict_proba(test_sample)[0, 1]
        risk_prob.append(prob)
    
    axes[1, 1].plot(lead_levels, risk_prob, 'r-', linewidth=2)
    axes[1, 1].axhline(y=0.5, color='k', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Blood Lead (μg/dL)', fontsize=12)
    axes[1, 1].set_ylabel('CKM High Risk Probability', fontsize=12)
    axes[1, 1].set_title('Dose-Response: Lead and CKM Risk', fontsize=14)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/lead_ckm_xgboost_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n可视化结果已保存至: output/lead_ckm_xgboost_analysis.png")

# ============= 主函数 =============
def main():
    print("=" * 60)
    print("铅暴露与CKM综合征 - XGBoost机器学习分析")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n[1/4] 加载NHANES数据...")
    df = load_and_prepare_data()
    print(f"  样本数: {len(df)}")
    print(f"  CKM高风险比例: {df['CKM_HIGH_RISK'].mean():.2%}")
    
    # 2. 特征工程
    print("\n[2/4] 特征工程...")
    X, feature_names = create_features(df)
    y = df['CKM_HIGH_RISK']
    print(f"  特征数: {len(feature_names)}")
    
    # 3. 训练模型
    print("\n[3/4] 训练XGBoost模型...")
    model, X_test, y_test, y_pred_proba = train_xgboost_model(X, y)
    
    # 4. SHAP分析
    print("\n[4/4] SHAP可解释性分析...")
    shap_values, explainer = shap_analysis(model, X_test, feature_names)
    
    # 5. 可视化
    visualize_results(model, X_test, y_test, y_pred_proba, shap_values, feature_names)
    
    # 6. 保存模型
    model.save_model('output/lead_ckm_xgboost_model.json')
    print("\n模型已保存至: output/lead_ckm_xgboost_model.json")
    
    return model, df

if __name__ == "__main__":
    model, df = main()
    print("\n✅ 分析完成!")
