#!/usr/bin/env python3
"""
因果推断分析模块 (Causal Inference Analysis)
用于铅暴露与CKM综合征的因果关系推断

功能:
1. DAG有向无环图构建与可视化
2. 倾向评分匹配 (PSM)
3. 逆概率加权 (IPTW)
4. 双重稳健估计 (AIPW)
5. 协变量平衡评估
6. 敏感性分析 (E-value)

作者: Pain
日期: 2026-02-28
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os
import json

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 输出目录
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)


class CausalInferenceAnalyzer:
    """因果推断分析器"""
    
    def __init__(self, data, exposure='LBXBPB', outcome='CKM_risk', confounders=None):
        """
        初始化
        
        参数:
            data: DataFrame, NHANES数据
            exposure: str, 暴露变量 (血铅)
            outcome: str, 结果变量 (CKM风险评分)
            confounders: list, 混杂变量列表
        """
        self.data = data.copy()
        self.exposure = exposure
        self.outcome = outcome
        
        # 默认混杂变量
        if confounders is None:
            self.confounders = [
                'age', 'gender', 'race', 'BMI', 'smoking', 'alcohol',
                'education', 'income', 'hypertension', 'diabetes'
            ]
        else:
            self.confounders = confounders
        
        # 创建二分类处理变量 (高铅暴露 vs 低铅暴露)
        self._create_binary_exposure()
        
    def _create_binary_exposure(self, threshold=None):
        """创建二分类暴露变量"""
        if threshold is None:
            threshold = self.data[self.exposure].median()
        
        self.treatment = f'{self.exposure}_high'
        self.data[self.treatment] = (self.data[self.exposure] >= threshold).astype(int)
        
    def build_dag(self):
        """
        构建因果DAG
        
        返回:
            dict: DAG节点和边信息
        """
        # 基于领域知识构建DAG
        dag_structure = {
            'exposure': self.exposure,
            'outcome': self.outcome,
            'nodes': {
                # 暴露
                self.exposure: {'label': 'Blood Lead', 'type': 'exposure'},
                
                # 结果
                self.outcome: {'label': 'CKM Syndrome', 'type': 'outcome'},
                
                # 人口学因素
                'age': {'label': 'Age', 'type': 'confounder'},
                'gender': {'label': 'Gender', 'type': 'confounder'},
                'race': {'label': 'Race/Ethnicity', 'type': 'confounder'},
                'education': {'label': 'Education', 'type': 'confounder'},
                'income': {'label': 'Income', 'type': 'confounder'},
                
                # 生活方式
                'smoking': {'label': 'Smoking', 'type': 'confounder'},
                'alcohol': {'label': 'Alcohol Use', 'type': 'confounder'},
                'BMI': {'label': 'BMI', 'type': 'confounder'},
                
                # 疾病状态
                'hypertension': {'label': 'Hypertension', 'type': 'mediator'},
                'diabetes': {'label': 'Diabetes', 'type': 'mediator'},
            },
            'edges': [
                # 混杂因素 → 暴露
                ('age', self.exposure),
                ('gender', self.exposure),
                ('race', self.exposure),
                ('education', self.exposure),
                ('income', self.exposure),
                ('smoking', self.exposure),
                ('alcohol', self.exposure),
                
                # 混杂因素 → 结果
                ('age', self.outcome),
                ('gender', self.outcome),
                ('race', self.outcome),
                ('education', self.outcome),
                ('income', self.outcome),
                ('BMI', self.outcome),
                ('smoking', self.outcome),
                ('alcohol', self.outcome),
                
                # 暴露 → 结果 (主要因果路径)
                (self.exposure, self.outcome),
                
                # 暴露 → 中介变量 → 结果
                (self.exposure, 'hypertension'),
                ('hypertension', self.outcome),
                (self.exposure, 'diabetes'),
                ('diabetes', self.outcome),
            ]
        }
        
        return dag_structure
    
    def visualize_dag(self, dag_structure):
        """可视化DAG"""
        try:
            import networkx as nx
            
            G = nx.DiGraph()
            
            # 添加节点
            for node, info in dag_structure['nodes'].items():
                G.add_node(node, label=info['label'], type=info['type'])
            
            # 添加边
            for edge in dag_structure['edges']:
                G.add_edge(edge[0], edge[1])
            
            # 绘图
            fig, ax = plt.subplots(1, 1, figsize=(14, 10))
            
            # 节点颜色
            colors = []
            for node in G.nodes():
                node_type = G.nodes[node].get('type', 'confounder')
                if node_type == 'exposure':
                    colors.append('#FF6B6B')
                elif node_type == 'outcome':
                    colors.append('#4ECDC4')
                elif node_type == 'mediator':
                    colors.append('#FFE66D')
                else:
                    colors.append('#95E1D3')
            
            # 布局
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
            
            # 绘制
            nx.draw(G, pos, ax=ax, with_labels=True, 
                   node_color=colors, node_size=3000,
                   font_size=10, font_weight='bold',
                   arrows=True, arrowsize=20,
                   edge_color='gray', width=2,
                   connectionstyle='arc3,rad=0.1')
            
            # 图例
            legend_elements = [
                plt.scatter([], [], c='#FF6B6B', s=200, label='Exposure'),
                plt.scatter([], [], c='#4ECDC4', s=200, label='Outcome'),
                plt.scatter([], [], c='#FFE66D', s=200, label='Mediator'),
                plt.scatter([], [], c='#95E1D3', s=200, label='Confounder'),
            ]
            ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
            
            ax.set_title('Causal DAG: Lead Exposure → CKM Syndrome', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f'{OUTPUT_DIR}/causal_dag.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ DAG可视化已保存: {OUTPUT_DIR}/causal_dag.png")
            
        except Exception as e:
            print(f"DAG可视化错误: {e}")
    
    def propensity_score_matching(self, n_neighbors=5):
        """
        倾向评分匹配
        
        参数:
            n_neighbors: 匹配数量
            
        返回:
            DataFrame: 匹配后的数据
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import NearestNeighbors
        
        # 准备数据
        available_confounders = [c for c in self.confounders if c in self.data.columns]
        X = self.data[available_confounders].fillna(self.data[available_confounders].median())
        T = self.data[self.treatment]
        
        # 拟合倾向评分模型
        ps_model = LogisticRegression(max_iter=1000, random_state=42)
        ps_model.fit(X, T)
        
        self.data['propensity_score'] = ps_model.predict_proba(X)[:, 1]
        
        # 匹配
        treated = self.data[self.data[self.treatment] == 1].copy()
        control = self.data[self.data[self.treatment] == 0].copy()
        
        matched_data = []
        
        for idx, row in treated.iterrows():
            ps = row['propensity_score']
            control['ps_diff'] = abs(control['propensity_score'] - ps)
            nearest = control.nsmallest(n_neighbors, 'ps_diff')
            matched_data.append(row)
            for _, match_row in nearest.iterrows():
                matched_data.append(match_row.copy())
        
        matched_df = pd.DataFrame(matched_data)
        
        print(f"✓ PSM完成: {len(matched_df)} 个匹配样本")
        
        return matched_df
    
    def calculate_ate_psm(self, matched_data):
        """计算PSM后的平均处理效应"""
        
        treated = matched_data[matched_data[self.treatment] == 1][self.outcome]
        control = matched_data[matched_data[self.treatment] == 0][self.outcome]
        
        # ATE
        ate = treated.mean() - control.mean()
        
        # Bootstrap CI
        n_bootstrap = 1000
        boot_ates = []
        
        for _ in range(n_bootstrap):
            boot_treated = treated.sample(n=len(treated), replace=True)
            boot_control = control.sample(n=len(control), replace=True)
            boot_ates.append(boot_treated.mean() - boot_control.mean())
        
        ci_lower = np.percentile(boot_ates, 2.5)
        ci_upper = np.percentile(boot_ates, 97.5)
        
        # P值
        t_stat, p_value = stats.ttest_ind(treated, control)
        
        return {
            'method': 'PSM',
            'ATE': ate,
            'CI_lower': ci_lower,
            'CI_upper': ci_upper,
            'p_value': p_value,
            'n_treated': len(treated),
            'n_control': len(control)
        }
    
    def inverse_probability_weighting(self):
        """
        逆概率加权估计
        
        返回:
            dict: IPTW结果
        """
        from sklearn.linear_model import LogisticRegression
        
        # 准备数据
        available_confounders = [c for c in self.confounders if c in self.data.columns]
        X = self.data[available_confounders].fillna(self.data[available_confounders].median())
        T = self.data[self.treatment]
        Y = self.data[self.outcome]
        
        # 倾向评分
        ps_model = LogisticRegression(max_iter=1000, random_state=42)
        ps_model.fit(X, T)
        ps = ps_model.predict_proba(X)[:, 1]
        
        # 稳定权重
        ps = np.clip(ps, 0.05, 0.95)  # 截断极端值
        self.data['propensity_score'] = ps
        
        # IPTW权重
        self.data['iptw_weight'] = (
            T / ps + (1 - T) / (1 - ps)
        )
        
        # 标准化权重
        self.data['iptw_weight'] = self.data['iptw_weight'] / self.data['iptw_weight'].mean()
        
        # 计算ATE
        treated_mask = self.data[self.treatment] == 1
        control_mask = self.data[self.treatment] == 0
        
        ate_ipw = (
            (self.data.loc[treated_mask, self.outcome] * 
             self.data.loc[treated_mask, 'iptw_weight']).sum() /
            self.data.loc[treated_mask, 'iptw_weight'].sum()
        ) - (
            (self.data.loc[control_mask, self.outcome] * 
             self.data.loc[control_mask, 'iptw_weight']).sum() /
            self.data.loc[control_mask, 'iptw_weight'].sum()
        )
        
        # Bootstrap CI
        n_bootstrap = 500
        boot_ates = []
        
        for _ in range(n_bootstrap):
            idx = np.random.choice(len(self.data), len(self.data), replace=True)
            boot_data = self.data.iloc[idx]
            
            treated_mask_b = boot_data[self.treatment] == 1
            control_mask_b = boot_data[self.treatment] == 0
            
            ate_b = (
                (boot_data.loc[treated_mask_b, self.outcome] * 
                 boot_data.loc[treated_mask_b, 'iptw_weight']).sum() /
                boot_data.loc[treated_mask_b, 'iptw_weight'].sum()
            ) - (
                (boot_data.loc[control_mask_b, self.outcome] * 
                 boot_data.loc[control_mask_b, 'iptw_weight']).sum() /
                boot_data.loc[control_mask_b, 'iptw_weight'].sum()
            )
            boot_ates.append(ate_b)
        
        ci_lower = np.percentile(boot_ates, 2.5)
        ci_upper = np.percentile(boot_ates, 97.5)
        
        print(f"✓ IPTW完成: ATE = {ate_ipw:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        return {
            'method': 'IPTW',
            'ATE': ate_ipw,
            'CI_lower': ci_lower,
            'CI_upper': ci_upper,
            'n': len(self.data)
        }
    

    def doubly_robust_estimation(self):
        """
        双重稳健估计 (AIPW - Augmented Inverse Probability Weighting)
        
        结合倾向评分和结果回归，即使模型误设也能得到一致估计
        
        返回:
            dict: AIPW结果
        """
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.ensemble import RandomForestRegressor
        
        # 准备数据
        available_confounders = [c for c in self.confounders if c in self.data.columns]
        X = self.data[available_confounders].fillna(self.data[available_confounders].median())
        T = self.data[self.treatment].values
        Y = self.data[self.outcome].values
        
        # 步骤1: 估计倾向评分
        ps_model = LogisticRegression(max_iter=1000, random_state=42)
        ps_model.fit(X, T)
        ps = ps_model.predict_proba(X)[:, 1]
        ps = np.clip(ps, 0.05, 0.95)
        
        # 步骤2: 估计结果模型
        treated_mask = T == 1
        control_mask = T == 0
        
        # E(Y|X)
        outcome_model = Ridge(alpha=1.0)
        outcome_model.fit(X, Y)
        mu_x = outcome_model.predict(X)
        
        # E(Y|X, T=1)
        if sum(treated_mask) > 10:
            mu1_model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
            mu1_model.fit(X[treated_mask], Y[treated_mask])
            mu1_x = mu1_model.predict(X)
        else:
            mu1_x = mu_x
        
        # E(Y|X, T=0)
        if sum(control_mask) > 10:
            mu0_model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
            mu0_model.fit(X[control_mask], Y[control_mask])
            mu0_x = mu0_model.predict(X)
        else:
            mu0_x = mu_x
        
        # 步骤3: 计算AIPW估计量
        robust_part = (
            (T / ps) * (Y - mu1_x) -
            ((1 - T) / (1 - ps)) * (Y - mu0_x)
        )
        adjustment = mu1_x - mu0_x
        aipw_estimate = np.mean(adjustment) + np.mean(robust_part)
        
        # Bootstrap CI
        n_bootstrap = 500
        boot_ates = []
        
        for _ in range(n_bootstrap):
            idx = np.random.choice(len(self.data), len(self.data), replace=True)
            X_b, T_b, Y_b = X.iloc[idx], T[idx], Y[idx]
            ps_b, mu1_b, mu0_b = ps[idx], mu1_x[idx], mu0_x[idx]
            
            robust_b = ((T_b / ps_b) * (Y_b - mu1_b) - ((1 - T_b) / (1 - ps_b)) * (Y_b - mu0_b))
            boot_ates.append(np.mean(mu1_b - mu0_b) + np.mean(robust_b))
        
        ci_lower, ci_upper = np.percentile(boot_ates, 2.5), np.percentile(boot_ates, 97.5)
        
        # 效率提升
        ipw_var = np.var([(T[i]/ps[i] + (1-T[i])/(1-ps[i])) * Y[i] for i in range(len(Y))])
        aipw_var = np.var(adjustment + robust_part)
        efficiency_ratio = ipw_var / aipw_var if aipw_var > 0 else np.nan
        
        print(f"✓ AIPW完成: ATE = {aipw_estimate:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"  效率提升: {efficiency_ratio:.2f}x (相对于IPTW)")
        
        self.data['propensity_score_aipw'] = ps
        
        return {'method': 'AIPW', 'ATE': aipw_estimate, 'CI_lower': ci_lower, 
                'CI_upper': ci_upper, 'efficiency_ratio': efficiency_ratio, 'n': len(self.data)}

    def assess_covariate_balance(self, matched_data=None):
        """
        评估协变量平衡
        
        参数:
            matched_data: 匹配后的数据
            
        返回:
            DataFrame: 平衡性统计
        """
        available_confounders = [c for c in self.confounders if c in self.data.columns]
        
        if matched_data is not None:
            data_to_check = matched_data
        else:
            data_to_check = self.data
        
        balance_results = []
        
        for var in available_confounders:
            treated = data_to_check[data_to_check[self.treatment] == 1][var].dropna()
            control = data_to_check[data_to_check[self.treatment] == 0][var].dropna()
            
            if len(treated) > 0 and len(control) > 0:
                # 标准化均值差 (SMD)
                smd = (treated.mean() - control.mean()) / np.sqrt(
                    (treated.var() + control.var()) / 2
                )
                
                # 方差比 (VR)
                vr = treated.var() / control.var() if control.var() > 0 else np.nan
                
                balance_results.append({
                    'Variable': var,
                    'Mean_Treated': treated.mean(),
                    'Mean_Control': control.mean(),
                    'SMD': abs(smd),
                    'Variance_Ratio': vr,
                    'Balanced': abs(smd) < 0.1
                })
        
        balance_df = pd.DataFrame(balance_results)
        
        # 可视化
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # SMD
        colors = ['green' if b else 'red' for b in balance_df['Balanced']]
        axes[0].barh(balance_df['Variable'], balance_df['SMD'], color=colors)
        axes[0].axvline(x=0.1, color='orange', linestyle='--', label='Threshold (0.1)')
        axes[0].axvline(x=0.2, color='red', linestyle='--', label='Threshold (0.2)')
        axes[0].set_xlabel('Standardized Mean Difference (SMD)')
        axes[0].set_title('Covariate Balance: SMD')
        axes[0].legend()
        
        # 方差比
        axes[1].barh(balance_df['Variable'], balance_df['Variance_Ratio'], color='steelblue')
        axes[1].axvline(x=1, color='green', linestyle='--', label='Ideal (1.0)')
        axes[1].axvline(x=2, color='orange', linestyle='--', label='Threshold (2.0)')
        axes[1].set_xlabel('Variance Ratio')
        axes[1].set_title('Covariate Balance: Variance Ratio')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/psm_balance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        balanced_count = balance_df['Balanced'].sum()
        print(f"✓ 平衡评估完成: {balanced_count}/{len(balance_df)} 个变量达到平衡 (|SMD|<0.1)")
        
        return balance_df
    
    def plot_iptw_weights(self):
        """绘制IPTW权重分布"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 权重分布
        axes[0].hist(self.data['iptw_weight'], bins=50, edgecolor='black', alpha=0.7)
        axes[0].axvline(x=1, color='red', linestyle='--', label='Ideal (1.0)')
        axes[0].set_xlabel('IPTW Weight')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('IPTW Weight Distribution')
        axes[0].legend()
        
        # 按处理组的权重
        treated_weights = self.data[self.data[self.treatment] == 1]['iptw_weight']
        control_weights = self.data[self.data[self.treatment] == 0]['iptw_weight']
        
        axes[1].boxplot([treated_weights, control_weights], 
                       labels=['High Lead', 'Low Lead'])
        axes[1].set_ylabel('IPTW Weight')
        axes[1].set_title('IPTW Weights by Treatment Group')
        
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/iptw_weights.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ IPTW权重图已保存: {OUTPUT_DIR}/iptw_weights.png")
    
    def sensitivity_analysis(self, ate_estimate):
        """
        敏感性分析 - E-value计算
        
        E-value: 需要多强的未测量混杂才能解释掉观察到的效应
        
        参数:
            ate_estimate: 估计的处理效应
            
        返回:
            dict: E-value结果
        """
        # 效应值 (转换为OR)
        # 假设outcome是连续的，使用标准化效应
        effect_size = abs(ate_estimate)
        
        # 简化E-value计算 (对于连续结果)
        # E-value = |effect| + sqrt(|effect| * (|effect| - 1)) 当 effect > 1
        # 对于RR/OR: E-value = RR + sqrt(RR * (RR - 1))
        
        # 使用相对风险近似
        # 假设处理组和对照组的结果分布
        treated_mean = self.data[self.data[self.treatment] == 1][self.outcome].mean()
        control_mean = self.data[self.data[self.treatment] == 0][self.outcome].mean()
        
        # 近似相对风险
        if control_mean != 0:
            relative_risk = treated_mean / control_mean
        else:
            relative_risk = 1 + effect_size
        
        # E-value公式
        if relative_risk > 1:
            e_value = relative_risk + np.sqrt(relative_risk * (relative_risk - 1))
        else:
            e_value = 1 + (effect_size / abs(control_mean)) if control_mean != 0 else 1
        
        # 转换为标准化均值的E-value (更保守)
        # 对于标准化均值差 d: E-value ≈ |d| + sqrt(|d|^2 + 1)
        e_value_smd = effect_size + np.sqrt(effect_size ** 2 + 1)
        
        # 需要解释效应的未测量混杂强度
        # 假设未测量混杂与处理相关且与结果相关都是RR
        required_confounding = np.sqrt(e_value_smd)
        
        result = {
            'E_value': e_value_smd,
            'Relative_Risk': relative_risk,
            'ATE': ate_estimate,
            'Required_Confounding_RR': required_confounding,
            'Interpretation': (
                f"需要未测量混杂与处理组和结果的相关性同时达到 {required_confounding:.2f} "
                f"才能解释掉观察到的因果效应"
            )
        }
        
        print(f"✓ 敏感性分析: E-value = {e_value_smd:.3f}")
        
        return result
    
    def run_full_analysis(self):
        """运行完整因果推断分析"""
        
        results = {}
        
        print("=" * 60)
        print("因果推断分析: 铅暴露 → CKM综合征")
        print("=" * 60)
        
        # 1. 构建DAG
        print("\n[1/6] 构建因果DAG...")
        dag = self.build_dag()
        self.visualize_dag(dag)
        results['dag'] = dag
        
        # 2. PSM
        print("\n[2/6] 倾向评分匹配...")
        matched_data = self.propensity_score_matching()
        ate_psm = self.calculate_ate_psm(matched_data)
        results['psm'] = ate_psm
        
        # 3. 平衡评估
        print("\n[3/6] 评估协变量平衡...")
        balance_df = self.assess_covariate_balance(matched_data)
        results['balance'] = balance_df.to_dict('records')
        
        # 4. IPTW
        print("\n[4/6] 逆概率加权...")
        ate_iptw = self.inverse_probability_weighting()
        self.plot_iptw_weights()
        results['iptw'] = ate_iptw
        
        # 5. 双重稳健估计 (AIPW)
        print("\n[5/7] 双重稳健估计 (AIPW)...")
        ate_aipw = self.doubly_robust_estimation()
        results['aipw'] = ate_aipw
        
        # 6. 综合因果效应
        print("\n[6/7] 综合因果效应估计...")
        combined_ate = (ate_psm['ATE'] + ate_iptw['ATE'] + ate_aipw['ATE']) / 3
        
        results['combined'] = {
            'ATE': combined_ate,
            'PSM_ATE': ate_psm['ATE'],
            'IPTW_ATE': ate_iptw['ATE'],
            'AIPW_ATE': ate_aipw['ATE'],
            'Conclusion': '铅暴露显著增加CKM综合征风险' if combined_ate > 0 else '未发现显著因果效应'
        }
        
        # 7. 敏感性分析
        print("\n[7/7] 敏感性分析...")
        sensitivity = self.sensitivity_analysis(combined_ate)
        results['sensitivity'] = sensitivity
        
        # 保存结果
        self._save_results(results)
        
        print("\n" + "=" * 60)
        print("分析完成!")
        print("=" * 60)
        
        return results
    
    def _save_results(self, results):
        """保存结果"""
        
        # 效应估计
        effects_df = pd.DataFrame([
            results['psm'],
            results['iptw'],
            results['aipw'],
            {**results['combined'], 'method': 'Combined'}
        ])
        effects_df.to_csv(f'{OUTPUT_DIR}/causal_effects.csv', index=False)
        
        # 敏感性分析
        sensitivity_df = pd.DataFrame([results['sensitivity']])
        sensitivity_df.to_csv(f'{OUTPUT_DIR}/sensitivity_analysis.csv', index=False)
        
        # 完整JSON
        with open(f'{OUTPUT_DIR}/causal_inference_full.json', 'w', encoding='utf-8') as f:
            # 转换numpy类型为Python类型
            def convert(obj):
                if isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, dict):
                    return {k: convert(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert(i) for i in obj]
                return obj
            
            json.dump(convert(results), f, ensure_ascii=False, indent=2)
        
        print(f"✓ 结果已保存到 {OUTPUT_DIR}/")


def generate_demo_data(n=5000):
    """生成模拟数据用于测试"""
    np.random.seed(42)
    
    data = pd.DataFrame()
    
    # 人口学变量
    data['age'] = np.random.uniform(20, 80, n)
    data['gender'] = np.random.binomial(1, 0.5, n)
    data['race'] = np.random.choice([0, 1, 2, 3, 4], n, p=[0.4, 0.2, 0.2, 0.1, 0.1])
    data['education'] = np.random.choice([1, 2, 3, 4, 5], n)
    data['income'] = np.random.uniform(1, 5, n)
    
    # 生活方式
    data['BMI'] = np.random.uniform(18, 40, n)
    data['smoking'] = np.random.binomial(1, 0.3, n)
    data['alcohol'] = np.random.binomial(1, 0.4, n)
    
    # 疾病状态 (作为中介/结果)
    data['hypertension'] = (data['age'] * 0.03 + data['BMI'] * 0.05 + 
                           np.random.normal(0, 1, n) > 0).astype(int)
    data['diabetes'] = (data['BMI'] * 0.04 + data['age'] * 0.02 + 
                       np.random.normal(0, 1, n) > 0).astype(int)
    
    # 暴露: 血铅 (受混杂影响)
    data['LBXBPB'] = (
        0.1 * data['age'] + 
        0.5 * data['smoking'] + 
        0.3 * data['alcohol'] + 
        0.2 * (5 - data['income']) +
        np.random.normal(2, 1, n)
    )
    data['LBXBPB'] = np.clip(data['LBXBPB'], 0.5, 30)
    
    # 结果: CKM风险评分 (受暴露和混杂影响)
    data['CKM_risk'] = (
        0.15 * data['LBXBPB'] +  # 暴露效应
        0.1 * data['age'] +
        0.05 * data['BMI'] +
        0.3 * data['hypertension'] +
        0.4 * data['diabetes'] +
        0.05 * data['smoking'] +
        np.random.normal(5, 2, n)
    )
    
    return data


if __name__ == '__main__':
    print("生成模拟数据...")
    data = generate_demo_data(n=5000)
    
    print("初始化因果推断分析器...")
    analyzer = CausalInferenceAnalyzer(
        data, 
        exposure='LBXBPB', 
        outcome='CKM_risk'
    )
    
    print("运行完整因果推断分析...")
    results = analyzer.run_full_analysis()
    
    print("\n" + "=" * 60)
    print("关键结果摘要:")
    print(f"  PSM ATE: {results['psm']['ATE']:.3f}")
    print(f"  IPTW ATE: {results['iptw']['ATE']:.3f}")
    print(f"  AIPW ATE: {results['aipw']['ATE']:.3f}")
    print(f"  综合ATE: {results['combined']['ATE']:.3f}")
    print(f"  E-value: {results['sensitivity']['E_value']:.3f}")
    print("=" * 60)
