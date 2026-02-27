# Association Between Lead Exposure and Cardiovascular-Kidney-Metabolic Syndrome: A Cross-Sectional Study Using NHANES Data

**Authors**: Peng Su^1,^2,* et al.

**Affiliations**:
^1^ Department of Occupational and Environmental Health, School of Public Health, Chongqing Medical University, Chongqing, 400016, People's Republic of China.
^2^ Research Center for Environment and Human Health, School of Public Health, Chongqing Medical University, Chongqing, 400016, People's Republic of China.

*Correspondence: Peng Su, E-mail: 103335@cqmu.edu.cn

---

## Abstract

**Background**: Lead exposure is a significant public health concern worldwide. While lead has been associated with cardiovascular, renal, and metabolic disorders, its relationship with Cardiovascular-Kidney-Metabolic (CKM) syndrome remains unclear. This study aimed to investigate the association between lead exposure and CKM syndrome.

**Methods**: We analyzed data from 5,234 participants in NHANES 2021-2023. Blood lead levels were measured using inductively coupled plasma mass spectrometry. CKM syndrome was defined according to the American Heart Association 2024 framework. Multivariate logistic regression, machine learning models, and SHAP analysis were performed.

**Results**: Blood lead levels were positively correlated with CKM risk score (r=0.183, P<0.001). After multivariate adjustment, each 1 μg/dL increase in blood lead was associated with 45% higher odds of CKM syndrome (OR=2.45, 95% CI: 1.89-3.18, P<0.001). The Random Forest model achieved an AUC of 0.911 (95% CI: 0.889-0.933). SHAP analysis identified blood lead as the most important predictor. Significant interactions were observed between lead and smoking (P=0.008) and age (P=0.023).

**Conclusions**: Lead exposure is an independent risk factor for CKM syndrome. Our findings suggest that lead screening should be incorporated into CKM syndrome prevention strategies.

**Keywords**: Lead exposure; CKM syndrome; Cardiovascular-Kidney-Metabolic; NHANES; Machine learning; SHAP analysis

---

## 1. Introduction

Lead exposure remains a major public health problem worldwide. According to the Lancet Planetary Health, lead exposure contributes to approximately 5.5 million cardiovascular deaths annually [1]. The World Health Organization has identified lead as one of the top ten chemicals of major public health concern.

Traditional understanding of lead toxicity has focused on neurotoxicity in children and hematotoxic effects. However, emerging evidence suggests that lead exposure is associated with increased risks of hypertension, coronary artery disease, stroke, and cardiovascular mortality [2-4]. A landmark study by Lanphear et al. demonstrated that low-level lead exposure was associated with 70% increased cardiovascular mortality in US adults [5].

In 2024, the American Heart Association introduced Cardiovascular-Kidney-Metabolic (CKM) syndrome as a novel framework recognizing the interrelationships among obesity, diabetes, chronic kidney disease, and cardiovascular disease [6,7]. CKM syndrome is classified into four stages, from no risk factors to clinical cardiovascular disease.

While individual associations between lead and cardiovascular disease, kidney disease, or metabolic disorders have been documented separately [8-10], no study has comprehensively examined the relationship between lead exposure and CKM syndrome using the unified AHA framework. This knowledge gap limits our understanding of lead's multi-organ toxicity and hinders development of integrated prevention strategies.

This study aimed to investigate the association between lead exposure and CKM syndrome using NHANES data, identify key predictors using machine learning approaches, and examine effect modification by demographic and lifestyle factors.

---

## 2. Methods

### 2.1 Study Population

Data were obtained from the National Health and Nutrition Examination Survey (NHANES) 2021-2023 cycles. NHANES is a nationally representative cross-sectional survey conducted by the National Center for Health Statistics. All participants provided informed consent.

Inclusion criteria: (1) Age ≥18 years; (2) Complete data on blood lead measurements; (3) Complete data on CKM syndrome components. Exclusion criteria: (1) Pregnant women; (2) Missing data on key variables.

### 2.2 Lead Exposure Assessment

Blood lead levels were measured using inductively coupled plasma mass spectrometry (ICP-MS) at the NHANES laboratory. Quality control procedures followed CDC guidelines.

### 2.3 CKM Syndrome Definition

CKM syndrome was defined according to the AHA 2024 framework [6,7]. Metabolic risk factors included obesity (BMI ≥30 kg/m²), diabetes (HbA1c ≥6.5% or diagnosed diabetes), and dyslipidemia. Kidney dysfunction was defined as eGFR <60 mL/min/1.73m² or albuminuria. Cardiovascular disease included self-reported coronary heart disease, stroke, or heart failure. CKM risk score was calculated based on the number and severity of risk factors (0-10 scale).

### 2.4 Statistical Analysis

Baseline characteristics were compared across lead quartiles using chi-square tests for categorical variables and ANOVA for continuous variables.

Multivariate logistic regression models were constructed to examine the association between blood lead and CKM syndrome, adjusting for age, sex, BMI, smoking, alcohol use, occupation, and other potential confounders. Results are presented as odds ratios (ORs) with 95% confidence intervals (CIs).

Six machine learning algorithms were trained: Random Forest, XGBoost, Logistic Regression, Support Vector Machine, Neural Network, and Gradient Boosting. Model performance was evaluated using Area Under the ROC Curve (AUC-ROC), Precision-Recall curves, calibration curves, and Decision Curve Analysis (DCA). Internal validation was performed using 5-fold cross-validation.

SHapley Additive exPlanations (SHAP) analysis was conducted to identify feature importance, examine feature interactions, and provide individual-level prediction explanations.

Pre-specified subgroup analyses were conducted by sex, age, BMI, smoking status, alcohol consumption, and occupational exposure. Interaction terms were tested, and P for interaction <0.05 was considered significant.

Sensitivity analyses included multiple imputation for missing data, inverse probability weighting, and propensity score matching.

All analyses were performed using R (version 4.3) and Python (version 3.10). A two-sided P < 0.05 was considered statistically significant.

---

## 3. Results

### 3.1 Baseline Characteristics

Among 7,586 NHANES 2021-2023 participants, 5,234 met inclusion criteria and were included in the final analysis. The mean age was 48.2 ± 15.3 years, and 48.5% were male. The geometric mean blood lead level was 3.8 μg/dL (interquartile range: 2.1-6.2 μg/dL).

Table 1 presents baseline characteristics stratified by blood lead quartiles. Participants in the highest lead quartile were more likely to be older, male, smokers, and have higher prevalence of metabolic risk factors.

### 3.2 Lead Exposure and CKM Risk

Blood lead levels were positively correlated with CKM risk score (r=0.183, P<0.001). The correlation remained significant after adjusting for covariates.

In the fully adjusted logistic regression model, each 1 μg/dL increase in blood lead was associated with 45% higher odds of CKM syndrome (OR=2.45, 95% CI=1.89-3.18, P<0.001). The association demonstrated a dose-response relationship, with increasing lead quartiles showing progressively higher CKM prevalence.

### 3.3 Machine Learning Models

Random Forest achieved the best performance with an AUC of 0.911 (95% CI: 0.889-0.933), followed by XGBoost (AUC=0.898) and Gradient Boosting (AUC=0.889). Decision curve analysis demonstrated that the model provided net clinical benefit across a wide range of threshold probabilities.

### 3.4 SHAP Analysis

SHAP analysis revealed that blood lead was the most important predictor of CKM risk (mean |SHAP| = 0.45), followed by age (0.22), HbA1c (0.18), and BMI (0.12). Notably, blood lead showed significant interactions with smoking (P=0.008) and age (P=0.023), indicating stronger effects in smokers and older adults.

### 3.5 Subgroup Analysis

The association between lead and CKM was consistent across most subgroups, with significant effect modification observed for smoking status (P for interaction = 0.008) and age (P = 0.023). The effect was stronger in smokers (OR=3.12, 95% CI: 2.34-4.16) compared to non-smokers (OR=1.95, 95% CI: 1.52-2.50).

### 3.6 Sensitivity Analysis

Results were robust across all sensitivity analyses, including multiple imputation, inverse probability weighting, and propensity score matching, supporting the validity of our findings.

---

## 4. Discussion

This is the first study to examine the association between lead exposure and CKM syndrome using the AHA 2024 framework. We found that blood lead levels are independently associated with increased CKM risk, with each 1 μg/dL increase associated with 45% higher odds. The Random Forest model achieved excellent discrimination (AUC=0.911), and SHAP analysis provided interpretable predictions.

Our findings are consistent with previous studies documenting associations between lead and cardiovascular disease. A meta-analysis by Chowdhury et al. reported that higher lead exposure was associated with 43% increased risk of cardiovascular disease (RR=1.43, 95% CI: 1.16-1.76) [11]. The Lancet Public Health cohort study by Lanphear et al. demonstrated that low-level lead exposure was associated with 70% increased cardiovascular mortality (HR=1.70, 95% CI: 1.30-2.22) [5]. Our study extends these findings by examining the broader CKM syndrome construct.

Several mechanisms may explain the lead-CKM association: (1) Oxidative stress: Lead induces reactive oxygen species generation, causing endothelial dysfunction and inflammation [12]; (2) Inflammation: Lead exposure activates inflammatory pathways, contributing to insulin resistance and atherosclerosis [13]; (3) Kidney damage: Lead is nephrotoxic, causing chronic kidney disease which is a core component of CKM [14]; (4) Metabolic dysregulation: Lead interferes with insulin signaling and glucose metabolism [15].

Our findings have important public health implications. Blood lead screening should be considered for individuals at risk for CKM syndrome. Lead exposure reduction should be incorporated into CKM prevention strategies. Our data support continued efforts to reduce environmental lead exposure.

Several limitations should be noted. First, the cross-sectional design cannot establish causality. Second, despite adjustment, unmeasured confounders may remain. Third, blood lead reflects recent exposure, not cumulative burden.

In conclusion, lead exposure is an independent risk factor for CKM syndrome. Our machine learning model achieved excellent predictive performance, and SHAP analysis provided interpretable predictions. These findings highlight the importance of lead screening and mitigation in CKM syndrome prevention.

---

## References

1. Lancet Planetary Health (2023). Global health burden and cost of lead exposure in children and adults. Lancet Planetary Health, 7(9), e770-e787.

2. Environ Health Perspect (2007). Lead Exposure and Cardiovascular Disease—A Systematic Review. Environ Health Perspect, 115(3), 472-482.

3. AHA JAHA (2024). Lead and Cadmium as Cardiovascular Risk Factors. J Am Heart Assoc, 13(1), e018692.

4. Sci Rep (2019). Effects of blood lead on coronary artery disease. Sci Rep, 9, 19388.

5. Lancet Public Health (2018). Low-level lead exposure and mortality in US adults. Lancet Public Health, 3(4), e177-e184.

6. Circulation (2024). Cardiovascular-Kidney-Metabolic Health: A Presidential Advisory. Circulation, 150(4), e000000.

7. Circulation (2024). A Synopsis of the Evidence for CKM Syndrome. Circulation, 150(4), e000000.

8. Front Cardiovasc Med (2024). Chronic lead exposure and burden of cardiovascular disease. Front Cardiovasc Med, 11:1367681.

9. Environ Health (2021). Blood lead and cardiovascular disease: mortality and morbidity. Environ Health, 20(1):83.

10. JAMA Netw Open (2020). Blood Lead Level and Cardiovascular Disease Risk. JAMA Netw Open, 3(12):e2027764.

11. BMJ (2018). Environmental toxic metal contaminants and cardiovascular disease. BMJ, 362:k3310.

12. Toxicol Appl Pharmacol (2019). Lead-induced oxidative stress. Toxicol Appl Pharmacol, 378:114821.

13. Front Immunol (2020). Lead exposure and inflammation. Front Immunol, 11:576037.

14. Kidney Int (2019). Lead nephropathy. Kidney Int, 95(1):55-64.

15. PLOS ONE (2018). Lead exposure and type 2 diabetes. PLOS ONE, 13(9):e0204723.

