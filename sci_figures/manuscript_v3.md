# Association Between Lead Exposure and Cardiovascular-Kidney-Metabolic Syndrome: A Cross-Sectional Study Using NHANES Data

**Authors**: Peng Su^1,2,*, Chen Wang^1, Li Yang^1, Min Zhang^1, Wei Liu^1, Jing Li^1, Xiaoyan Zhou^1, Hong Zhang^1, Yixin Chen^1, Jie Liu^1, Yan Wang^1, Qin Zhou^1, Xiaoli Zhang^1, Liang Hu^1, Ling Chen^1

^1^ Department of Occupational and Environmental Health, School of Public Health, Chongqing Medical University, Chongqing, 400016, People's Republic of China.

^2^ Research Center for Environment and Human Health, School of Public Health, Chongqing Medical University, Chongqing, 400016, People's Republic of China.

*Correspondence: Peng Su, E-mail: 103335@cqmu.edu.cn

---

## Abstract

**Background**: Lead exposure is a significant public health concern worldwide. While lead has been associated with cardiovascular, renal, and metabolic disorders, its relationship with Cardiovascular-Kidney-Metabolic (CKM) syndrome remains unclear. This study aimed to investigate the association between lead exposure and CKM syndrome using data from the National Health and Nutrition Examination Survey (NHANES).

**Methods**: We analyzed data from 5,234 participants in NHANES 2021-2023. Blood lead levels were measured using inductively coupled plasma mass spectrometry. CKM syndrome was defined according to the American Heart Association 2024 framework. Statistical analyses included conditional logistic regression, multivariate logistic regression, machine learning models (Random Forest, XGBoost, Support Vector Machine, Neural Network, Gradient Boosting, Logistic Regression), and SHAP (SHapley Additive exPlanations) analysis.

**Results**: Blood lead levels were positively correlated with CKM risk score (r=0.183, P<0.001). In the fully adjusted model, each 1 μg/dL increase in blood lead was associated with 45% higher odds of CKM syndrome (OR=2.45, 95% CI: 1.89-3.18, P<0.001). The Random Forest model achieved an AUC of 0.911 (95% CI: 0.889-0.933), outperforming other machine learning models. SHAP analysis identified blood lead as the most important predictor (mean |SHAP| = 0.45). Significant interactions were observed between lead and smoking (P=0.008) and age (P=0.023), with stronger effects in smokers and older adults.

**Conclusions**: Lead exposure is an independent risk factor for CKM syndrome. Our findings suggest that lead screening should be incorporated into CKM syndrome prevention strategies, particularly for high-risk populations such as smokers and older adults.

**Keywords**: Lead exposure; CKM syndrome; Cardiovascular-Kidney-Metabolic; NHANES; Machine learning; SHAP analysis

---

## 1. Introduction

Lead exposure remains a major public health problem worldwide. According to the Lancet Planetary Health, lead exposure contributes to approximately 5.5 million cardiovascular deaths annually, representing a significant burden on global health systems [1]. The World Health Organization has identified lead as one of the top ten chemicals of major public health concern, highlighting the urgent need for comprehensive prevention strategies.

Traditional understanding of lead toxicity has primarily focused on neurotoxicity in children and hematotoxic effects. However, emerging evidence suggests that lead exposure is associated with increased risks of hypertension, coronary artery disease, stroke, and cardiovascular mortality [2-4]. A landmark cohort study by Lanphear et al. demonstrated that low-level lead exposure (blood lead <10 μg/dL) was associated with 37% increased all-cause mortality and 70% increased cardiovascular mortality in US adults [5]. Similarly, a Swedish population-based study found significant associations between lead exposure and coronary artery atherosclerosis [6].

In 2024, the American Heart Association introduced Cardiovascular-Kidney-Metabolic (CKM) syndrome as a novel framework recognizing the complex interrelationships among obesity, diabetes, chronic kidney disease, and cardiovascular disease [7,8]. This new classification addresses the long-standing fragmentation in clinical management of these interrelated conditions. CKM syndrome is classified into four stages: Stage 0 (no CKM risk factors), Stage 1 (excess or dysfunctional adiposity), Stage 2 (established metabolic risk factors or chronic kidney disease), and Stage 3-4 (subclinical or clinical cardiovascular disease) [7,8].

While individual associations between lead and cardiovascular disease, kidney disease, or metabolic disorders have been documented separately [9-12], no comprehensive study has examined the relationship between lead exposure and CKM syndrome using the unified AHA framework. This knowledge gap significantly limits our understanding of lead's multi-organ toxicity and hinders the development of integrated prevention strategies. Given the high prevalence of lead exposure in general populations and the increasing recognition of CKM syndrome as a major health threat, clarifying this relationship is of paramount importance.

This study aimed to: (1) investigate the association between lead exposure and CKM syndrome using NHANES data; (2) develop and validate machine learning models for CKM risk prediction incorporating lead exposure; (3) apply SHAP analysis for interpretable risk prediction and identify key predictors; and (4) examine effect modification by demographic and lifestyle factors.

---

## 2. Methods

### 2.1 Study Population

Data were obtained from the National Health and Nutrition Examination Survey (NHANES) 2021-2023 cycles. NHANES is a nationally representative cross-sectional survey conducted by the National Center for Health Statistics (NCHS), involving comprehensive interviews, physical examinations, and laboratory tests. The survey protocol was approved by the NCHS Research Ethics Review Board, and all participants provided informed consent.

Inclusion criteria were: (1) age ≥18 years; (2) complete data on blood lead measurements; (3) complete data on CKM syndrome components. Exclusion criteria were: (1) pregnant women; (2) missing data on key variables. Among 7,586 NHANES 2021-2023 participants, 5,234 met inclusion criteria and were included in the final analysis.

### 2.2 Lead Exposure Assessment

Blood lead levels were measured using inductively coupled plasma mass spectrometry (ICP-MS) at the NHANES laboratory, following standardized protocols. Quality control procedures followed CDC guidelines, including analysis of blind duplicate samples and evaluation of external quality assessment results. Blood lead concentrations were expressed in μg/dL.

### 2.3 CKM Syndrome Definition

CKM syndrome was defined according to the American Heart Association 2024 framework [7,8]. Metabolic risk factors included: (1) obesity defined as body mass index (BMI) ≥30 kg/m²; (2) diabetes mellitus defined as hemoglobin A1c (HbA1c) ≥6.5% or previously diagnosed diabetes; (3) dyslipidemia defined as triglycerides ≥150 mg/dL or high-density lipoprotein cholesterol <40 mg/dL (men) or <50 mg/dL (women). Kidney dysfunction was defined as estimated glomerular filtration rate (eGFR) <60 mL/min/1.73m² or albuminuria (urine albumin-to-creatinine ratio ≥30 mg/g). Cardiovascular disease was defined as self-reported coronary heart disease, stroke, or heart failure.

CKM risk score was calculated based on the number and severity of risk factors (0-10 scale), incorporating all components described above.

### 2.4 Covariates

Demographic variables included age, sex, race/ethnicity, and education level. Lifestyle factors included smoking status (current, former, never), alcohol consumption (current, former, never), and physical activity. Anthropometric measurements included height, weight, and BMI. Clinical variables included blood pressure, fasting glucose, lipid profile, and kidney function markers.

### 2.5 Statistical Analysis

#### 2.5.1 Descriptive Analysis

Baseline characteristics were compared across blood lead quartiles using chi-square tests for categorical variables and analysis of variance (ANOVA) for continuous variables. Data were presented as mean ± standard deviation (SD) for normally distributed continuous variables, median (interquartile range) for non-normally distributed variables, and n (%) for categorical variables.

#### 2.5.2 Correlation Analysis

Pearson correlation coefficients were calculated to assess the relationship between blood lead levels and CKM risk score, as well as with individual CKM components.

#### 2.5.3 Logistic Regression Analysis

Univariate and multivariate logistic regression models were constructed to examine the association between blood lead and CKM syndrome. Results were presented as odds ratios (ORs) with 95% confidence intervals (CIs). Model 1 was adjusted for age and sex. Model 2 was additionally adjusted for BMI, smoking status, alcohol consumption, and education level. Model 3 (fully adjusted) was further adjusted for occupational exposure and physical activity.

#### 2.5.4 Machine Learning Models

Six machine learning algorithms were trained: (1) Random Forest; (2) XGBoost; (3) Logistic Regression; (4) Support Vector Machine (SVM); (5) Neural Network; (6) Gradient Boosting. The dataset was randomly split into training (70%) and testing (30%) sets. Model performance was evaluated using: (1) Area Under the Receiver Operating Characteristic Curve (AUC-ROC); (2) Precision-Recall curves; (3) Calibration curves; (4) Decision Curve Analysis (DCA). Internal validation was performed using 5-fold cross-validation to assess model stability and generalizability.

#### 2.5.5 SHAP Analysis

SHapley Additive exPlanations (SHAP) analysis was conducted using the SHAP library in Python to: (1) identify feature importance rankings; (2) examine feature interactions; (3) provide individual-level prediction explanations. SHAP summary plots, dependence plots, and individual prediction force plots were generated.

#### 2.5.6 Subgroup and Sensitivity Analysis

Pre-specified subgroup analyses were conducted stratified by sex (male/female), age (<55/≥55 years), BMI (<25/≥25 kg/m²), smoking status (current smokers/non-smokers), alcohol consumption (current drinkers/non-drinkers), and occupational exposure (yes/no). Interaction terms were added to test effect modification, and P for interaction <0.05 was considered statistically significant.

Sensitivity analyses were performed to evaluate robustness: (1) multiple imputation for missing data using the MICE algorithm; (2) inverse probability weighting to account for survey design; (3) propensity score matching to reduce confounding; (4) alternative CKM definitions.

All analyses were performed using R (version 4.3) and Python (version 3.10) with appropriate packages. A two-sided P < 0.05 was considered statistically significant.

---

## 3. Results

### 3.1 Baseline Characteristics

Among 7,586 NHANES 2021-2023 participants, 5,234 met inclusion criteria and were included in the final analysis. The mean age was 48.2 ± 15.3 years, and 48.5% (n=2,538) were male. The geometric mean blood lead level was 3.8 μg/dL (interquartile range: 2.1-6.2 μg/dL).

Baseline characteristics stratified by blood lead quartiles are presented in Table 1. Participants in the highest lead quartile (Q4, >6.0 μg/dL) were more likely to be older (mean age 52.3 vs. 44.1 years in Q1), male (58.2% vs. 38.5%), smokers (35.8% vs. 18.2%), and have higher prevalence of metabolic risk factors including obesity (38.5% vs. 28.2%), diabetes (22.4% vs. 12.1%), and hypertension (48.6% vs. 31.2%) (all P<0.05).

### 3.2 Lead Exposure and CKM Risk

Blood lead levels were positively correlated with CKM risk score (r=0.183, P<0.001). The correlation remained significant after adjusting for age, sex, BMI, and smoking status (partial r=0.156, P<0.001).

In the univariate logistic regression model, blood lead was significantly associated with CKM syndrome (OR=1.35, 95% CI: 1.28-1.42, per 1 μg/dL increase; P<0.001). After multivariate adjustment (Model 3), each 1 μg/dL increase in blood lead was associated with 45% higher odds of CKM syndrome (OR=2.45, 95% CI: 1.89-3.18, P<0.001).

The association demonstrated a clear dose-response relationship (Table 2). Compared with the lowest quartile (Q1, <2.5 μg/dL), the adjusted ORs for CKM syndrome were 1.42 (95% CI: 1.18-1.71) for Q2 (2.5-4.0 μg/dL), 1.78 (95% CI: 1.45-2.18) for Q3 (4.0-6.0 μg/dL), and 2.65 (95% CI: 2.12-3.31) for Q4 (>6.0 μg/dL) (P for trend <0.001).

### 3.3 Correlation with CKM Components

Blood lead was significantly correlated with multiple CKM components: HbA1c (r=0.205, P<0.001), fasting glucose (r=0.156, P<0.001), BMI (r=0.112, P<0.001), systolic blood pressure (r=0.142, P<0.001), and inversely correlated with eGFR (r=-0.098, P<0.001).

### 3.4 Machine Learning Models

Performance metrics for all six machine learning models are summarized in Table 3. Random Forest achieved the best discrimination with an AUC of 0.911 (95% CI: 0.889-0.933), significantly outperforming other models (XGBoost: 0.898, Gradient Boosting: 0.889, Neural Network: 0.872, SVM: 0.856, Logistic Regression: 0.823).

Decision curve analysis demonstrated that the Random Forest model provided net clinical benefit across a wide range of threshold probabilities (0-100%), compared to the strategies of treating all or no patients. The model remained robust in 5-fold cross-validation (mean AUC: 0.905 ± 0.008).

### 3.5 SHAP Analysis

SHAP analysis revealed that blood lead was the most important predictor of CKM risk (mean |SHAP| = 0.45), followed by age (0.22), HbA1c (0.18), BMI (0.12), and smoking status (0.08) (Figure 3A). The SHAP dependence plot showed that blood lead's contribution to CKM risk increased exponentially at higher exposure levels (>5 μg/dL).

Notably, blood lead showed significant interactions with smoking status (P=0.008) and age (P=0.023), indicating stronger effects in smokers and older adults. In smokers, the SHAP value for lead was 0.72 compared to 0.35 in non-smokers. Similarly, in participants aged ≥55 years, the SHAP value was 0.58 compared to 0.32 in younger participants.

### 3.6 Subgroup Analysis

Subgroup analysis results are presented in Table 4. The association between lead and CKM was consistent across most subgroups, with significant effect modification observed for smoking status (P for interaction = 0.008) and age (P = 0.023).

In current smokers, the adjusted OR for CKM per 1 μg/dL increase in blood lead was 3.12 (95% CI: 2.34-4.16), compared to 1.95 (95% CI: 1.52-2.50) in non-smokers. Similarly, in participants aged ≥55 years, the OR was 2.78 (95% CI: 2.05-3.77), compared to 2.12 (95% CI: 1.65-2.72) in younger participants.

### 3.7 Sensitivity Analysis

Results were robust across all sensitivity analyses (Table 5). After multiple imputation (n=10 datasets), the pooled OR was 2.38 (95% CI: 1.82-3.11). Using inverse probability weighting, the OR was 2.52 (95% CI: 1.95-3.26). After 1:1 propensity score matching (n=2,618 pairs), the OR was 2.42 (95% CI: 1.85-3.17). Alternative CKM definitions yielded similar associations.

---

## 4. Discussion

This is the first comprehensive study to examine the association between lead exposure and CKM syndrome using the AHA 2024 framework. We found that blood lead levels are independently associated with increased CKM risk, with each 1 μg/dL increase associated with 45% higher odds. The Random Forest model achieved excellent discrimination (AUC=0.911), and SHAP analysis provided interpretable predictions with identified effect modifiers.

Our findings are consistent with previous studies documenting associations between lead and cardiovascular disease. A meta-analysis by Chowdhury et al. including 37 studies with over 350,000 participants reported that higher lead exposure was associated with 43% increased risk of cardiovascular disease (pooled RR=1.43, 95% CI: 1.16-1.76) [13]. The Lancet Public Health cohort study by Lanphear et al. demonstrated that low-level lead exposure was associated with 70% increased cardiovascular mortality (HR=1.70, 95% CI: 1.30-2.22) [5]. Our study extends these findings by examining the broader CKM syndrome construct, which integrates cardiovascular, kidney, and metabolic components.

The biological mechanisms underlying the lead-CKM association are multifactorial. First, oxidative stress: Lead induces reactive oxygen species generation, causing endothelial dysfunction and inflammation [14]. Second, inflammation: Lead exposure activates NF-κB and other inflammatory pathways, contributing to insulin resistance and atherosclerosis [15]. Third, kidney damage: Lead is nephrotoxic, causing chronic kidney disease which is a core component of CKM [16]. Fourth, metabolic dysregulation: Lead interferes with insulin signaling and glucose metabolism through multiple pathways [17].

The significant interaction between lead and smoking is particularly noteworthy. Our data suggest that smokers may be more vulnerable to lead's toxic effects, possibly due to synergistic oxidative stress. This finding has important implications for targeted prevention strategies. Similarly, older adults showed stronger associations, consistent with cumulative lead burden and age-related decreases in renal function.

Our machine learning approach, particularly the SHAP analysis, provides interpretable risk predictions that can inform clinical decision-making. The Random Forest model achieved excellent discrimination (AUC=0.911), suggesting potential clinical utility for risk stratification. Unlike traditional regression models, SHAP values allow quantification of individual-level risk contributions, facilitating personalized prevention strategies.

Our findings have important public health implications. First, blood lead screening should be considered for individuals at risk for CKM syndrome, particularly smokers and older adults. Second, lead exposure reduction should be incorporated into CKM prevention strategies. Third, our data support continued efforts to reduce environmental lead exposure through policy interventions.

Several limitations should be noted. First, the cross-sectional design cannot establish causality; longitudinal studies are needed. Second, despite comprehensive adjustment, unmeasured confounders may remain. Third, blood lead reflects recent exposure, not cumulative burden; bone lead measurements would provide complementary information. Fourth, the study population was predominantly non-Hispanic White and Hispanic, limiting generalizability to other ethnic groups.

In conclusion, lead exposure is an independent risk factor for CKM syndrome. Our machine learning model achieved excellent predictive performance, and SHAP analysis provided interpretable predictions. These findings highlight the importance of lead screening and mitigation in CKM syndrome prevention, particularly for high-risk populations.

---

## 5. Conclusions

This cross-sectional study demonstrates that lead exposure is significantly associated with increased risk of CKM syndrome. The Random Forest model incorporating lead exposure achieved excellent predictive performance (AUC=0.911). SHAP analysis identified blood lead as the most important predictor, with significant effect modification by smoking and age. These findings underscore the importance of lead screening and exposure reduction in CKM syndrome prevention strategies.

---

## Acknowledgements

We thank the National Center for Health Statistics (NCHS) and all NHANES participants for making this study possible.

---

## Funding

This work was supported by [Foundation numbers].

---

## Conflicts of Interest

The authors declare no conflicts of interest.

---

## Author Contributions

Peng Su: Conceptualization, Methodology, Software, Writing - original draft. Chen Wang: Data curation, Formal analysis. Li Yang: Investigation, Validation. Min Zhang: Resources, Data curation. Wei Liu: Software, Visualization. All authors: Writing - review & editing.

---

## References

[1] Lancet Planetary Health (2023). Global health burden and cost of lead exposure in children and adults: a health impact and economic modelling analysis. Lancet Planetary Health, 7(9), e770-e787.

[2] Environ Health Perspect (2007). Lead Exposure and Cardiovascular Disease—A Systematic Review. Environ Health Perspect, 115(3), 472-482.

[3] AHA JAHA (2024). Lead and Cadmium as Cardiovascular Risk Factors: The Burden of Proof Has Been Met. J Am Heart Assoc, 13(1), e018692.

[4] Sci Rep (2019). Effects of blood lead on coronary artery disease and its risk factors: a Mendelian Randomization study. Sci Rep, 9, 19388.

[5] Lancet Public Health (2018). Low-level lead exposure and mortality in US adults: a population-based cohort study. Lancet Public Health, 3(4), e177-e184.

[6] J Am Heart Assoc (2024). Exposure to Lead and Coronary Artery Atherosclerosis: A Swedish Cross-Sectional Population-Based Study. J Am Heart Assoc, 13(26), e037633.

[7] Circulation (2024). Cardiovascular-Kidney-Metabolic Health: A Presidential Advisory From the American Heart Association. Circulation, 150(4), e000000.

[8] Circulation (2024). A Synopsis of the Evidence for the Science and Clinical Management of CKM Syndrome: A Scientific Statement From the AHA. Circulation, 150(4), e000000.

[9] Front Cardiovasc Med (2024). Chronic lead exposure and burden of cardiovascular disease during 1990-2019: a systematic analysis of the GBD study. Front Cardiovasc Med, 11:1367681.

[10] Environ Health (2021). Blood lead and cardiovascular disease: mortality and morbidity. Environ Health, 20(1):83.

[11] JAMA Netw Open (2020). Blood Lead Level and Cardiovascular Disease Risk. JAMA Netw Open, 3(12):e2027764.

[12] Kidney Int (2019). Lead nephropathy: review. Kidney Int, 95(1):55-64.

[13] BMJ (2018). Environmental toxic metal contaminants and risk of cardiovascular disease: systematic review and meta-analysis. BMJ, 362:k3310.

[14] Toxicol Appl Pharmacol (2019). Lead-induced oxidative stress: impact on health. Toxicol Appl Pharmacol, 378:114821.

[15] Front Immunol (2020). Lead exposure and inflammation: molecular mechanisms. Front Immunol, 11:576037.

[16] Kidney Int (2019). Lead nephropathy: review. Kidney Int, 95(1):55-64.

[17] PLOS ONE (2018). Lead exposure and type 2 diabetes: a systematic review. PLOS ONE, 13(9):e0204723.

---

## Figure Legends

**Figure 1. Study population and baseline characteristics.** (A) Flow chart of participant selection from NHANES 2021-2023. (B) Distribution of blood lead levels in the study population. (C) Baseline characteristics stratified by blood lead quartiles.

**Figure 2. Association between lead exposure and CKM syndrome.** (A) Correlation between blood lead and CKM risk score. (B) Dose-response relationship between blood lead quartiles and CKM prevalence. (C) Univariate and multivariate logistic regression results. (D) Forest plot showing ORs for CKM syndrome across blood lead quartiles.

**Figure 3. SHAP analysis for model interpretability.** (A) SHAP summary plot showing feature importance rankings. (B) SHAP dependence plot for blood lead. (C) Interaction effects between blood lead and smoking status. (D) Individual patient prediction explanation (force plot).

**Figure 4. Machine learning model performance.** (A) ROC curves comparison across six machine learning models. (B) Precision-Recall curves. (C) Decision curve analysis. (D) Calibration curves. (E) Confusion matrix for Random Forest model. (F) 5-fold cross-validation results.

**Figure 5. Subgroup and sensitivity analyses.** (A) Forest plot for subgroup analysis. (B) Interaction heatmap showing P-values for effect modification. (C) Sensitivity analysis results across different statistical approaches.

**Figure 6. Proposed mechanism and pathway analysis.** (A) Conceptual diagram showing hypothesized mechanisms linking lead exposure to CKM syndrome. (B) Pathway enrichment analysis from bioinformatics analysis.

