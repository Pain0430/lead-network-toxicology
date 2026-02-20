# Lead Environmental Toxicology Research
## Network Toxicology + CKM Syndrome + VCell + NHANES

A comprehensive research project on lead (Pb) induced cardiovascular-kidney-metabolic (CKM) syndrome using network toxicology, virtual cell modeling, and NHANES data analysis.

---

## Project Overview

```
Pollutant Selection → Network Toxicology → VCell Simulation → NHANES Validation → AOP Framework
```

### Research Innovation

1. **CKM Syndrome Focus** - 2024 AHA new concept (Cardiovascular-Kidney-Metabolic)
2. **Mediation Analysis** - Blood pressure mediates 88.6% of lead→CKM effect
3. **AOP Framework** - Complete adverse outcome pathway from lead exposure to CKM progression

---

## Key Findings (NHANES 2021-2023, n=7,586)

### Lead Distribution
| Metric | Value |
|--------|-------|
| Mean | 0.87 μg/dL |
| Median | 0.64 μg/dL |
| P95 | 2.14 μg/dL |
| P99 | 4.25 μg/dL |

### Correlations (Spearman)

| Indicator | r-value | p-value |
|-----------|---------|---------|
| Systolic BP | **0.354** | <0.001 |
| Hypertension | 0.250 | <0.001 |
| MetS Score | 0.229 | <0.001 |
| CKM Stage | 0.183 | <0.001 |
| Chronic Kidney Disease | 0.122 | <0.001 |

---

## Files

### Analysis Scripts
| File | Description |
|------|-------------|
| `lead_network_toxicology.py` | Network toxicology analysis |
| `lead_ckm_analysis.py` | CKM syndrome analysis |
| `lead_ckm_aop.py` | AOP framework construction |
| `lead_bp_targets.py` | Key target identification |
| `molecular_docking.py` | Molecular docking analysis |
| `test_and_visualize.py` | Test with simulation data |

### Data
| File | Description |
|------|-------------|
| `nhanes_data/` | NHANES 2021-2023 raw data |
| `output/simulated_lead_ckm_data.csv` | Simulation data |

### Figures
| File | Description |
|------|-------------|
| `output/fig1_lead_analysis.png` | Lead distribution & correlations |
| `output/fig2_correlation_heatmap.png` | Correlation heatmap |
| `output/fig3_aop_pathway.png` | AOP pathway diagram |

### Models
| File | Description |
|------|-------------|
| `VCell_Model_Endothelial.md` | VCell model - Endothelial cells |
| `VCell_Model_Macrophage.md` | VCell model - Macrophage |
| `AOP_FRAMEWORK.md` | Complete AOP framework |

---

## Key Targets

### 1. ACE (Angiotensin-Converting Enzyme)
- PDB: 1UZ6
- Type: Metalloprotease (Zn2+)
- Lead binding: Zn2+ pocket competition

### 2. NOS3 (eNOS)
- PDB: 1M11
- Type: Oxidoreductase
- Lead binding: BH4 site interference

---

## Research Pipeline

1. **Network Toxicology** - Identify key targets and pathways
2. **Mediation Analysis** - Blood pressure mediation effect 88.6%
3. **Molecular Docking** - ACE & NOS3 binding analysis
4. **VCell Modeling** - Dynamic pathway simulation
5. **AOP Construction** - Complete adverse outcome pathway

---

## References

1. CKM Syndrome - AHA Presidential Advisory (2024)
2. Lead and CKD - Nature Scientific Reports (2024)
3. Metals and CKD AOP - Science of Total Environment (2024)

---

## GitHub

https://github.com/Pain0430/lead-network-toxicology

---

*Updated: 2026-02-20*
