#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test and Generate Simulation Data for All Analysis Scripts
Verify code works with simulated data and generate figures
"""

import os
import sys
import pandas as pd
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_simulation_data():
    """Generate simulation data for testing"""
    print("="*60)
    print("Generating Simulation Data for Testing")
    print("="*60)
    
    np.random.seed(42)
    n = 500
    
    # Simulated blood lead and CKM data
    data = {
        'SEQN': range(1, n+1),
        'Blood_Lead': np.random.lognormal(mean=-0.5, sigma=0.8, size=n),
        'Blood_Cd': np.random.lognormal(mean=-2, sigma=0.5, size=n),
        'Age': np.random.randint(20, 80, size=n),
        'Gender': np.random.randint(1, 3, size=n),
        'SBP': np.random.normal(125, 15, size=n) + np.random.lognormal(0, 0.3, size=n) * 10,
        'DBP': np.random.normal(80, 10, size=n),
        'BMI': np.random.normal(27, 5, size=n),
        'Waist': np.random.normal(90, 12, size=n),
        'HDL': np.random.normal(50, 15, size=n),
        'Triglycerides': np.random.lognormal(5, 0.5, size=n),
        'HbA1c': np.random.normal(5.5, 1, size=n),
    }
    
    df = pd.DataFrame(data)
    
    # Add correlations (lead affects other variables)
    df['SBP'] = df['SBP'] + df['Blood_Lead'] * 5
    df['HbA1c'] = df['HbA1c'] + df['Blood_Lead'] * 0.1
    df['Triglycerides'] = df['Triglycerides'] + df['Blood_Lead'] * 10
    
    # Calculate CKM indicators
    df['High_TG'] = (df['Triglycerides'] >= 150).astype(int)
    df['Low_HDL'] = np.where(df['Gender']==2, df['HDL']<50, df['HDL']<40).astype(int)
    df['High_BP'] = ((df['SBP']>=130) | (df['DBP']>=85)).astype(int)
    df['High_HbA1c'] = (df['HbA1c']>=5.7).astype(int)
    df['MetS'] = df['High_TG'] + df['Low_HDL'] + df['High_BP'] + df['High_HbA1c']
    
    # Save simulation data
    df.to_csv(f"{OUTPUT_DIR}/simulated_lead_ckm_data.csv", index=False)
    print(f"Saved: {OUTPUT_DIR}/simulated_lead_ckm_data.csv")
    print(f"Sample size: {len(df)}")
    print(f"Blood Lead range: {df['Blood_Lead'].min():.2f} - {df['Blood_Lead'].max():.2f}")
    
    return df

def generate_figures():
    """Generate figures using matplotlib"""
    print("\n" + "="*60)
    print("Generating Figures")
    print("="*60)
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        # Load data
        df = pd.read_csv(f"{OUTPUT_DIR}/simulated_lead_ckm_data.csv")
        
        # Figure 1: Blood Lead Distribution
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Blood Lead Distribution
        axes[0, 0].hist(df['Blood_Lead'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Blood Lead (ug/dL)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Blood Lead')
        axes[0, 0].axvline(df['Blood_Lead'].mean(), color='red', linestyle='--', label=f"Mean: {df['Blood_Lead'].mean():.2f}")
        axes[0, 0].legend()
        
        # 2. Blood Lead vs SBP
        axes[0, 1].scatter(df['Blood_Lead'], df['SBP'], alpha=0.5, c='coral', s=20)
        axes[0, 1].set_xlabel('Blood Lead (ug/dL)')
        axes[0, 1].set_ylabel('Systolic Blood Pressure (mmHg)')
        axes[0, 1].set_title('Blood Lead vs SBP')
        # Add trend line
        z = np.polyfit(df['Blood_Lead'], df['SBP'], 1)
        p = np.poly1d(z)
        axes[0, 1].plot(df['Blood_Lead'].sort_values(), p(df['Blood_Lead'].sort_values()), 
                       "r--", alpha=0.8, label='Trend')
        axes[0, 1].legend()
        
        # 3. Blood Lead vs HbA1c
        axes[1, 0].scatter(df['Blood_Lead'], df['HbA1c'], alpha=0.5, c='green', s=20)
        axes[1, 0].set_xlabel('Blood Lead (ug/dL)')
        axes[1, 0].set_ylabel('HbA1c (%)')
        axes[1, 0].set_title('Blood Lead vs HbA1c')
        
        # 4. CKM Risk Score by Lead Quartiles
        df['Lead_Quartile'] = pd.qcut(df['Blood_Lead'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        ckm_by_quartile = df.groupby('Lead_Quartile')['MetS'].mean()
        axes[1, 1].bar(ckm_by_quartile.index, ckm_by_quartile.values, color='purple', alpha=0.7)
        axes[1, 1].set_xlabel('Blood Lead Quartile')
        axes[1, 1].set_ylabel('Mean MetS Score')
        axes[1, 1].set_title('Metabolic Syndrome Score by Lead Quartile')
        
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/fig1_lead_analysis.png", dpi=150, bbox_inches='tight')
        print(f"Saved: {OUTPUT_DIR}/fig1_lead_analysis.png")
        
        # Figure 2: Correlation Heatmap
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        corr_cols = ['Blood_Lead', 'SBP', 'DBP', 'BMI', 'HbA1c', 'Triglycerides', 'HDL', 'MetS']
        corr_matrix = df[corr_cols].corr()
        im = ax2.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        ax2.set_xticks(range(len(corr_cols)))
        ax2.set_yticks(range(len(corr_cols)))
        ax2.set_xticklabels(corr_cols, rotation=45, ha='right')
        ax2.set_yticklabels(corr_cols)
        ax2.set_title('Correlation Heatmap: Lead and CKM Indicators')
        plt.colorbar(im, ax=ax2)
        
        # Add correlation values
        for i in range(len(corr_cols)):
            for j in range(len(corr_cols)):
                text = ax2.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/fig2_correlation_heatmap.png", dpi=150, bbox_inches='tight')
        print(f"Saved: {OUTPUT_DIR}/fig2_correlation_heatmap.png")
        
        # Figure 3: Pathway Diagram (conceptual)
        fig3, ax3 = plt.subplots(figsize=(12, 8))
        ax3.set_xlim(0, 10)
        ax3.set_ylim(0, 10)
        ax3.axis('off')
        ax3.set_title('Lead-induced CKM Syndrome: AOP Framework', fontsize=14, fontweight='bold')
        
        # Draw boxes
        boxes = [
            (5, 9, 'Lead Exposure', 'lightblue'),
            (5, 7, 'Oxidative Stress\n(SOD, CAT, NOS)', 'lightyellow'),
            (2, 5, 'RAS Activation\n(ACE, AngII)', 'lightcoral'),
            (5, 5, 'Endothelial Dysfunction\n(NO down)', 'lightsalmon'),
            (8, 5, 'Inflammation\n(NF-kB, IL-1b)', 'lavender'),
            (3, 3, 'Hypertension\n(SBP up)', 'lightgreen'),
            (5, 3, 'Metabolic Syndrome', 'lightyellow'),
            (7, 3, 'Kidney Damage\n(CKD)', 'lightgray'),
            (5, 1, 'CKM Syndrome Progression', 'orange'),
        ]
        
        for x, y, text, color in boxes:
            rect = plt.Rectangle((x-1.3, y-0.4), 2.6, 0.8, fill=True, facecolor=color, edgecolor='black', linewidth=2)
            ax3.add_patch(rect)
            ax3.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Draw arrows
        arrows = [
            (5, 8.6, 5, 7.4),
            (5, 6.6, 3.4, 5.4),
            (5, 6.6, 5, 5.4),
            (5, 6.6, 6.6, 5.4),
            (2.6, 5, 2.4, 3.4),
            (5.4, 5, 4.6, 3.4),
            (6.6, 5, 6.4, 3.4),
            (3.4, 2.6, 4.4, 1.4),
            (5.4, 2.6, 4.6, 1.4),
            (6.4, 2.6, 5.6, 1.4),
        ]
        
        for x1, y1, x2, y2 in arrows:
            ax3.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        
        plt.savefig(f"{OUTPUT_DIR}/fig3_aop_pathway.png", dpi=150, bbox_inches='tight')
        print(f"Saved: {OUTPUT_DIR}/fig3_aop_pathway.png")
        
        plt.close('all')
        print("\nAll figures generated successfully!")
        return True
        
    except ImportError as e:
        print(f"Matplotlib not available: {e}")
        return False
    except Exception as e:
        print(f"Error generating figures: {e}")
        return False

def main():
    print("="*60)
    print("Testing Code with Simulation Data")
    print("="*60)
    
    # Step 1: Generate simulation data
    generate_simulation_data()
    
    # Step 2: Generate figures
    generate_figures()
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Output directory: {OUTPUT_DIR}")
    print("Files generated:")
    for f in os.listdir(OUTPUT_DIR):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024
        print(f"  - {f} ({size:.1f} KB)")

if __name__ == "__main__":
    main()
