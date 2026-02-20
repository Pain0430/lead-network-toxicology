#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Professional Scientific Figure Generator
High-quality figures for CNS-level publications

Based on:
- Paul Tol's Schemes ( Colorscientific standard)
- Nature/Science/Cell figure guidelines
- Color theory for scientific visualization
"""

import os
import sys
import pandas as pd
import numpy as np

# Set matplotlib backend before import
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42  # Embed fonts
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
rcParams['axes.unicode_minus'] = False

# ============================================================================
# PAUL TOL'S COLOR SCHEMES (Scientific Standard)
# ============================================================================

# Bright scheme - for qualitative categories (most distinguishable)
TOL_BRIGHT = [
    '#4477AA',  # Blue
    '#EE6677',  # Red  
    '#228833',  # Green
    '#CCBB44',  # Yellow
    '#66CCEE',  # Cyan
    '#AA3377',  # Purple
    '#BBBBBB',  # Grey
]

# Muted scheme - for when brightness needs to be reduced
TOL_MUTED = [
    '#117733',  # Green
    '#88CCEE',  # Cyan
    '#44AA99',  # Teal
    '#999933',  # Olive
    '#882255',  # Ruby
    '#AA66CC',  # Purple
    '#DD7788',  # Pink
]

# Medium contrast scheme
TOL_MEDIUM = [
    '#332288',  # Indigo
    '#88Ccee',  # Cyan
    '#44AA99',  # Teal
    '#117733',  # Green
    '#999933',  # Olive
    '#DD7788',  # Pink
    '#8877AA',  # Lavender
]

# Dark blue scheme
TOL_DARK_BLUE = [
    '#001c7f', '#140e3a', '#b1400d', '#12711b', '#8c08ac'
]

# Colorblind-friendly palette
COLORBLIND_SAFE = [
    '#E69F00',  # Orange
    '#56B4E9',  # Sky Blue
    '#009E73',  # Bluish Green
    '#F0E442',  # Yellow
    '#0072B2',  # Blue
    '#D55E00',  # Vermillion
    '#CC79A7',  # Reddish Purple
]

# ============================================================================
# CUSTOM PROFESSIONAL PALETTES
# ============================================================================

# Lead research theme - sophisticated blues
LEAD_PALETTE = {
    'primary': '#1a5276',      # Deep blue
    'secondary': '#2980b9',    # Blue
    'accent': '#e74c3c',       # Red (for lead/important)
    'neutral': '#7f8c8d',      # Grey
    'light': '#ebf5fb',        # Light blue
    'dark': '#0a2540',         # Dark navy
    'highlight': '#f39c12',    # Gold/amber
    'success': '#27ae60',      # Green
    'gradient': ['#0a2540', '#1a5276', '#2980b9', '#5dade2', '#aed6f1'],
}

# CKM syndrome stages
CKM_STAGES = {
    0: '#27ae60',   # Green - healthy
    1: '#f1c40f',   # Yellow - risk
    2: '#e67e22',   # Orange - disease
    3: '#e74c3c',   # Red - severe
    4: '#8e44ad',   # Purple - critical
}

# ============================================================================
# FIGURE CLASSES
# ============================================================================

class ScientificFigure:
    """Base class for scientific figures"""
    
    def __init__(self, figsize=(10, 8), dpi=300):
        self.figsize = figsize
        self.dpi = dpi
        self.colors = LEAD_PALETTE
        self.tol_colors = TOL_BRIGHT
        
    def setup_figure(self, nrows=1, ncols=1, figsize=None):
        """Create figure with proper settings"""
        if figsize is None:
            figsize = self.figsize
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=self.dpi)
        fig.patch.set_facecolor('white')
        return fig, axes
    
    def style_axis(self, ax, title='', xlabel='', ylabel='', 
                   title_fontsize=14, label_fontsize=12,
                   hide_top_right=True):
        """Apply professional styling to axis"""
        if title:
            ax.set_title(title, fontsize=title_fontsize, fontweight='bold', pad=10)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=label_fontsize)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=label_fontsize)
        
        # Style spines
        if hide_top_right:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        # Style tick labels
        ax.tick_params(labelsize=10)
        
        # Add subtle grid
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
    def save_figure(self, fig, filename, bbox_inches='tight', pad_inches=0.1):
        """Save figure with proper settings"""
        fig.savefig(filename, dpi=self.dpi, bbox_inches=bbox_inches, 
                   pad_inches=pad_inches, facecolor='white', edgecolor='none')
        print(f"Saved: {filename}")
        plt.close(fig)


class Figure1_LeadDistribution(ScientificFigure):
    """Figure 1: Lead Distribution and Correlations"""
    
    def create(self, df, output_dir):
        print("Creating Figure 1: Lead Distribution and Correlations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=300)
        fig.patch.set_facecolor('white')
        
        # Color palette
        lead_color = self.colors['primary']
        accent_color = self.colors['accent']
        
        # ========== Panel A: Lead Distribution ==========
        ax = axes[0, 0]
        
        # Histogram with professional styling
        n, bins, patches = ax.hist(df['Blood_Lead'], bins=35, 
                                    color=lead_color, alpha=0.8,
                                    edgecolor='white', linewidth=0.5)
        
        # Add mean line
        mean_val = df['Blood_Lead'].mean()
        median_val = df['Blood_Lead'].median()
        
        ax.axvline(mean_val, color=accent_color, linestyle='--', 
                   linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color=self.colors['highlight'], linestyle=':', 
                   linewidth=2, label=f'Median: {median_val:.2f}')
        
        ax.set_xlabel('Blood Lead Concentration (μg/dL)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('A', fontsize=16, fontweight='bold', loc='left', pad=10)
        
        # Legend
        ax.legend(frameon=True, fontsize=10, loc='upper right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # ========== Panel B: Lead vs SBP ==========
        ax = axes[0, 1]
        
        # Scatter with density coloring
        scatter = ax.scatter(df['Blood_Lead'], df['SBP'], 
                           c=lead_color, alpha=0.4, s=30, edgecolor='none')
        
        # Trend line
        z = np.polyfit(df['Blood_Lead'], df['SBP'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df['Blood_Lead'].min(), df['Blood_Lead'].max(), 100)
        ax.plot(x_line, p(x_line), color=accent_color, linewidth=2.5, 
                linestyle='-', label='Linear trend')
        
        # Add correlation
        r = np.corrcoef(df['Blood_Lead'], df['SBP'])[0, 1]
        ax.text(0.05, 0.95, f'r = {r:.3f}', transform=ax.transAxes,
               fontsize=12, fontweight='bold', va='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Blood Lead (μg/dL)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Systolic Blood Pressure (mmHg)', fontsize=12, fontweight='bold')
        ax.set_title('B', fontsize=16, fontweight='bold', loc='left', pad=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # ========== Panel C: Lead vs HbA1c ==========
        ax = axes[1, 0]
        
        scatter = ax.scatter(df['Blood_Lead'], df['HbA1c'], 
                           c=self.colors['secondary'], alpha=0.4, s=30, edgecolor='none')
        
        # Trend line
        z = np.polyfit(df['Blood_Lead'], df['HbA1c'], 1)
        p = np.poly1d(z)
        ax.plot(x_line, p(x_line), color=self.colors['accent'], linewidth=2.5)
        
        r = np.corrcoef(df['Blood_Lead'], df['HbA1c'])[0, 1]
        ax.text(0.05, 0.95, f'r = {r:.3f}', transform=ax.transAxes,
               fontsize=12, fontweight='bold', va='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Blood Lead (μg/dL)', fontsize=12, fontweight='bold')
        ax.set_ylabel('HbA1c (%)', fontsize=12, fontweight='bold')
        ax.set_title('C', fontsize=16, fontweight='bold', loc='left', pad=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # ========== Panel D: MetS by Lead Quartile ==========
        ax = axes[1, 1]
        
        df['Lead_Quartile'] = pd.qcut(df['Blood_Lead'], 4, labels=['Q1\n(Low)', 'Q2', 'Q3', 'Q4\n(High)'])
        quartile_means = df.groupby('Lead_Quartile')['MetS'].mean()
        
        # Professional bar chart
        bars = ax.bar(quartile_means.index, quartile_means.values, 
                     color=self.colors['gradient'][1:5], 
                     edgecolor='white', linewidth=1)
        
        # Add value labels on bars
        for bar, val in zip(bars, quartile_means.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_xlabel('Blood Lead Quartile', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Metabolic Syndrome Score', fontsize=12, fontweight='bold')
        ax.set_title('D', fontsize=16, fontweight='bold', loc='left', pad=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add panel labels
        for i, ax in enumerate(axes.flat):
            ax.text(-0.12, 1.05, chr(65+i), transform=ax.transAxes,
                   fontsize=18, fontweight='bold', va='top')
        
        plt.tight_layout()
        
        # Save
        filename = os.path.join(output_dir, 'fig1_lead_analysis_v2.png')
        self.save_figure(fig, filename)
        
        return fig


class Figure2_CorrelationHeatmap(ScientificFigure):
    """Figure 2: Professional Correlation Heatmap"""
    
    def create(self, df, output_dir):
        print("Creating Figure 2: Correlation Heatmap...")
        
        fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
        fig.patch.set_facecolor('white')
        
        # Select columns for correlation
        corr_cols = ['Blood_Lead', 'SBP', 'DBP', 'BMI', 'HbA1c', 
                    'Triglycerides', 'HDL', 'MetS']
        
        # Short labels
        labels = ['Blood Pb', 'SBP', 'DBP', 'BMI', 'HbA1c', 
                 'Triglycerides', 'HDL', 'MetS']
        
        # Calculate correlation
        corr_matrix = df[corr_cols].corr()
        
        # Create heatmap with diverging colormap
        cmap = plt.cm.RdBu_r  # Red-Blue diverging
        vmin, vmax = -1, 1
        
        im = ax.imshow(corr_matrix.values, cmap=cmap, aspect='auto', 
                      vmin=vmin, vmax=vmax)
        
        # Set ticks
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, fontsize=11, fontweight='bold', rotation=45, ha='right')
        ax.set_yticklabels(labels, fontsize=11, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('Pearson Correlation', fontsize=12, fontweight='bold')
        cbar.ax.tick_params(labelsize=10)
        
        # Add correlation values
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = corr_matrix.iloc[i, j]
                color = 'white' if abs(val) > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                       fontsize=10, fontweight='bold', color=color)
        
        ax.set_title('Correlation Matrix: Lead and CKM Indicators', 
                    fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        filename = os.path.join(output_dir, 'fig2_correlation_heatmap_v2.png')
        self.save_figure(fig, filename)
        
        return fig


class Figure3_AOPPathway(ScientificFigure):
    """Figure 3: AOP Pathway Diagram - Professional Style"""
    
    def create(self, output_dir):
        print("Creating Figure 3: AOP Pathway Diagram...")
        
        fig, ax = plt.subplots(figsize=(16, 12), dpi=300)
        fig.patch.set_facecolor('white')
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 12)
        ax.axis('off')
        
        # Title
        ax.text(8, 11.5, 'Adverse Outcome Pathway: Lead-induced CKM Syndrome', 
               fontsize=20, fontweight='bold', ha='center', va='center')
        ax.text(8, 10.8, 'A Systems Toxicology Approach', 
               fontsize=14, ha='center', va='center', style='italic',
               color=self.colors['neutral'])
        
        # Color scheme for pathways
        pathway_colors = {
            'exposure': self.colors['primary'],
            'oxidative': '#e74c3c',
            'ras': '#27ae60',
            'endothelial': '#3498db',
            'inflammation': '#9b59b6',
            'outcome': self.colors['accent']
        }
        
        # Box style function
        def draw_box(ax, x, y, width, height, text, color, fontsize=11):
            """Draw a professional box"""
            rect = FancyBboxPatch((x-width/2, y-height/2), width, height,
                                  boxstyle="round,pad=0.02,rounding_size=0.3",
                                  facecolor=color, edgecolor='white', 
                                  linewidth=2, alpha=0.9)
            ax.add_patch(rect)
            ax.text(x, y, text, ha='center', va='center', 
                   fontsize=fontsize, fontweight='bold', color='white')
        
        # Draw pathway
        # Row 1: MIE
        draw_box(ax, 8, 9, 4, 0.8, 'Lead Exposure\n(MIE)', pathway_colors['exposure'], 12)
        
        # Arrow
        ax.annotate('', xy=(8, 8.2), xytext=(8, 8.6),
                   arrowprops=dict(arrowstyle='->', color='black', lw=2))
        
        # Row 2: KE1 - Oxidative Stress (Hub)
        draw_box(ax, 8, 7.5, 5, 1, 'Oxidative Stress\n(KE1: Key Event)', pathway_colors['oxidative'], 12)
        
        # Row 3: Three branches
        branches = [
            (3, 5, 'RAS Activation\n(KE2a)', pathway_colors['ras']),
            (8, 5, 'Endothelial\nDysfunction (KE2b)', pathway_colors['endothelial']),
            (13, 5, 'Inflammation\n(KE2c)', pathway_colors['inflammation']),
        ]
        
        for x, y, text, color in branches:
            draw_box(ax, x, y, 3.5, 1, text, color)
            # Arrow from KE1
            ax.annotate('', xy=(x, 5.6), xytext=(8, 7),
                       arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, 
                                     connectionstyle='arc3,rad=0.1'))
        
        # Row 4: Organ damage
        damage_boxes = [
            (3, 3, 'Hypertension', pathway_colors['ras']),
            (8, 3, 'Metabolic\nSyndrome', pathway_colors['endothelial']),
            (13, 3, 'Kidney Damage', pathway_colors['inflammation']),
        ]
        
        for x, y, text, color in damage_boxes:
            draw_box(ax, x, y, 3, 0.8, text, color, 11)
            # Arrows from KE2 to damage
            if x == 3:
                ax.annotate('', xy=(3, 3.6), xytext=(3, 4.6),
                           arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
            elif x == 8:
                ax.annotate('', xy=(8, 3.6), xytext=(8, 4.6),
                           arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
            else:
                ax.annotate('', xy=(13, 3.6), xytext=(13, 4.6),
                           arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
        
        # Row 5: Final outcome
        ax.annotate('', xy=(8, 2.2), xytext=(8, 2.6),
                   arrowprops=dict(arrowstyle='->', color='black', lw=2))
        draw_box(ax, 8, 1.5, 6, 1, 'CKM Syndrome Progression\n(Adverse Outcome)', 
                pathway_colors['outcome'], 13)
        
        # Add legend/notes
        note_text = """
        MIE: Molecular Initiating Event
        KE: Key Event
        CKM: Cardiovascular-Kidney-Metabolic
        
        Mediation Analysis: SBP mediates 88.6% of Pb→CKM effect
        """
        ax.text(0.5, 1, note_text, fontsize=9, va='bottom', ha='left',
               color=self.colors['neutral'])
        
        # Add methodology box
        method_text = """Methods: NHANES 2021-2023 (n=7,586)
Network Toxicology: 96 targets, 10 pathways
VCell: Endothelial & Macrophage models"""
        ax.text(15.5, 1, method_text, fontsize=9, va='bottom', ha='right',
               color=self.colors['neutral'])
        
        filename = os.path.join(output_dir, 'fig3_aop_pathway_v2.png')
        self.save_figure(fig, filename)
        
        return fig


def generate_all_figures(data_file, output_dir):
    """Generate all figures"""
    
    print("="*60)
    print("Generating Professional Scientific Figures")
    print("="*60)
    
    # Load data
    if os.path.exists(data_file):
        df = pd.read_csv(data_file)
        print(f"Loaded data: {len(df)} samples")
    else:
        print(f"Data file not found: {data_file}")
        print("Generating simulation data...")
        # Generate simulation data
        np.random.seed(42)
        n = 500
        df = pd.DataFrame({
            'Blood_Lead': np.random.lognormal(mean=-0.5, sigma=0.8, size=n),
            'SBP': np.random.normal(125, 15, size=n),
            'DBP': np.random.normal(80, 10, size=n),
            'BMI': np.random.normal(27, 5, size=n),
            'HbA1c': np.random.normal(5.5, 1, size=n),
            'Triglycerides': np.random.lognormal(5, 0.5, size=n),
            'HDL': np.random.normal(50, 15, size=n),
            'MetS': np.random.randint(0, 4, size=n),
        })
        # Add correlations
        df['SBP'] = df['SBP'] + df['Blood_Lead'] * 5
        df['HbA1c'] = df['HbA1c'] + df['Blood_Lead'] * 0.1
    
    # Create figures
    fig1 = Figure1_LeadDistribution()
    fig1.create(df, output_dir)
    
    fig2 = Figure2_CorrelationHeatmap()
    fig2.create(df, output_dir)
    
    fig3 = Figure3_AOPPathway()
    fig3.create(output_dir)
    
    print("\n" + "="*60)
    print("All figures generated successfully!")
    print("="*60)


if __name__ == "__main__":
    OUTPUT_DIR = "output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Try to use real data, fallback to simulation
    data_file = os.path.join(OUTPUT_DIR, "simulated_lead_ckm_data.csv")
    generate_all_figures(data_file, OUTPUT_DIR)
