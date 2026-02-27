#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scientific Figure Style Module
CNS-level publication quality figures

Usage:
    from figure_style import ScientificStyle, LEAD_PALETTE
    fig, ax = plt.subplots()
    ax.set_color(LEAD_PALETTE['primary'])
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# PAUL TOL'S COLOR SCHEMES (Scientific Standard)
# ============================================================================

# Bright scheme - for qualitative categories
TOL_BRIGHT = [
    '#4477AA',  # Blue
    '#EE6677',  # Red  
    '#228833',  # Green
    '#CCBB44',  # Yellow
    '#66CCEE',  # Cyan
    '#AA3377',  # Purple
    '#BBBBBB',  # Grey
]

# Muted scheme
TOL_MUTED = [
    '#117733',  # Green
    '#88CCEE',  # Cyan
    '#44AA99',  # Teal
    '#999933',  # Olive
    '#882255',  # Ruby
    '#AA66CC',  # Purple
    '#DD7788',  # Pink
]

# Colorblind-friendly palette (Wong)
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

# Lead research theme
LEAD_PALETTE = {
    'primary': '#1a5276',      # Deep blue
    'secondary': '#2980b9',    # Blue
    'accent': '#e74c3c',       # Red
    'neutral': '#7f8c8d',      # Grey
    'light': '#ebf5fb',        # Light blue
    'dark': '#0a2540',         # Dark navy
    'highlight': '#f39c12',    # Gold
    'success': '#27ae60',      # Green
    'gradient': ['#0a2540', '#1a5276', '#2980b9', '#5dade2', '#aed6f1'],
}

# CKM stages
CKM_STAGES = {
    0: '#27ae60',   # Green - healthy
    1: '#f1c40f',   # Yellow - risk
    2: '#e67e22',   # Orange - disease
    3: '#e74c3c',   # Red - severe
    4: '#8e44ad',   # Purple - critical
}

# Nature-style palette
NATURE_PALETTE = {
    'blue': '#3B4992',
    'red': '#EE0000',
    'green': '#008B45',
    'orange': '#FF8C00',
    'purple': '#800080',
    'grey': '#808080',
}

# ============================================================================
# STYLE FUNCTIONS
# ============================================================================

def apply_scientific_style(ax, hide_top_right=True, grid=True):
    """Apply scientific publication style to axis"""
    if hide_top_right:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    if grid:
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
    
    ax.tick_params(labelsize=10)
    return ax


def add_correlation_annotation(ax, r, x=0.05, y=0.95):
    """Add correlation coefficient annotation"""
    ax.text(x, y, f'r = {r:.3f}', transform=ax.transAxes,
           fontsize=12, fontweight='bold', va='top',
           bbox=dict(boxstyle='round', facecolor='white', 
                    alpha=0.8, edgecolor='gray'))


def create_publication_figure(figsize=(10, 8), dpi=300):
    """Create figure with publication settings"""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor('white')
    return fig, ax


def save_publication_figure(fig, filename, dpi=300, bbox_inches='tight'):
    """Save figure with publication settings"""
    fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches,
               facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Saved: {filename}")


# ============================================================================
# QUICK PLOT FUNCTIONS
# ============================================================================

def plot_distribution(data, title, xlabel, filename, color='#1a5276'):
    """Quick distribution plot"""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    fig.patch.set_facecolor('white')
    
    ax.hist(data, bins=30, color=color, alpha=0.8, edgecolor='white')
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    apply_scientific_style(ax)
    save_publication_figure(fig, filename)


def plot_scatter_with_trend(x, y, title, xlabel, ylabel, filename, 
                            color='#2980b9', annotate_r=True):
    """Quick scatter plot with trend line"""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    fig.patch.set_facecolor('white')
    
    ax.scatter(x, y, c=color, alpha=0.4, s=30, edgecolor='none')
    
    # Trend line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, p(x_line), color='#e74c3c', linewidth=2.5)
    
    if annotate_r:
        r = np.corrcoef(x, y)[0, 1]
        add_correlation_annotation(ax, r)
    
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    apply_scientific_style(ax)
    save_publication_figure(fig, filename)


def plot_correlation_heatmap(corr_matrix, title, filename, labels=None):
    """Quick correlation heatmap"""
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    fig.patch.set_facecolor('white')
    
    cmap = plt.cm.RdBu_r
    im = ax.imshow(corr_matrix.values, cmap=cmap, aspect='auto', vmin=-1, vmax=1)
    
    if labels is not None:
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)
    
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            val = corr_matrix.iloc[i, j]
            color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                   fontsize=9, fontweight='bold', color=color)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    save_publication_figure(fig, filename)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Create a simple figure
    print("Figure Style Module")
    print("Available palettes: TOL_BRIGHT, TOL_MUTED, COLORBLIND_SAFE, LEAD_PALETTE")
    print("Functions: apply_scientific_style, add_correlation_annotation,")
    print("          plot_distribution, plot_scatter_with_trend, plot_correlation_heatmap")
