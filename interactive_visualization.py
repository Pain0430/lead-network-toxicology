#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive Visualization Module using Plotly
Enhanced scientific visualization for lead-network-toxicology project

Features:
- Interactive scatter plots with hover tooltips
- Interactive heatmaps for correlation matrices
- Interactive network visualization
- Dose-response curves with confidence intervals
- CKM syndrome staging visualization
"""

import os
import json
import numpy as np
import pandas as pd

# Plotly imports
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Color schemes
TOL_BRIGHT = [
    '#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377', '#BBBBBB'
]

CKM_COLORS = {
    0: '#27ae60',   # Green - healthy
    1: '#f1c40f',   # Yellow - risk
    2: '#e67e22',   # Orange - disease
    3: '#e74c3c',   # Red - severe
    4: '#8e44ad',   # Purple - critical
}

METAL_COLORS = {
    'Pb': '#4477AA',   # Blue - Lead
    'As': '#EE6677',   # Red - Arsenic
    'Cd': '#228833',   # Green - Cadmium
    'Hg': '#CCBB44',   # Yellow - Mercury
    'Mn': '#66CCEE',   # Cyan - Manganese
}

PFAS_COLORS = {
    'PFOA': '#4477AA',
    'PFOS': '#EE6677',
    'PFNA': '#228833',
    'GenX': '#CCBB44',
}


class InteractiveVisualization:
    """Interactive visualization class for toxicology data"""
    
    def __init__(self, output_dir='output'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    # =========================================================================
    # CORRELATION VISUALIZATIONS
    # =========================================================================
    
    def create_interactive_correlation_heatmap(self, df, metals=None, 
                                                  title="Heavy Metal Correlation Heatmap"):
        """
        Create interactive correlation heatmap
        
        Args:
            df: DataFrame with metal concentrations
            metals: List of metal columns to include
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        if metals is None:
            metals = ['LBXBPB', 'LBXIAS', 'LBXBCD', 'LBXIHG', 'LBXBMN']
            
        # Calculate correlation matrix
        corr_matrix = df[metals].corr()
        
        # Map column names to labels
        metal_labels = {
            'LBXBPB': 'Lead (Pb)',
            'LBXIAS': 'Arsenic (As)', 
            'LBXBCD': 'Cadmium (Cd)',
            'LBXIHG': 'Mercury (Hg)',
            'LBXBMN': 'Manganese (Mn)'
        }
        
        # Rename for display
        corr_display = corr_matrix.rename(index=metal_labels, columns=metal_labels)
        
        # Create heatmap
        fig = px.imshow(
            corr_display,
            color_continuous_scale='RdBu_r',
            range_color=[-1, 1],
            title=title,
            labels=dict(color="Correlation")
        )
        
        fig.update_layout(
            font=dict(family="Arial", size=12),
            width=700,
            height=600,
            title_font_size=16
        )
        
        # Add correlation values as text
        fig.update_traces(
            text=corr_display.round(2).values,
            texttemplate="%{text}",
            textfont=dict(size=10)
        )
        
        return fig
    
    def create_interactive_scatter_matrix(self, df, metals=None, 
                                           color_col=None, title="Metal Scatter Matrix"):
        """
        Create interactive scatter plot matrix
        
        Args:
            df: DataFrame with data
            metals: List of metal columns
            color_col: Column to color by (e.g., CKM stage)
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        if metals is None:
            metals = ['LBXBPB', 'LBXIAS', 'LBXBCD', 'LBXBMN']
            
        metal_labels = {
            'LBXBPB': 'Lead (Pb)',
            'LBXIAS': 'Arsenic (As)', 
            'LBXBCD': 'Cadmium (Cd)',
            'LBXBMN': 'Manganese (Mn)'
        }
        
        # Prepare data
        plot_df = df[metals + ([color_col] if color_col else [])].copy()
        plot_df.columns = [metal_labels.get(c, c) for c in plot_df.columns]
        
        # Create scatter matrix
        fig = px.scatter_matrix(
            plot_df,
            dimensions=metal_labels.values(),
            color=color_col,
            color_continuous_scale='Viridis' if not color_col else None,
            title=title
        )
        
        fig.update_traces(diagonal_visible=False, showupperhalf=False)
        fig.update_layout(
            width=900,
            height=800,
            font=dict(family="Arial", size=10)
        )
        
        return fig
    
    # =========================================================================
    # DOSE-RESPONSE VISUALIZATIONS
    # =========================================================================
    
    def create_dose_response_curve(self, df, metal_col, outcome_col, 
                                     title=None, color=None):
        """
        Create interactive dose-response curve with confidence interval
        
        Args:
            df: DataFrame with data
            metal_col: Metal concentration column
            outcome_col: Outcome variable column
            title: Plot title
            color: Line color
            
        Returns:
            Plotly figure object
        """
        metal_name = metal_col.replace('LBX', '').replace('BPB', 'Lead')
        
        if title is None:
            title = f"{metal_name} vs {outcome_col}"
            
        if color is None:
            color = METAL_COLORS.get(metal_name[:2], '#4477AA')
        
        # Create bins for metal levels
        df_clean = df[[metal_col, outcome_col]].dropna()
        
        if len(df_clean) < 50:
            # Simple scatter if too few points
            fig = px.scatter(
                df_clean, x=metal_col, y=outcome_col,
                title=title,
                trendline="lowess",
                color_discrete_sequence=[color]
            )
        else:
            # Create quantile-based bins
            df_clean['metal_quantile'] = pd.qcut(df_clean[metal_col], q=5, labels=False, duplicates='drop')
            
            # Calculate mean and CI for each bin
            grouped = df_clean.groupby('metal_quantile').agg({
                metal_col: ['mean', 'std', 'count'],
                outcome_col: 'mean'
            }).reset_index()
            grouped.columns = ['quantile', 'metal_mean', 'metal_std', 'n', 'outcome_mean']
            grouped['se'] = grouped['metal_std'] / np.sqrt(grouped['n'])
            grouped['ci95'] = 1.96 * grouped['se']
            
            # Create figure with error bars
            fig = go.Figure()
            
            # Data points with error bars
            fig.add_trace(go.Scatter(
                x=grouped['metal_mean'],
                y=grouped['outcome_mean'],
                mode='markers+lines',
                name='Mean ± 95% CI',
                error_y=dict(
                    type='data',
                    array=grouped['ci95'],
                    visible=True
                ),
                line=dict(color=color, width=2),
                marker=dict(size=10, color=color)
            ))
            
            # Add trendline using all data
            x_range = np.linspace(df_clean[metal_col].min(), df_clean[metal_col].max(), 100)
            
            fig.update_layout(
                title=title,
                xaxis_title=f"{metal_name} (μg/dL)",
                yaxis_title=outcome_col,
                width=700,
                height=500,
                font=dict(family="Arial", size=12),
                showlegend=True
            )
        
        return fig
    
    # =========================================================================
    # CKM SYNDROME VISUALIZATION
    # =========================================================================
    
    def create_ckm_staging_plot(self, df, lead_col='LBXBPB', 
                                  ckm_stage_col='CKM_STAGE'):
        """
        Create interactive CKM syndrome staging visualization
        
        Args:
            df: DataFrame with lead and CKM stage data
            lead_col: Blood lead column
            ckm_stage_col: CKM stage column
            
        Returns:
            Plotly figure object
        """
        # Box plot by CKM stage
        fig = px.box(
            df, 
            x=ckm_stage_col, 
            y=lead_col,
            color=ckm_stage_col,
            color_discrete_map=CKM_COLORS,
            title="Blood Lead Levels by CKM Syndrome Stage",
            labels={
                ckm_stage_col: 'CKM Stage',
                lead_col: 'Blood Lead (μg/dL)'
            }
        )
        
        # Add stage descriptions
        stage_names = {
            0: 'Stage 0: Healthy',
            1: 'Stage 1: At Risk', 
            2: 'Stage 2: Cardiovascular-Kidney-Metabolic Disease',
            3: 'Stage 3: Advanced Disease',
            4: 'Stage 4: Critical'
        }
        
        fig.update_layout(
            width=800,
            height=500,
            font=dict(family="Arial", size=12),
            showlegend=False,
            xaxis=dict(
                tickmode='array',
                tickvals=[0, 1, 2, 3, 4],
                ticktext=[stage_names.get(i, f'Stage {i}') for i in range(5)]
            )
        )
        
        return fig
    
    def create_ckm_risk_gauge(self, lead_level, ckm_score):
        """
        Create gauge chart for combined risk assessment
        
        Args:
            lead_level: Blood lead level (μg/dL)
            ckm_score: CKM risk score
            
        Returns:
            Plotly figure object
        """
        # Calculate combined risk
        lead_risk = min(lead_level / 10, 1) * 50  # Max 10 μg/dL as threshold
        ckm_risk = ckm_score / 4 * 50  # Max score of 4
        combined_risk = lead_risk + ckm_risk
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = combined_risk,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Combined Health Risk Score"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 25], 'color': "#27ae60"},
                    {'range': [25, 50], 'color': "#f1c40f"},
                    {'range': [50, 75], 'color': "#e67e22"},
                    {'range': [75, 100], 'color': "#e74c3c"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        
        fig.update_layout(
            width=500,
            height=400,
            font=dict(family="Arial", size=14)
        )
        
        return fig
    
    # =========================================================================
    # NETWORK VISUALIZATION
    # =========================================================================
    
    def create_metal_disease_network(self, associations, title="Metal-Disease Network"):
        """
        Create interactive network visualization
        
        Args:
            associations: Dict of {metal: {disease: correlation}}
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        # Prepare node data
        nodes = []
        edges = []
        
        # Add metals as nodes
        for metal in associations.keys():
            nodes.append({
                'id': metal,
                'label': metal,
                'type': 'metal',
                'color': METAL_COLORS.get(metal, '#4477AA')
            })
            
        # Add diseases and edges
        for metal, diseases in associations.items():
            for disease, corr in diseases.items():
                # Add disease node if not exists
                if disease not in [n['id'] for n in nodes]:
                    nodes.append({
                        'id': disease,
                        'label': disease,
                        'type': 'disease',
                        'color': '#AA3377'
                    })
                    
                # Add edge
                edges.append({
                    'from': metal,
                    'to': disease,
                    'weight': abs(corr),
                    'correlation': corr
                })
        
        # Create node positions (simple circular layout)
        node_list = [n['id'] for n in nodes]
        n_nodes = len(node_list)
        
        # Separate metals and diseases
        metals = [m for m in associations.keys()]
        diseases = set()
        for d in associations.values():
            diseases.update(d.keys())
        diseases = list(diseases)
        
        # Calculate positions
        positions = {}
        
        # Metals in inner circle
        for i, metal in enumerate(metals):
            angle = 2 * np.pi * i / len(metals)
            positions[metal] = (0.3 * np.cos(angle), 0.3 * np.sin(angle))
            
        # Diseases in outer circle
        for i, disease in enumerate(diseases):
            angle = 2 * np.pi * i / len(diseases)
            positions[disease] = (0.7 * np.cos(angle), 0.7 * np.sin(angle))
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        for edge in edges:
            fig.add_trace(go.Scatter(
                x=[positions[edge['from']][0], positions[edge['to']][0]],
                y=[positions[edge['from']][1], positions[edge['to']][1]],
                mode='lines',
                line=dict(
                    width=edge['weight'] * 5,
                    color=f"rgba(128, 128, 128, {0.3 + edge['weight'] * 0.5})"
                ),
                hoverinfo='text',
                hovertext=f"{edge['from']} → {edge['to']}: r={edge['correlation']:.3f}",
                showlegend=False
            ))
        
        # Add nodes
        for node in nodes:
            fig.add_trace(go.Scatter(
                x=[positions[node['id']][0]],
                y=[positions[node['id']][1]],
                mode='markers+text',
                marker=dict(
                    size=30 if node['type'] == 'metal' else 20,
                    color=node['color']
                ),
                text=[node['label']],
                textposition="middle center",
                textfont=dict(size=10),
                hovertemplate=f"{node['label']}<br>Type: {node['type']}<extra></extra>",
                showlegend=False
            ))
        
        fig.update_layout(
            title=title,
            width=800,
            height=700,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
        
        return fig
    
    # =========================================================================
    # PFAS VISUALIZATIONS
    # =========================================================================
    
    def create_pfas_comparison_chart(self, pfas_data, title="PFAS Compound Comparison"):
        """
        Create comparison chart for multiple PFAS compounds
        
        Args:
            pfas_data: Dict or DataFrame with PFAS data
            
        Returns:
            Plotly figure object
        """
        if isinstance(pfas_data, dict):
            # Convert to DataFrame
            df = pd.DataFrame([
                {'Compound': k, 'Target_Genes': v.get('n_targets', 0), 
                 'Toxicity_Score': v.get('toxicity_score', 0)}
                for k, v in pfas_data.items()
            ])
        else:
            df = pfas_data
            
        # Create grouped bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=df['Compound'],
            y=df['Target_Genes'],
            name='Target Genes',
            marker_color='#4477AA',
            yaxis='y'
        ))
        
        if 'Toxicity_Score' in df.columns:
            fig.add_trace(go.Bar(
                x=df['Compound'],
                y=df['Toxicity_Score'],
                name='Toxicity Score',
                marker_color='#EE6677',
                yaxis='y2'
            ))
        
        fig.update_layout(
            title=title,
            barmode='group',
            width=700,
            height=500,
            yaxis=dict(title="Number of Target Genes"),
            yaxis2=dict(
                title="Toxicity Score",
                overlaying='y',
                side='right'
            ),
            font=dict(family="Arial", size=12)
        )
        
        return fig
    
    # =========================================================================
    # TIME SERIES VISUALIZATION
    # =========================================================================
    
    def create_temporal_trend(self, df, time_col, value_cols, title="Temporal Trends"):
        """
        Create interactive time series visualization
        
        Args:
            df: DataFrame with time and value columns
            time_col: Time/date column
            value_cols: List of value columns to plot
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        # Melt dataframe for plotly
        plot_df = df[[time_col] + value_cols].melt(
            id_vars=time_col, 
            var_name='Variable',
            value_name='Value'
        )
        
        # Create line chart
        fig = px.line(
            plot_df,
            x=time_col,
            y='Value',
            color='Variable',
            title=title,
            markers=True
        )
        
        fig.update_layout(
            width=800,
            height=500,
            font=dict(family="Arial", size=12),
            xaxis_title=time_col,
            yaxis_title="Value"
        )
        
        return fig
    
    # =========================================================================
    # SAVE FUNCTIONS
    # =========================================================================
    
    def save_html(self, fig, filename):
        """Save figure as interactive HTML"""
        filepath = os.path.join(self.output_dir, filename)
        fig.write_html(filepath, include_plotlyjs='cdn')
        print(f"Interactive plot saved: {filepath}")
        return filepath
    
    def save_static(self, fig, filename, format='png', scale=2):
        """Save figure as static image"""
        filepath = os.path.join(self.output_dir, filename)
        fig.write_image(filepath, format=format, scale=scale)
        print(f"Static plot saved: {filepath}")
        return filepath


# =============================================================================
# MAIN FUNCTION - DEMO
# =============================================================================

def main():
    """Demo function showing all visualization capabilities"""
    
    viz = InteractiveVisualization(output_dir='output')
    
    print("=" * 60)
    print("Interactive Visualization Demo")
    print("=" * 60)
    
    # Try to load NHANES data - check CSV first, then XPT files
    nhanes_files = [
        'nhanes_data/nhanes_lead_blood.csv',
        'output/nhanes_lead_blood.csv',
        'output/nhanes_merged_data.csv'
    ]
    
    df = None
    for nhanes_file in nhanes_files:
        if os.path.exists(nhanes_file):
            print(f"\n📊 Loading NHANES data from {nhanes_file}...")
            df = pd.read_csv(nhanes_file)
            break
    
    # If no CSV, try loading from XPT files directly
    if df is None:
        xpt_dir = 'nhanes_data'
        if os.path.exists(xpt_dir):
            try:
                print(f"\n📊 Loading NHANES data from XPT files...")
                # Load metals data
                metals = pd.read_sas(os.path.join(xpt_dir, 'PBCD_L.xpt'), format='xport')
                
                # Load CBC data for blood counts
                cbc = pd.read_sas(os.path.join(xpt_dir, 'CBC_L.xpt'), format='xport')
                
                # Load biochemistry data
                bio = pd.read_sas(os.path.join(xpt_dir, 'BIOPRO_L.xpt'), format='xport')
                
                # Merge on SEQN
                df = metals.merge(cbc[['SEQN', 'LBXWBCSI', 'LBXLYPCT', 'LBXNEPCT', 'LBXEOPCT', 
                                       'LBXRBCSI', 'LBXHGB', 'LBXHCT']], on='SEQN', how='left')
                df = df.merge(bio[['SEQN', 'LBXSASSI', 'LBXSNASI', 'LBXSGB', 'LBXSGU']], 
                             on='SEQN', how='left')
                
                print(f"   Loaded {len(df)} samples with metals data")
            except Exception as e:
                print(f"   Warning: Could not load XPT files: {e}")
                df = None
    
    if df is not None:
        
        # Check for metal columns - NHANES naming conventions
        metal_cols = ['LBXBPB', 'LBXBCD', 'LBXTHG', 'LBXBSE', 'LBXBMN', 'LBXSASSI']
        available_metals = [c for c in metal_cols if c in df.columns]
        
        # Map to display names
        metal_display_names = {
            'LBXBPB': 'Lead (Pb)',
            'LBXBCD': 'Cadmium (Cd)', 
            'LBXTHG': 'Mercury (Hg)',
            'LBXBSE': 'Selenium (Se)',
            'LBXBMN': 'Manganese (Mn)',
            'LBXSASSI': 'Arsenic (As)'
        }
        
        if len(available_metals) >= 2:
            # 1. Correlation Heatmap
            print("\n1. Creating correlation heatmap...")
            fig1 = viz.create_interactive_correlation_heatmap(
                df, available_metals, 
                "Heavy Metal Correlation (NHANES)"
            )
            viz.save_html(fig1, 'interactive_correlation_heatmap.html')
        
        # 2. CKM Staging Plot (if available)
        if 'CKM_STAGE' in df.columns and 'LBXBPB' in df.columns:
            print("\n2. Creating CKM staging plot...")
            fig2 = viz.create_ckm_staging_plot(df)
            viz.save_html(fig2, 'interactive_ckm_staging.html')
        
        # 3. Dose-Response Curves
        outcome_cols = ['LBXGH', 'LBXGLU', 'LBXSATSI']  # HbA1c, Glucose, Iron
        for outcome in outcome_cols:
            if outcome in df.columns:
                print(f"\n3. Creating dose-response: Pb vs {outcome}...")
                fig3 = viz.create_dose_response_curve(
                    df, 'LBXBPB', outcome,
                    title=f"Blood Lead vs {outcome}"
                )
                safe_name = outcome.lower()
                viz.save_html(fig3, f'interactive_dose_response_{safe_name}.html')
        
        # 4. Risk Gauge
        if 'LBXBPB' in df.columns:
            lead_median = df['LBXBPB'].median()
            print(f"\n4. Creating risk gauge (Lead: {lead_median:.2f} μg/dL)...")
            fig4 = viz.create_ckm_risk_gauge(lead_median, 2.0)
            viz.save_html(fig4, 'interactive_risk_gauge.html')
            
    else:
        print(f"\n⚠️ NHANES data not found at {nhanes_file}")
        print("Creating demo visualizations with synthetic data...")
        
        # Create demo data
        np.random.seed(42)
        n = 200
        
        demo_data = pd.DataFrame({
            'LBXBPB': np.random.lognormal(0, 1, n),  # Lead
            'LBXIAS': np.random.lognormal(-0.5, 0.8, n),  # Arsenic
            'LBXBCD': np.random.lognormal(-1, 0.9, n),  # Cadmium
            'LBXBMN': np.random.lognormal(2, 0.5, n),  # Manganese
            'LBXGH': np.random.normal(5.5, 1, n),  # HbA1c
            'CKM_STAGE': np.random.choice([0, 1, 2, 3], n, p=[0.4, 0.3, 0.2, 0.1])
        })
        
        # Demo plots
        fig1 = viz.create_interactive_correlation_heatmap(
            demo_data, 
            ['LBXBPB', 'LBXIAS', 'LBXBCD', 'LBXBMN'],
            "Heavy Metal Correlation (Demo)"
        )
        viz.save_html(fig1, 'demo_correlation_heatmap.html')
        
        fig2 = viz.create_ckm_staging_plot(demo_data)
        viz.save_html(fig2, 'demo_ckm_staging.html')
        
        fig3 = viz.create_dose_response_curve(demo_data, 'LBXBPB', 'LBXGH')
        viz.save_html(fig3, 'demo_dose_response.html')
        
        fig4 = viz.create_ckm_risk_gauge(3.5, 2.5)
        viz.save_html(fig4, 'demo_risk_gauge.html')
        
        # Demo network
        demo_network = {
            'Pb': {'Cardiovascular': 0.25, 'Neurotoxicity': 0.35, 'CKD': 0.30},
            'As': {'CVD': 0.20, 'Cancer': 0.45, 'Diabetes': 0.28},
            'Cd': {'CKD': 0.38, 'Osteoporosis': 0.32, 'CVD': 0.22}
        }
        
        fig5 = viz.create_metal_disease_network(demo_network)
        viz.save_html(fig5, 'demo_metal_disease_network.html')
    
    print("\n" + "=" * 60)
    print("✅ Demo complete! Check the output directory.")
    print("=" * 60)


if __name__ == "__main__":
    main()
