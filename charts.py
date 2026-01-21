import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import io
import base64
import plotly.express as px
import plotly.graph_objects as go
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

def plot_pca_summary_section(pca_table):
    html = "<h2>PCA Component Summary (Top 10)</h2>"
    html += pca_table.head(10).to_html(classes='data', border=1, float_format="%.4f", index=False)
    return html

def plot_pca_scatter_section(df_pca, pca_table):
    html = "<h2>Global PCA Scatter (Static)</h2>"
    fig, ax = plt.subplots(figsize=(12, 8))
    pc1_var = pca_table.iloc[0]['Explained Variance']
    pc2_var = pca_table.iloc[1]['Explained Variance']
    
    sns.scatterplot(x='PC1', y='PC2', data=df_pca, alpha=0.6, ax=ax)
    ax.set_xlabel(f'PC1 ({pc1_var:.1%} var)')
    ax.set_ylabel(f'PC2 ({pc2_var:.1%} var)')
    
    ax.set_title("Stocks projected onto PC1 vs PC2 (Entire Period)")
    
    html += f'<img src="data:image/png;base64,{fig_to_base64(fig)}" style="width:100%">'
    return html

def plot_scree_section(pca_table):
    explained_var = pca_table['Explained Variance'].values
    cum_explained_var = pca_table['Cumulative Variance'].values
    
    html = "<h2>Scree Plot</h2>"
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x_range = range(1, len(explained_var) + 1)
    
    ax.bar(x_range, explained_var, alpha=0.5, label='Individual')
    ax.step(x_range, cum_explained_var, where='mid', label='Cumulative', color='red')
    
    ax.set_title("Explained Variance by Component")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Ratio")
    ax.legend()
    
    html += f'<img src="data:image/png;base64,{fig_to_base64(fig)}" style="width:100%">'
    return html

def plot_static_kmeans_section(df_pca_clustered, pca_table, kmeans):
    html = "<h2>Global PCA with K-Means Clustering</h2>"
    
    explained_variance = pca_table['Explained Variance'].values
    
    fig = plot_clustered_pca(df_pca_clustered, explained_variance, kmeans)
    
    html += f'<img src="data:image/png;base64,{fig_to_base64(fig)}" style="width:100%">'
    return html

def plot_clustering_section(df_pca_static, explained_var, n_clusters, results_time, timestamps):
    html = "<h2>PCA-Based Dynamic Clustering</h2>"
    
    cluster_changes = []
    for i in range(1, len(results_time)):
        prev_df = results_time[i-1]
        curr_df = results_time[i]
        common = list(set(prev_df.index) & set(curr_df.index))
        if len(common) > 0:
            changes = sum(1 for s in common if prev_df.loc[s, 'Cluster'] != curr_df.loc[s, 'Cluster'])
            cluster_changes.append(100 * changes / len(common))
        else:
            cluster_changes.append(0)
            
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(timestamps[1:], cluster_changes, marker='o', linestyle='-', color='blue')
    ax.set_title("PCA Cluster Instability (Churn Rate)")
    ax.set_ylabel("% Changed Cluster")
    ax.grid(True, alpha=0.3)
    
    html += f"<h3>PCA Stability</h3>"
    html += f'<img src="data:image/png;base64,{fig_to_base64(fig)}" style="width:100%">'
    
    last_df = results_time[-1]
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=last_df, palette='viridis', s=100, ax=ax2)
    ax2.set_title(f"PCA Clusters (Last Window: {timestamps[-1].date()})")
    html += f"<h3>Latest PCA Clusters</h3>"
    html += f'<img src="data:image/png;base64,{fig_to_base64(fig2)}" style="width:100%">'
    
    return [html] 

def plot_tsne_section(tsne_results, timestamps, stability_scores):
    if not tsne_results:
        return "<p>No t-SNE results generated.</p>"
        
    html = "<h2>t-SNE Based Dynamic Clustering</h2>"
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(timestamps, stability_scores, color='#e74c3c', marker='o', linewidth=2)
    ax.set_title('t-SNE Cluster Instability (Churn Rate)')
    ax.set_ylabel('% Changed Cluster')
    ax.grid(True, alpha=0.3)
    ax.axhline(np.mean(stability_scores), color='gray', linestyle='--', label='Average')
    ax.legend()
    
    html += f"<h3>t-SNE Stability</h3>"
    html += f'<img src="data:image/png;base64,{fig_to_base64(fig)}" style="width:100%">'
    
    last_df = tsne_results[-1]
    fig2, ax2 = plt.subplots(figsize=(12, 10))
    sns.scatterplot(data=last_df, x='x', y='y', hue='Cluster', palette='tab10', s=100, ax=ax2)
    ax2.set_title(f"t-SNE Map (Last Window: {timestamps[-1].date()})")
    
    html += f"<h3>Latest t-SNE Map</h3>"
    html += f'<img src="data:image/png;base64,{fig_to_base64(fig2)}" style="width:100%">'
    
    return html

def plot_clustered_pca(df_pca_results, explained_variance, kmeans):
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.scatterplot(
        x='PC1', 
        y='PC2', 
        hue='Cluster',
        data=df_pca_results, 
        alpha=0.7,
        s=100,
        palette='viridis',
        ax=ax
    )
    
    centers = kmeans.cluster_centers_
    ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=300, edgecolors='black', linewidths=2, label='Centroids')
    
    ax.set_title('PCA with KMeans Clustering')
    ax.set_xlabel(f'PC1: Market Component ({explained_variance[0]:.1%} variance)')
    ax.set_ylabel(f'PC2: Main Divergence Factor ({explained_variance[1]:.1%} variance)')
    ax.axvline(0, color='black', linestyle='--', alpha=0.3)
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3)
    
    ax.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    fig.tight_layout()
    return fig

def plot_model_comparison(metrics_df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    sns.barplot(x="Method", y="Silhouette", data=metrics_df, ax=axes[0], palette="viridis", hue="Method", legend=False)
    axes[0].set_title("Silhouette Score (Higher is Better)")
    axes[0].set_ylim(bottom=0) # Set bottom limit to 0 (or -1 if scores are negative)
    
    for p in axes[0].patches:
        if p.get_height() > 0:
            axes[0].annotate(f'{p.get_height():.4f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='bottom', xytext=(0, 5), textcoords='offset points')

    sns.barplot(x="Method", y="Calinski-Harabasz", data=metrics_df, ax=axes[1], palette="magma", hue="Method", legend=False)
    axes[1].set_title("Calinski-Harabasz Index (Higher is Better)")
    
    sns.barplot(x="Method", y="Davies-Bouldin", data=metrics_df, ax=axes[2], palette="rocket_r", hue="Method", legend=False)
    axes[2].set_title("Davies-Bouldin Index (Lower is Better)")

    plt.tight_layout()
    
    html = "<h2>Part 4: Clustering Quality Comparison</h2>"
    html += "<p>Comparison of clustering quality on the <b>original high-dimensional data</b>. "
    html += "Silhouette Score closer to 1.0 indicates clear, dense clusters. Scores near 0 indicate overlapping clusters (common in finance).</p>"
    html += f'<img src="data:image/png;base64,{fig_to_base64(fig)}" style="width:100%">'
    return html

def get_static_dbscan_figure(df_res):
    df_plot = df_res.copy()
    df_plot['Cluster_Str'] = df_plot['Cluster'].astype(str)
    
    df_plot['Cluster_Label'] = df_plot['Cluster'].apply(lambda x: 'Noise' if x == -1 else f'Cluster {x}')
    
    fig = px.scatter(
        df_plot, 
        x='x', 
        y='y',
        color='Cluster_Label',
        hover_name=df_plot.index,
        title='Static t-SNE + DBSCAN Structure (Global)',
        color_discrete_sequence=px.colors.qualitative.G10
    )
    
    fig.update_traces(marker=dict(size=6, opacity=0.8, line=dict(width=0.5, color='DarkSlateGrey')))
    fig.update_layout(
        template='plotly_white',
        legend_title_text='Group',
        height=600
    )
    
    return fig

def get_static_tsne_figure(df_coords):
    fig = px.scatter(
        df_coords, 
        x='x', 
        y='y',
        hover_name=df_coords.index, # Nazwa spółki po najechaniu
        title='Static t-SNE Geometry (Raw Projection)',
        opacity=0.7
    )
    
    fig.update_traces(marker=dict(size=6, color='#3366CC', line=dict(width=0.5, color='white')))
    
    fig.update_layout(
        template='plotly_white',
        height=700,
        xaxis_title="t-SNE Dimension 1",
        yaxis_title="t-SNE Dimension 2"
    )
    
    return fig