import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import plotly.express as px

plt.switch_backend('Agg')
sns.set_style("whitegrid")

def _render_fig(fig, title=None):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    header = f"<h2>{title}</h2>" if title else ""
    return f'{header}<img src="data:image/png;base64,{img}" style="width:100%; border:1px solid #ddd; margin-bottom:20px;">'

def plot_static_kmeans(df_pca, kmeans_model):
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.scatterplot(
        x='PC1', y='PC2', hue='Cluster', data=df_pca, 
        alpha=0.7, s=100, palette='viridis', ax=ax
    )
    
    centers = kmeans_model.cluster_centers_
    ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=300, 
               edgecolors='black', linewidths=2, label='Centroids')
    
    ax.set_title('Global PCA with K-Means Clustering')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()
    
    return _render_fig(fig, title="Global PCA with K-Means Clustering")

def build_dynamic_report(results_list, timestamps, x_col, y_col, method_name):
    if not results_list:
        return f"<p>No results for {method_name}</p>"
    
    html = ""
    churn_rates = []
    for i in range(1, len(results_list)):
        prev_df, curr_df = results_list[i-1], results_list[i]
        common = list(set(prev_df.index) & set(curr_df.index))
        if common:
            diffs = sum(1 for s in common if prev_df.loc[s, 'Cluster'] != curr_df.loc[s, 'Cluster'])
            churn_rates.append(100 * diffs / len(common))
        else:
            churn_rates.append(0)

    fig1, ax1 = plt.subplots(figsize=(12, 5))
    plot_times = timestamps[1:] if len(timestamps) > len(churn_rates) else timestamps[:len(churn_rates)]
    
    ax1.plot(plot_times, churn_rates, marker='o', linestyle='-', color='tab:blue')
    ax1.set_title(f"{method_name} Cluster Instability (Churn Rate)")
    ax1.set_ylabel("% Changed Cluster")
    ax1.grid(True, alpha=0.3)
    
    html += _render_fig(fig1, title=f"{method_name} Stability")

    last_df = results_list[-1]
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    sns.scatterplot(data=last_df, x=x_col, y=y_col, hue='Cluster', palette='viridis', s=100, ax=ax2)
    ax2.set_title(f"Latest {method_name} Map ({timestamps[-1].date()})")
    
    html += _render_fig(fig2, title=f"Latest {method_name} Clusters")
    
    return html

def build_quality_report(metrics_df):
    if metrics_df is None: return ""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [("Silhouette", "viridis", "Higher is Better"), 
               ("Calinski-Harabasz", "magma", "Higher is Better"), 
               ("Davies-Bouldin", "rocket_r", "Lower is Better")]
    
    for i, (col, pal, desc) in enumerate(metrics):
        sns.barplot(x="Method", y=col, data=metrics_df, ax=axes[i], palette=pal, hue="Method", legend=False)
        axes[i].set_title(f"{col}\n({desc})")
        for p in axes[i].patches:
            if p.get_height() > 0:
                axes[i].annotate(f'{p.get_height():.2f}', 
                                 (p.get_x() + p.get_width() / 2., p.get_height()),
                                 ha='center', va='bottom')
        
    plt.tight_layout()
    return _render_fig(fig, title="Clustering Quality Comparison")

def get_plotly_figure(df, x_col, y_col, cluster_col=None, title="Market Map"):
    color = None
    if cluster_col and cluster_col in df.columns:
        df = df.copy()
        df['Label'] = df[cluster_col].apply(lambda x: 'Noise' if x == -1 else f'Cluster {x}')
        color = 'Label'

    fig = px.scatter(
        df, x=x_col, y=y_col, color=color,
        hover_name=df.index, title=title,
        color_discrete_sequence=px.colors.qualitative.G10,
        opacity=0.8
    )
    fig.update_layout(template='plotly_white', height=600)
    fig.update_traces(marker=dict(size=6, line=dict(width=0.5, color='DarkSlateGrey')))
    
    return fig