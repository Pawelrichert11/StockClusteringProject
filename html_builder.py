import datetime
import numpy as np
import pandas as pd
import plotly.io as pio
from scipy.spatial.distance import pdist, squareform

from charts import (
    plot_model_comparison,  
    plot_pca_summary_section,
    plot_pca_scatter_section,
    plot_scree_section,
    plot_clustering_section, 
    plot_tsne_section,
    plot_static_kmeans_section,
    get_static_dbscan_figure,
    get_static_tsne_figure
)
STYLE_DESC = "font-family: 'Times New Roman', serif; font-size: 1.1em; text-align: justify; margin-bottom: 20px; color: #333;"

def build_pca(df_pca, pca_table):
    html_parts = []
    html_parts.append("<h1>Part 1: Global Market Structure (PCA)</h1>")
    
    html_parts.append(f"""
    <div style="{STYLE_DESC}">
    <b>Table 1. Explained Variance by Principal Components.</b><br>
    The table below quantifies the information retention of the dimensionality reduction process. 
    'Explained Variance' denotes the proportion of total dataset variance captured by each orthogonal component. 
    The 'Cumulative Variance' indicates the total information retained by the top <i>k</i> components, serving as a metric for the efficiency of the linear projection.
    </div>
    """)
    html_parts.append(plot_pca_summary_section(pca_table))
    
    html_parts.append(f"""
    <div style="{STYLE_DESC}">
    <b>Figure 1. Eigenvalue Spectrum (Scree Plot).</b><br>
    The plot displays the eigenvalues associated with each principal component in descending order. 
    The 'elbow point' in the curve suggests the optimal number of latent factors required to describe the market structure without overfitting to noise. 
    </div>
    """)
    html_parts.append(plot_scree_section(pca_table))

    html_parts.append(f"""
    <div style="{STYLE_DESC}">
    <b>Figure 2. Projection of Asset Universe onto the First Two Principal Components.</b><br>
    The scatter plot visualizes the asset space reduced to a 2D plane defined by PC1 and PC2.
    PC1 (x-axis), typically represents the market beta (systematic risk). 
    PC2 (y-axis) often distinguishes between cyclical and defensive sectors or other dominant factor exposures.
    Clustering of points indicates high similarity in historical return profiles in the linear space.
    </div>
    """)
    html_parts.append(plot_pca_scatter_section(df_pca, pca_table))
    
    return html_parts

def build_kmeans(results_pca, times_pca, df_pca, pca_table, n_clusters):
    html_parts = []
    html_parts.append("<h1>Part 2: Dynamic Clustering (K-Means with PCA)</h1>")
    
    html_parts.append(f"""
    <div style="{STYLE_DESC}">
    <b>Figure 3. Temporal Stability and Cluster Assignments (PCA-KMeans).</b><br>
    The analysis below presents the dynamic evolution of clusters identified by the K-Means algorithm ($k={n_clusters}$) applied to the principal components.
    <br><br>
    (a) <b>Churn Rate (Stability)</b>: The time-series graph depicts the percentage of assets migrating between clusters between consecutive time windows. 
    A high churn rate indicates a regime shift or structural instability in the market.
    <br>
    (b) <b>Cluster Map</b>: The scatter plot shows the final grouping of assets. Colors represent distinct clusters. 
    The spatial separation indicates the effectiveness of the algorithm in identifying distinct market segments.
    </div>
    """)
    
    html_parts.extend(plot_clustering_section(df_pca, pca_table, n_clusters, results_pca, times_pca))
    
    return html_parts

def build_header(X_filtered):
    html_parts = []
    html_parts.append(f"<h1>Comprehensive Market Clustering Report</h1>")
    html_parts.append(f"<p>Generated: {datetime.datetime.now().isoformat()}</p>")
    html_parts.append(f"<p>Analysis restricted to {len(X_filtered.columns)} highly correlated stocks (Outliers removed automatically).</p>")
    return html_parts

def build_tsne_evolution(results_tsne, times_tsne, scores_tsne):
    html_parts = []
    html_parts.append("<h1>Part 3: Dynamic Clustering (DBSCAN on t-SNE)</h1>")
    
    if len(results_tsne) > 0:
        n_found = len(set(results_tsne[-1]['Cluster'])) - (1 if -1 in results_tsne[-1]['Cluster'].values else 0)
        noise_pct = (results_tsne[-1]['Cluster'] == -1).mean() * 100
        stats_info = f"DBSCAN found {n_found} clusters in the last window. Noise points: {noise_pct:.1f}%"
    else:
        stats_info = "No results."
    
    html_parts.append(f"""
    <div style="{STYLE_DESC}">
    <b>Figure 4. Non-linear Manifold Learning and Density Clustering (t-SNE + DBSCAN).</b><br>
    This section utilizes t-Distributed Stochastic Neighbor Embedding (t-SNE) to capture non-linear dependencies, followed by DBSCAN for cluster identification.
    <br><br>
    (a) <b>Stability Analysis</b>: The plot illustrates the cluster churn rate. Fluctuations reflect evolving non-linear relationships.
    <br>
    (b) <b>Manifold Projection</b>: The 2D map prioritizes local neighborhood preservation. 
    <i>{stats_info}</i>
    </div>
    """)
    
    html_parts.append(plot_tsne_section(results_tsne, times_tsne, scores_tsne))
    
    return html_parts

def build_quality_metrics(df_metrics):
    if df_metrics is not None:
        html_parts = []
        html_parts.append("<h1>Part 4: Method Comparison & Quality</h1>")
        
        html_parts.append(f"""
        <div style="{STYLE_DESC}">
        <b>Figure 5. Comparative Analysis of Internal Clustering Validation Metrics.</b><br>
        This panel compares the performance of PCA-KMeans (Linear) vs. t-SNE-DBSCAN (Non-linear) using three standard validation indices:<br>
        1. <b>Silhouette Score</b>: Measures cluster cohesion and separation (Range: -1 to 1; Higher is better).<br>
        2. <b>Calinski-Harabasz Index</b>: Ratio of between-cluster dispersion to within-cluster dispersion (Higher is better).<br>
        3. <b>Davies-Bouldin Index</b>: Average similarity measure of each cluster with its most similar cluster (Lower is better).
        </div>
        """)
        
        html_parts.append(plot_model_comparison(df_metrics))
        return html_parts

def save_report(html_parts, filename="full_market_report.html"):
    html = build_final_html(html_parts)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nReport saved to: {filename}")

def build_pairs_analysis(df_pca, results_tsne, df_returns_filtered, html_pca_pairs, html_tsne_pairs):
    html_parts = []
    html_parts.append("<h1>Part 5: Top Similar Stocks Analysis</h1>")

    html_parts.append(f"""
    <div style="{STYLE_DESC}">
    <b>Table 3. Pairwise Similarity Analysis in Reduced Feature Space.</b><br>
    The tables below identify asset pairs exhibiting the minimal Euclidean distance within the projected lower-dimensional space. 
    The <b>'Actual Correlation'</b> column reports the historical Pearson correlation coefficient computed on the original return series. 
    High historical correlation validates the geometric proximity found by the dimensionality reduction algorithm.
    </div>
    """)

    try:
        pca_coords = df_pca.iloc[:, :5] 
        html_parts.append(html_pca_pairs)
    except Exception as e:
        print(f"Error generating PCA pairs: {e}")

    try:
        if len(results_tsne) > 0:
            last_tsne_df = results_tsne[-1]
            tsne_coords = last_tsne_df[['x', 'y']]
            html_parts.append(html_tsne_pairs)
    except Exception as e:
        print(f"Error generating t-SNE pairs: {e}")
        
    return html_parts

def build_final_html(parts):
    body = "\n<hr>".join(parts)
    return f"""
    <html>
    <head>
        <title>Market Clustering Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; max-width: 1200px; margin: 0 auto; padding: 20px; }}
            img {{ max-width: 100%; height: auto; border: 1px solid #ddd; margin: 20px 0; }}
            table.data {{ border-collapse: collapse; width: 100%; font-family: 'Times New Roman', serif; }}
            table.data th, table.data td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            table.data th {{ background-color: #f2f2f2; }}
            h1 {{ color: #2c3e50; margin-top: 60px; font-family: 'Arial', sans-serif; }}
            h2 {{ color: #34495e; border-bottom: 2px solid #eee; padding-bottom: 10px; margin-top: 40px; font-family: 'Arial', sans-serif; }}
        </style>
    </head>
    <body>
        {body}
    </body>
    </html>
    """

def generate_trustworthiness_section(score_pca, score_tsne):
    color_pca = "green" if score_pca > score_tsne else "black"
    color_tsne = "green" if score_tsne >= score_pca else "black"
    weight_pca = "bold" if score_pca > score_tsne else "normal"
    weight_tsne = "bold" if score_tsne >= score_pca else "normal"
    
    html = r"""
    <div style='background-color:
        <h2 style='margin-top: 0;'>Neighborhood Preservation Analysis (Trustworthiness)</h2>
        <div style="font-family: 'Times New Roman', serif; font-size: 1.1em; text-align: justify; margin-bottom: 20px; color: #333;">
        <b>Table 2. Trustworthiness Metric Comparison ($0 \le T \le 1$).</b><br>
        This metric quantifies the extent to which the dimensionality reduction technique preserves the local neighborhood topology of the original high-dimensional feature space. 
        A score closer to 1.0 indicates minimal distortion of local neighborhoods. This is critical for assessing whether the visual clusters represent true statistical proximity.
        </div>

        <table class="data" style="width: 50%; margin: 20px 0;">
            <thead>
                <tr>
                    <th>Method</th>
                    <th>Score (Closer to 1.0 is better)</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><b>PCA</b> (Global Structure)</td>
                    <td style="font-size: 1.2em; color: {color_pca}; font-weight: {weight_pca};">{score_pca:.4f}</td>
                </tr>
                <tr>
                    <td><b>t-SNE</b> (Local Manifold)</td>
                    <td style="font-size: 1.2em; color: {color_tsne}; font-weight: {weight_tsne};">{score_tsne:.4f}</td>
                </tr>
            </tbody>
        </table>
        <p style='font-family: "Times New Roman", serif; font-size: 1.0em; font-style: italic;'>
            Interpretation: A higher t-SNE score suggests that non-linear manifold learning is more effective at capturing local industry-specific relationships than linear PCA.
        </p>
    </div>
    """
    return html.format(score_pca=score_pca, score_tsne=score_tsne, color_pca=color_pca, color_tsne=color_tsne, weight_pca=weight_pca, weight_tsne=weight_tsne)

def build_pairs_html(pairs_data, method_name="PCA"):
    top_n = len(pairs_data)
    
    html = f"<h2>Top {top_n} Most Similar Pairs ({method_name})</h2>"
    
    html += """
    <table class="data" style="width:100%; border-collapse: collapse;">
        <thead>
            <tr style="background-color: #f2f2f2;">
                <th>Rank</th>
                <th>Stock A</th>
                <th>Stock B</th>
                <th>Distance (Lower is better)</th>
                <th>Actual Correlation (Higher is better)</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for rank, item in enumerate(pairs_data, 1):
        corr_val = item['correlation']
        
        if pd.isna(corr_val):
            corr_str = "N/A"
        else:
            corr_color = "green" if corr_val > 0.8 else "black"
            corr_str = f"<span style='color:{corr_color}; font-weight:bold'>{corr_val:.4f}</span>"
            
        html += f"""
            <tr>
                <td>{rank}</td>
                <td><b>{item['stock_a']}</b></td>
                <td><b>{item['stock_b']}</b></td>
                <td>{item['distance']:.4f}</td>
                <td>{corr_str}</td>
            </tr>
        """
        
    html += "</tbody></table>"
    return html

def calculate_top_pairs(coords_df, original_returns, top_n=10):
    dist_matrix = squareform(pdist(coords_df.values))
    np.fill_diagonal(dist_matrix, np.inf)
    
    pairs = []
    tickers = coords_df.index.tolist()
    
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            dist = dist_matrix[i, j]
            pairs.append((tickers[i], tickers[j], dist))
            
    pairs.sort(key=lambda x: x[2])
    best_pairs = pairs[:top_n]
    
    results = []
    for t1, t2, dist in best_pairs:
        correlation = np.nan
        try:
            if original_returns is not None:
                correlation = original_returns[t1].corr(original_returns[t2])
        except Exception:
            pass
            
        results.append({
            'stock_a': t1,
            'stock_b': t2,
            'distance': dist,
            'correlation': correlation
        })
        
    return results

def build_static_kmeans(df_pca, pca_table, kmeans_model):
    html_parts = []
    html_parts.append("<h1>Part 2: Static Clustering (Global Snapshot)</h1>")
    
    html_parts.append(f"""
    <div style="{STYLE_DESC}">
    <b>Figure 3. Global Market Segmentation via K-Means.</b><br>
    This section applies K-Means clustering to the global Principal Components calculated over the entire timeframe.
    Red 'X' markers denote the centroids of each cluster. 
    This view assumes the market structure is constant over the analyzed period and groups stocks based on their overall similarity.
    </div>
    """)
    
    html_parts.append(plot_static_kmeans_section(df_pca, pca_table, kmeans_model))
    return html_parts

def build_static_dbscan(df_res):
    fig = get_static_dbscan_figure(df_res)
    chart_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
    
    n_clusters = len(set(df_res['Cluster'])) - (1 if -1 in df_res['Cluster'] else 0)
    noise_count = list(df_res['Cluster']).count(-1)
    noise_pct = noise_count / len(df_res) * 100
    
    html = f"""
    <div class="section">
        <h2>Static t-SNE + DBSCAN Analysis</h2>
        <p>
            This analysis runs t-SNE on the entire dataset (filtered) at once, followed by DBSCAN clustering.
            It reveals the global, persistent structure of the market.
        </p>
        <div class="stats-box">
            <ul>
                <li><strong>Total Clusters Found:</strong> {n_clusters}</li>
                <li><strong>Noise Points (Outliers):</strong> {noise_count} ({noise_pct:.1f}%)</li>
                <li><strong>Parameters:</strong> Perplexity=30, Eps=0.5, Min_Samples=5</li>
            </ul>
        </div>
        {chart_html}
    </div>
    """
    return html

def build_static_tsne_only(df_coords):
    fig = get_static_tsne_figure(df_coords)
    chart_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
    
    html = f"""
    <div class="section">
        <h2>Static t-SNE Projection (Geometry)</h2>
        <p>
            This chart shows the raw 2D projection of the stock market using t-SNE based on 
            log-returns history. Stocks close to each other behave similarly. 
            No clustering algorithm has been applied yet - this is the pure "shape" of the market.
        </p>
        {chart_html}
    </div>
    """
    return html