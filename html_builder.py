import datetime
import numpy as np
import pandas as pd
import plotly.io as pio

import charts 

STYLE_DESC = "font-family: 'Times New Roman', serif; font-size: 1.1em; text-align: justify; margin-bottom: 20px; color: #333;"

def build_header(X_filtered):
    html_parts = []
    html_parts.append(f"<h1>Clustering Report for stock data</h1>")
    html_parts.append(f"<p>Pawel Richert 465470 </p>")
    html_parts.append(
        "<p>This project aims to optimize the identification of tradable asset pairs "
        "by reducing the computational complexity of the search space. "
        "In a universe of O(<var>N</var>) stocks, a brute-force search for "
        "cointegrated pairs requires O(<var>N</var><sup>2</sup>) operations, "
        "which is computationally expensive and prone to false positives. "
        "By applying clustering, we reduce this complexity by grouping stocks "
        "into clusters based on similar variability. We employ two distinct approaches: "
        "Linear Dimensionality Reduction (PCA) combined with K-Means clustering  "
        "and t-SNE combined with DBSCAN, which is present to compare the results. "
        "This paper also includes a comparative analysis of cluster evolution over time.</p>"
    )
    html_parts.append(f"<p>Generated: {datetime.datetime.now().isoformat()} Analysis restricted to {len(X_filtered.columns)} stocks.</p>")
    return html_parts

def build_static_kmeans(df_pca, kmeans_model):
    html_parts = []
    html_parts.append("<h1>Part 2: Static Clustering (Global Snapshot)</h1>")
    html_parts.append(f"""
    <div style="{STYLE_DESC}">
    <b>Global Market Segmentation via K-Means.</b><br>
    This section applies K-Means clustering to the global Principal Components calculated over the entire timeframe.
    </div>
    """)
    
    html_parts.append(charts.plot_static_kmeans(df_pca, kmeans_model))
    return html_parts

def build_dynamic_kmeans(results_pca, times_pca):
    html_parts = []
    html_parts.append("<h1>Part 2b: Dynamic Clustering (K-Means with PCA)</h1>")
    html_parts.append(f"""
    <div style="{STYLE_DESC}">
    <b>Dynamic Evolution (Rolling Window).</b><br>
    Dynamic clustering allows us to distinguish between temporary market noise and robust, persistent relationships.
    <p>PCA + K-Means shows higher instability than t-SNE + DBSCAN because K-Means
    is a centroid-based algorithm that forces every data point, including noise
    and outliers, into a cluster. In volatile stock market data, even minor
    price fluctuations shift the centroids, causing frequent reassignments and
    high churn.</p>
    </div>
    """)
    
    html_parts.append(charts.build_dynamic_report(
        results_pca, times_pca, x_col='PC1', y_col='PC2', method_name="PCA K-Means"
    ))
    return html_parts

def build_static_dbscan(df_res):
    fig = charts.get_plotly_figure(df_res, x_col='x', y_col='y', cluster_col='Cluster', title="Static t-SNE + DBSCAN")
    chart_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
    
    return f"""
    <div class="section">
        <h2>Static t-SNE + DBSCAN Analysis</h2>
        <p>Global clustering structure using Density-Based Spatial Clustering.</p>
        {chart_html}
    </div>
    """

def build_tsne_evolution(results_tsne, times_tsne):
    html_parts = []
    html_parts.append("<h1>Part 3: Dynamic Clustering (DBSCAN on t-SNE)</h1>")
    html_parts.append(f"""
    <div style="{STYLE_DESC}">
    <b>Non-linear Manifold Learning (Rolling Window).</b><br>
    Analyzing how t-SNE manifolds and DBSCAN clusters evolve over time.
    <p>DBSCAN performs better because it is a density-based algorithm that identifies clusters
    of any shape and effectively handles outliers. Unlike K-Means, which forces every point
    into a group, DBSCAN labels irregular or isolated data points as noise. When paired with
    t-SNE ability to preserve non-linear relationships, it creates more distinct and stable
    boundaries, resulting in superior separation and higher clustering quality metrics.</p>
    </div>
    """)
    html_parts.append(charts.build_dynamic_report(
        results_tsne, times_tsne, x_col='x', y_col='y', method_name="t-SNE DBSCAN"
    ))
    return html_parts

def build_quality_metrics(df_metrics):
    html_parts = []
    html_parts.append("<h1>Part 4: Method Comparison</h1>")
    html_parts.append(f"""
            <p>
                The <b>Silhouette Score</b> suggests that the linear PCA approach creates slightly better-defined groups, 
                while the t-SNE method likely suffers from overlapping clusters or "noise" assignments. 
                The <b>Calinski-Harabasz Index</b> shows a significant advantage for PCA + K-Means, which indicates 
                that the PCA-based clusters are much denser and more distinct—a factor crucial for identifying reliable cointegrated pairs. 
                While the <b>Davies-Bouldin Index</b> for t-SNE + DBSCAN shows a lower score (traditionally suggesting better clustering), 
                it may instead indicate that DBSCAN is creating very tight but isolated clusters while treating much of the data as noise.
            </p>
        """)
    html_parts.append(charts.build_quality_report(df_metrics))
    return html_parts

def build_pairs_analysis(html_pca_pairs, html_tsne_pairs):
    html_parts = []
    html_parts.append("<h1>Part 5: Top Similar Stocks Analysis</h1>")
    html_parts.append(f"<div style='{STYLE_DESC}'>Identifying closest pairs in reduced dimensions.</div>")
    
    if html_pca_pairs: html_parts.append(html_pca_pairs)
    if html_tsne_pairs: html_parts.append(html_tsne_pairs)
        
    return html_parts

def build_pca_stats(pca_table, top_n=10):
    subset = pca_table.head(top_n)
    table_html = subset.to_html(classes='data', float_format="{:.4f}".format, border=0)
    
    return f"""
    <div class="section" style="{STYLE_DESC}">
        <h2>Top {top_n} Principal Components (Explained Variance)</h2>
        <p>
            A PC1 of 34% means that the first principal component captures
            over one-third of the total variance across all stocks. After 
            filtering out outliers, this indicates that the data has a strong 
            common signal representing the "Market Factor" or the general movement of the broad market.
        </p>
        {table_html}
    </div>
    """

def generate_trustworthiness_section(score_pca, score_tsne):
    return f"""
    <div>
        <h2>Trustworthiness (Topology Preservation)</h2>
        <p>PCA Score: <b>{score_pca:.4f}</b> | t-SNE Score: <b>{score_tsne:.4f}</b></p>
        <p>
            The trustworthiness analysis reveals a significant disparity between the two methods, with PCA scoring 0.5305 and t-SNE scoring 0.7394.<br><br>
            This gap demonstrates that the underlying stock market data possesses a complex, non-linear structure that linear PCA struggles to represent accurately, effectively distorting local neighborhoods nearly half the time.
        </p>
    </div>
    """

def build_pairs_html(pairs_data, method_name="PCA"):
    top_n = len(pairs_data)
    html = f"<h2>Top {top_n} Most Similar Pairs ({method_name})</h2>"
    html += "<table class='data' style='width:100%; border-collapse: collapse;'><thead><tr style='background-color: #f2f2f2;'><th>Stock A</th><th>Stock B</th><th>Distance</th><th>Correlation</th></tr></thead><tbody>"
    
    for item in pairs_data:
        corr = item['correlation']
        corr_str = f"{corr:.4f}" if not pd.isna(corr) else "N/A"
        html += f"<tr><td>{item['stock_a']}</td><td>{item['stock_b']}</td><td>{item['distance']:.4f}</td><td>{corr_str}</td></tr>"
        
    html += "</tbody></table>"
    return html

def save_report(html_parts, filename="full_market_report.html"):
    body = "\n<hr>".join(html_parts)
    html = f"""
    <html><head><title>Market Clustering Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        table.data {{ border-collapse: collapse; width: 100%; }}
        table.data th, table.data td {{ border: 1px solid #ddd; padding: 8px; }}
        h1 {{ color: #2c3e50; margin-top: 60px; }}
    </style></head><body>{body}</body></html>
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nReport saved to: {filename}")