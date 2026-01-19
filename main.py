import pandas as pd
import numpy as np
from tqdm import tqdm
import glob
import os
import datetime

# --- IMPORTS ---
from pca import preprocess_data, run_pca

# Import from your separate files
from kmeans import analyze_clusters_over_time  # PCA-based clustering
from tsne import analyze_tsne_clusters         # t-SNE-based clustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from charts import plot_model_comparison, plot_tsne_quality 

from charts import (
    plot_corr_matrix_section,
    plot_pca_summary_section,
    plot_pca_scatter_section,
    plot_scree_section,
    plot_clustering_section, 
    plot_tsne_section,       
    build_final_html
)

def load_stock_data(data_path, stock_limit=None, start_date='2012-01-01', min_dollar_volume=1000000, min_volatility=0.005):
    files = glob.glob(data_path)
    price_data = {}
    volume_data = {} 
    
    files.sort()
    print(f"Scanning {len(files)} files...")
    
    for file_path in tqdm(files, desc="Loading Data"):
        if stock_limit and len(price_data) >= stock_limit:
            break
        try:
            ticker = os.path.basename(file_path).split('.')[0]
            df = pd.read_csv(file_path)
            df.columns = [c.lower() for c in df.columns]
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            if df.index[-1] < pd.to_datetime(start_date):
                continue 
                
            price_data[ticker] = df['close']
            if 'volume' in df.columns:
                volume_data[ticker] = df['volume']
        except Exception:
            continue

    print("Aligning data...")
    df_prices = pd.DataFrame(price_data)
    df_vol = pd.DataFrame(volume_data)
    
    df_prices = df_prices.loc[start_date:]
    df_vol = df_vol.loc[start_date:]
    
    if df_prices.empty: return df_prices

    df_prices.dropna(axis=0, how='all', inplace=True)
    df_prices.ffill(inplace=True)
    df_prices.dropna(axis=1, how='any', inplace=True) 
    
    # Filter
    valid_stocks = []
    if not df_vol.empty:
        df_vol = df_vol.reindex(df_prices.index).ffill()
        for col in df_prices.columns:
            if col in df_vol.columns:
                avg_val = (df_prices[col] * df_vol[col]).mean()
                ret_std = df_prices[col].pct_change().std()
                if avg_val > min_dollar_volume and ret_std > min_volatility:
                    valid_stocks.append(col)
        df_prices = df_prices[valid_stocks]
    
    print(f"Final Dataset: {df_prices.shape[1]} stocks.")
    return df_prices


if __name__ == "__main__":
    # --- CONFIG ---
    DATA_PATH = "Stocks/*.txt" 
    STOCK_LIMIT = 400
    N_COMPONENTS = 50  
    N_CLUSTERS = 5
    
    # Clustering Config
    WINDOW_SIZE = 60
    STEP_SIZE = 10 

    df_prices = load_stock_data(DATA_PATH, stock_limit=STOCK_LIMIT)

    html_parts = []
    html_parts.append(f"<h1>Comprehensive Market Clustering Report</h1>")
    html_parts.append(f"<p>Generated: {datetime.datetime.now().isoformat()}</p>")

    if not df_prices.empty:
        X_scaled, tickers_index, df_returns = preprocess_data(df_prices)
        
        corr_with_market = X_scaled.corrwith(X_scaled.mean(axis=1)).abs()
        high_corr_stocks = corr_with_market[corr_with_market > corr_with_market.median()].index
        X_filtered = X_scaled[high_corr_stocks]
        df_returns_filtered = df_returns[high_corr_stocks]
        
        html_parts.append(f"<p>Analysis based on {len(high_corr_stocks)} highly correlated stocks.</p>")

        html_parts.append("<h1>Part 1: Global Market Structure (PCA)</h1>")
        html_parts.append(plot_corr_matrix_section(X_filtered, corr_with_market))
        
        df_pca, explained_var, cum_explained_var, pca_table = run_pca(X_filtered, X_filtered.columns.tolist(), N_COMPONENTS)
        html_parts.append(plot_pca_summary_section(pca_table))
        html_parts.append(plot_scree_section(explained_var, cum_explained_var))
        html_parts.append(plot_pca_scatter_section(df_pca, explained_var))
        
        print("\n--- Running PCA Cluster Evolution (kmeans.py) ---")
        results_pca, times_pca = analyze_clusters_over_time(
            df_returns_filtered, window_size=WINDOW_SIZE, step_size=STEP_SIZE, n_clusters=N_CLUSTERS
        )
        html_parts.append("<h1>Part 2: Dynamic Clustering (K-Means with PCA)</h1>")
        html_parts.extend(plot_clustering_section(df_pca, explained_var, N_CLUSTERS, results_pca, times_pca))

        print("\n--- Running t-SNE Cluster Evolution (tsne.py) ---")
        results_tsne, times_tsne, scores_tsne = analyze_tsne_clusters(
            df_returns_filtered, window_size=WINDOW_SIZE, step_size=STEP_SIZE, n_clusters=N_CLUSTERS, perplexity=30
        )
        html_parts.append("<h1>Part 3: Dynamic Clustering (K-Means with t-SNE)</h1>")
        html_parts.append(plot_tsne_section(results_tsne, times_tsne, scores_tsne))

        print("\n--- Calculating Comparison Metrics (PCA vs t-SNE) ---")

        if len(results_pca) > 0 and len(results_tsne) > 0:
            try:
                last_labels_pca = results_pca[-1]['Cluster']
                last_labels_tsne = results_tsne[-1]['Cluster']
                
                last_date = results_pca[-1]['date'] if 'date' in results_pca[-1] else times_pca[-1]

                if last_date in df_returns_filtered.index:
                    end_idx = df_returns_filtered.index.get_loc(last_date)
                    start_idx = max(0, end_idx - WINDOW_SIZE)
                    
                    X_window = df_returns_filtered.iloc[start_idx:end_idx]
                    
                    X_eval = (X_window - X_window.mean()) / X_window.std()
                    X_eval = X_eval.fillna(0).T 

                    def get_metrics(X, labels, method_name):
                        return {
                            "Method": method_name,
                            "Silhouette": silhouette_score(X, labels),
                            "Calinski-Harabasz": calinski_harabasz_score(X, labels),
                            "Davies-Bouldin": davies_bouldin_score(X, labels)
                        }

                    metrics_data = []
                    metrics_data.append(get_metrics(X_eval, last_labels_pca, "PCA + K-Means"))
                    metrics_data.append(get_metrics(X_eval, last_labels_tsne, "t-SNE + K-Means"))
                    
                    df_metrics = pd.DataFrame(metrics_data)
                    print("\nFinal Metrics Comparison (Last Window):")
                    print(df_metrics)

                    html_parts.append("<h1>Part 4: Method Comparison & Quality</h1>")
                    html_parts.append(plot_model_comparison(df_metrics))
                    
                    if isinstance(scores_tsne, list) and len(scores_tsne) > 0:
                        final_kl = scores_tsne[-1]
                        html_parts.append(f"""
                        <div style='background-color: #f8f9fa; padding: 15px; border-radius: 5px; border: 1px solid #ddd; margin-top: 20px;'>
                            <h3>Final t-SNE KL Divergence: {final_kl:.4f}</h3>
                            <p>Low KL Divergence indicates t-SNE successfully mapped local neighborhoods.</p>
                        </div>
                        """)
                else:
                    print(f"Warning: Last date {last_date} not found in returns index.")

            except KeyError as e:
                print(f"\nERROR: Could not find key {e} in results.")
                print("Available columns in PCA results:", results_pca[-1].columns if hasattr(results_pca[-1], 'columns') else "Not a DataFrame")
            except Exception as e:
                print(f"\nERROR calculating metrics: {e}")

        html = build_final_html(html_parts)
        with open("full_market_report.html", "w", encoding="utf-8") as f:
            f.write(html)
        print("\nDONE! Report saved to: full_market_report.html")
    
    else:
        print("No data found.")