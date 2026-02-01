import pandas as pd
import numpy as np
from tqdm import tqdm
import glob
import os
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from dbscan import analyze_rolling_dbscan, perform_static_dbscan_analysis
from pca import run_pca
from kmeans import dynamic_kmeans, cluster_pca_results

from html_builder import (
    build_dynamic_kmeans, 
    build_header,
    build_static_dbscan, 
    build_tsne_evolution, 
    build_quality_metrics,
    save_report,
    build_pairs_analysis,
    generate_trustworthiness_section,
    build_pairs_html,
    build_static_kmeans,
    build_pca_stats
)

DATA_PATH = "Stocks/*.txt" 
STOCK_LIMIT = 20000
N_COMPONENTS = 50  
N_CLUSTERS = 5
WINDOW_SIZE = 60
STEP_SIZE = 10 
START_DATE = '2012-01-01'

def load_stock_data(data_path, stock_limit=None, start_date='2012-01-01', min_dollar_volume=1000000, min_volatility=0.005):
    files = glob.glob(data_path)
    price_data = {}
    volume_data = {} 
    
    files.sort()
    print(f"Scanning {len(files)} files")
    
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

    df_prices = pd.DataFrame(price_data)
    df_vol = pd.DataFrame(volume_data)
    
    df_prices = df_prices.loc[start_date:]
    df_vol = df_vol.loc[start_date:]
    
    if df_prices.empty: return df_prices

    df_prices.dropna(axis=0, how='all', inplace=True)
    df_prices.ffill(inplace=True)
    df_prices.dropna(axis=1, how='any', inplace=True) 
    
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
    
    print(f"Final Dataset has: {df_prices.shape[1]} stocks.")
    return df_prices

def preprocess_data(df_prices, filter_outliers=True):
    df_returns = np.log(df_prices / df_prices.shift(1)).dropna()
    scaler = StandardScaler()
    X_scaled_values = scaler.fit_transform(df_returns)
    X_scaled = pd.DataFrame(X_scaled_values, index=df_returns.index, columns=df_returns.columns)
    
    if filter_outliers:
        market_proxy = X_scaled.mean(axis=1)
        corr_with_market = X_scaled.corrwith(market_proxy).abs()
    
        median_corr = corr_with_market.median()
        high_corr_stocks = corr_with_market[corr_with_market > median_corr].index
        
        print(f"Preprocessing: Filtering outliers. Keeping {len(high_corr_stocks)}/{len(X_scaled.columns)} stocks (Corr > {median_corr:.3f})")
        X_scaled = X_scaled[high_corr_stocks]
        df_returns = df_returns[high_corr_stocks]
    else:
        print(f"Preprocessing: No filtering. Keeping all {len(X_scaled.columns)} stocks.")

    return X_scaled, df_returns

def calculate_trustworthiness_comparison(raw_returns_data, df_pca, df_static_dbscan):
    common_indices = [
        idx for idx in raw_returns_data.columns 
        if idx in df_pca.index and idx in df_static_dbscan.index
    ]
    if not common_indices:
        print("Error: No common stocks found between Returns, PCA, and t-SNE data.")
        return None
    X_high = raw_returns_data.loc[:, common_indices].T.values
    X_low_pca = df_pca.loc[common_indices, ['PC1', 'PC2']].values
    X_low_tsne = df_static_dbscan.loc[common_indices, ['x', 'y']].values

    score_pca = trustworthiness(X_high, X_low_pca, n_neighbors=5)
    score_tsne = trustworthiness(X_high, X_low_tsne, n_neighbors=5)

    return generate_trustworthiness_section(score_pca, score_tsne)

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
        if original_returns is not None:
            correlation = original_returns[t1].corr(original_returns[t2])

        results.append({
            'stock_a': t1,
            'stock_b': t2,
            'distance': dist,
            'correlation': correlation
        })
        
    return results

def calculate_pairs_html(coords_kmeans, coords_dbscan, original_returns, top_n=10):
    pairs_data1 = calculate_top_pairs(coords_kmeans, original_returns, top_n)
    html_pca_pairs = build_pairs_html(pairs_data1, "k-Means PCA")

    pairs_data2 = calculate_top_pairs(coords_dbscan, original_returns, top_n)
    html_tsne_pairs = build_pairs_html(pairs_data2, "DBSCAN t-SNE")

    return html_pca_pairs, html_tsne_pairs

def calculate_comparison_metrics(raw_returns_data, results_pca, times_pca, results_tsne, window_size):
    last_labels_pca = results_pca[-1]['Cluster']
    last_labels_tsne = results_tsne[-1]['Cluster']
    
    last_date = results_pca[-1]['date'] if 'date' in results_pca[-1] else times_pca[-1]

    if last_date not in raw_returns_data.index:
        print(f"Warning: Last date {last_date} not found in returns index.")
        return None

    end_idx = raw_returns_data.index.get_loc(last_date)
    start_idx = max(0, end_idx - window_size)
        
    X_window = raw_returns_data.iloc[start_idx:end_idx]
    X_eval = (X_window - X_window.mean()) / X_window.std()
    X_eval = X_eval.fillna(0).T 

    def get_metrics(X, labels, method_name):
        unique_labels = set(labels)
        if len(unique_labels) < 2:
            print(f"Warning: {method_name} found only {len(unique_labels)} cluster(s). Skipping metrics.")
            return {
                "Method": method_name,
                "Silhouette": 0,
                "Calinski-Harabasz": 0,
                "Davies-Bouldin": 0 
            }

        return {
                "Method": method_name,
                "Silhouette": silhouette_score(X, labels),
                "Calinski-Harabasz": calinski_harabasz_score(X, labels),
                "Davies-Bouldin": davies_bouldin_score(X, labels)
            }

    metrics_data = []
    metrics_data.append(get_metrics(X_eval, last_labels_pca, "PCA + K-Means"))
    metrics_data.append(get_metrics(X_eval, last_labels_tsne, "t-SNE + DBSCAN"))
        
    df_metrics = pd.DataFrame(metrics_data)
    return df_metrics

if __name__ == "__main__":
    html_parts = []
    df_prices = load_stock_data(DATA_PATH, stock_limit=STOCK_LIMIT, start_date=START_DATE)

    if df_prices.empty:
        print("No data found. Exiting.")
        exit()

    scaled_returns_data, raw_returns_data = preprocess_data(df_prices, filter_outliers=True)
    html_parts.extend(build_header(scaled_returns_data))

    df_pca, pca_table = run_pca(scaled_returns_data, scaled_returns_data.columns.tolist(), N_COMPONENTS)

    html_parts.append(build_pca_stats(pca_table, top_n=10))
    df_pca_static, kmeans_static = cluster_pca_results(df_pca, N_CLUSTERS)
    html_parts.extend(build_static_kmeans(df_pca_static, kmeans_static))
    
    df_static_dbscan = perform_static_dbscan_analysis(
        scaled_returns_data, 
        perplexity=30, 
        eps=0.5, 
        min_samples=5
    )
    html_parts.append(build_static_dbscan(df_static_dbscan))
    
    results_pca, times_pca = dynamic_kmeans(
        raw_returns_data, 
        window_size=WINDOW_SIZE, 
        step_size=STEP_SIZE, 
        n_clusters=N_CLUSTERS
    )
    html_parts.extend(build_dynamic_kmeans(results_pca, times_pca))

    results_tsne, times_tsne, scores_tsne = analyze_rolling_dbscan(
        raw_returns_data,
        window_size=WINDOW_SIZE, 
        step_size=STEP_SIZE, 
        perplexity=30,
        eps=0.2,        
        min_samples=4    
    )

    html_parts.extend(build_tsne_evolution(results_tsne, times_tsne))

    trust_html = calculate_trustworthiness_comparison(raw_returns_data, df_pca, df_static_dbscan)
    if trust_html:
        html_parts.append(trust_html)
    
    df_metrics = calculate_comparison_metrics(raw_returns_data, results_pca, times_pca, results_tsne, WINDOW_SIZE)
    html_parts.extend(build_quality_metrics(df_metrics))
    
    html_pca_pairs, html_tsne_pairs = calculate_pairs_html(
        df_pca.iloc[:, :5], 
        df_static_dbscan[['x', 'y']], 
        raw_returns_data, 
        top_n=10
    )
    html_parts.extend(build_pairs_analysis(html_pca_pairs, html_tsne_pairs))

    save_report(html_parts)