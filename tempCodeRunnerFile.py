import pandas as pd
import numpy as np
from tqdm import tqdm
import glob
import os
import io
import base64
import datetime

# Upewnij się, że pliki dimension_reduction.py i clustering.py są w tym samym folderze
from pca import preprocess_data, run_pca
from kmeans import (cluster_pca_results, analyze_clusters_over_time)
from charts import (
    plot_corr_matrix_section,
    plot_pca_summary_section,
    plot_pca_scatter_section,
    plot_scree_section,
    plot_clustering_section,
    build_final_html
)


def load_stock_data(data_path, stock_limit=None, start_date='2012-01-01', min_dollar_volume=1000000, min_history_days=504, min_volatility=0.005):
    """
    Loads stocks, enforces time alignment, and patches small data gaps using ffill.
    """
    files = glob.glob(data_path)
    price_data = {}
    volume_data = {} 
    
    files.sort()
    
    print(f"Scanning {len(files)} files...")
    
    # 1. First Pass: Load everything loosely
    for file_path in tqdm(files, desc="Loading Raw Data"):
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

    print("Aligning data to common timeframe...")
    
    # 2. Create Master DataFrame
    df_prices = pd.DataFrame(price_data)
    df_vol = pd.DataFrame(volume_data)
    
    # 3. Slice by Date
    df_prices = df_prices.loc[start_date:]
    df_vol = df_vol.loc[start_date:]
    
    if df_prices.empty:
        print(f"Error: No data found after {start_date}.")
        return df_prices

    # Patching Holes
    df_prices.dropna(axis=0, how='all', inplace=True)
    df_prices.ffill(inplace=True)
    
    before_drop = df_prices.shape[1]
    df_prices.dropna(axis=1, how='any', inplace=True) 
    after_drop = df_prices.shape[1]
    
    print(f"Dropped {before_drop - after_drop} stocks that didn't exist at {start_date}.")
    
    # 4. Liquidity Filter
    valid_stocks = []
    if not df_vol.empty and not df_prices.empty:
        df_vol = df_vol.reindex(df_prices.index).ffill()
        
        for col in df_prices.columns:
            if col in df_vol.columns:
                avg_turnover = (df_prices[col] * df_vol[col]).mean()
                daily_returns = df_prices[col].pct_change().std()
                if avg_turnover > min_dollar_volume and daily_returns > min_volatility:
                    valid_stocks.append(col)
        df_prices = df_prices[valid_stocks]
    
    print(f"Final Dataset: {df_prices.shape[1]} stocks fully synchronized from {start_date}")
    
    return df_prices


if __name__ == "__main__":
    # --- CONFIGURATION ---
    DATA_PATH = "Stocks/*.txt" 
    STOCK_LIMIT = 400
    MIN_HISTORY_DAYS = 252 * 12
    N_COMPONENTS = 50  
    N_CLUSTERS = 5
    MIN_DOLLAR_VOLUME = 1000000
    MIN_VOLATILITY = 0.01 

    # --- EXECUTION ---
    df_prices = load_stock_data(DATA_PATH, stock_limit=STOCK_LIMIT, min_dollar_volume=MIN_DOLLAR_VOLUME, min_volatility=MIN_VOLATILITY)

    html_parts = []
    html_parts.append(f"<h1>ClusteringProject Report</h1>")
    html_parts.append(f"<p>Generated: {datetime.datetime.now().isoformat()}</p>")

    if not df_prices.empty:
        # Dataset summary
        html_parts.append("<h2>Dataset</h2>")
        html_parts.append(f"<p>Shape: {df_prices.shape[0]} rows x {df_prices.shape[1]} columns</p>")
        html_parts.append(df_prices.head().to_html(classes='data', border=1))

        # Dimension reduction
        X_scaled, tickers_index, df_returns = preprocess_data(df_prices)
        
        # Filter high-correlation stocks
        corr_with_market = X_scaled.corrwith(X_scaled.mean(axis=1)).abs()
        high_corr_stocks = corr_with_market[corr_with_market > corr_with_market.median()].index
        X_scaled_filtered = X_scaled[high_corr_stocks]
        html_parts.append(f"<p>Filtered to {len(high_corr_stocks)} highly-correlated stocks (from {len(tickers_index)})</p>")

        # --- Wykresy (Część 1) ---
        html_parts.append(plot_corr_matrix_section(X_scaled_filtered, corr_with_market))
        
        # FIX: NO TRANSPOSE HERE! X_scaled_filtered is (Dates x Stocks).
        df_pca, explained_var, cum_explained_var, pca_table = run_pca(
            X_scaled_filtered, 
            X_scaled_filtered.columns.tolist(), 
            n_components=N_COMPONENTS
        )
        
        html_parts.append(plot_pca_summary_section(pca_table))
        html_parts.append(plot_pca_scatter_section(df_pca, explained_var))
        html_parts.append(plot_scree_section(explained_var, cum_explained_var))
        
        # 6. Dynamic Clustering Calculation (MUST BE DONE BEFORE PLOTTING)
        print("Running cluster evolution analysis...")
        
        df_returns_filtered = df_returns[X_scaled_filtered.columns]
        
        results_time, timestamps = analyze_clusters_over_time(
            df_returns_filtered, 
            n_periods=10, 
            n_clusters=N_CLUSTERS, 
            n_components=10
        )

        # --- Wykresy (Część 2: Klastrowanie) ---
        # Przekazujemy wyniki analizy czasu do funkcji generującej wykresy
        html_parts.extend(plot_clustering_section(df_pca, explained_var, N_CLUSTERS, results_time, timestamps))

        # Build final HTML
        html = build_final_html(html_parts)

        out_path = os.path.join(os.getcwd(), "report.html")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)

        print(f"Report written to: {out_path}")
    else:
        print("No data loaded.")