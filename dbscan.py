import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from tsne import compute_tsne_snapshot

def perform_static_dbscan_analysis(df_returns, perplexity=30, eps=0.5, min_samples=5):
    df_coords = compute_tsne_snapshot(df_returns, perplexity=perplexity)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)

    clusters = dbscan.fit_predict(df_coords)
    df_result = df_coords.copy()
    df_result['Cluster'] = clusters    
    return df_result

def analyze_rolling_dbscan(df_returns, window_size=60, step_size=20, perplexity=30, eps=3.0, min_samples=3):
    results = []      
    timestamps = []   
    stability_scores = [] 
    
    prev_labels = None
    prev_stocks = None
    
    total_steps = (len(df_returns) - window_size) // step_size + 1
    
    for i in tqdm(range(total_steps), desc="Rolling t-SNE+DBSCAN"):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        
        if end_idx > len(df_returns):
            break
            
        df_window_raw = df_returns.iloc[start_idx:end_idx]
        current_stocks = df_window_raw.columns.tolist()
        
        scaler = StandardScaler()
        X_window_scaled_vals = scaler.fit_transform(df_window_raw)
        X_window_scaled = pd.DataFrame(X_window_scaled_vals, index=df_window_raw.index, columns=df_window_raw.columns)
        
        df_coords = compute_tsne_snapshot(X_window_scaled, perplexity=perplexity)
        X_embedded = df_coords[['x', 'y']].values

        labels, churn_rate = _run_dbscan_step_with_memory(
            X_embedded, 
            current_stocks, 
            prev_labels, 
            prev_stocks, 
            eps=eps, 
            min_samples=min_samples
        )
        
        stability_scores.append(churn_rate)
        
        df_res = df_coords.copy()
        df_res['Cluster'] = labels
        results.append(df_res)
        timestamps.append(df_returns.index[end_idx - 1])
        
        prev_labels = labels
        prev_stocks = current_stocks

    return results, timestamps, stability_scores

def _run_dbscan_step_with_memory(X_embedded, current_stocks, prev_labels, prev_stocks, eps, min_samples):
    scaler_tsne = StandardScaler()
    X_embedded_norm = scaler_tsne.fit_transform(X_embedded)
    
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_embedded_norm)
    
    # Logika stabilności kolorów
    aligned_labels = labels.copy()
    churn_rate = 0
    
    if prev_labels is not None:
        mapping = {}
        unique_new = np.unique(labels)
        unique_new = unique_new[unique_new != -1] # Pomiń szum
        
        # Sortujemy od największych klastrów
        cluster_sizes = [(lbl, np.sum(labels == lbl)) for lbl in unique_new]
        cluster_sizes.sort(key=lambda x: x[1], reverse=True)
        
        for new_lbl, _ in cluster_sizes:
            indices = [idx for idx, lbl in enumerate(labels) if lbl == new_lbl]
            stocks_in_new = [current_stocks[idx] for idx in indices]
            
            old_votes = []
            for stock in stocks_in_new:
                if stock in prev_stocks:
                    prev_idx = prev_stocks.index(stock)
                    old_lbl = prev_labels[prev_idx]
                    if old_lbl != -1:
                        old_votes.append(old_lbl)
            
            if old_votes:
                # Wygrywa ten stary klaster, który ma najwięcej reprezentantów w nowym
                most_common_old = max(set(old_votes), key=old_votes.count)
                mapping[new_lbl] = most_common_old
            else:
                # Nowy klaster bez historii -> nowe ID (przesunięte o 100)
                mapping[new_lbl] = new_lbl + 100 
        
        aligned_labels = np.array([mapping.get(l, l) if l != -1 else -1 for l in labels])
        
        # Obliczanie Churn Rate
        changes = 0
        common_count = 0
        for idx, stock in enumerate(current_stocks):
            if stock in prev_stocks:
                prev_idx = prev_stocks.index(stock)
                if prev_labels[prev_idx] != aligned_labels[idx]:
                    changes += 1
                common_count += 1
        churn_rate = (changes / common_count * 100) if common_count > 0 else 0
        
    return aligned_labels, churn_rate