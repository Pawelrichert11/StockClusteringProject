import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from tqdm import tqdm

from tsne import compute_tsne_snapshot

def perform_static_dbscan_analysis(df_returns, perplexity=30, eps=0.5, min_samples=5):
    df_coords = compute_tsne_snapshot(df_returns, perplexity=perplexity)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)

    clusters = dbscan.fit_predict(df_coords)
    df_result = df_coords.copy()
    df_result['Cluster'] = clusters    
    return df_result

def analyze_rolling_dbscan(df_returns, window_size=60, step_size=20, perplexity=30, eps=0.5, min_samples=3):
    results = []      
    timestamps = []   
    stability_scores = [] 
    
    prev_labels = None
    prev_stocks = None
    
    next_cluster_id = 0
    
    total_steps = (len(df_returns) - window_size) // step_size + 1
    
    for i in tqdm(range(total_steps), desc="Rolling t-SNE+DBSCAN"):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        
        if end_idx > len(df_returns):
            break
            
        df_window_raw = df_returns.iloc[start_idx:end_idx]
    
        df_window_stocks = df_window_raw.T 
        current_stocks = df_window_stocks.index.tolist()
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_window_stocks) #scaling to stabilize eps
        
        tsne = TSNE(n_components=2, perplexity=perplexity, init='pca')
        X_embedded = tsne.fit_transform(X_scaled)

        labels, churn_rate, next_cluster_id = _run_dbscan_step_with_memory(
            X_embedded, 
            current_stocks, 
            prev_labels, 
            prev_stocks, 
            eps=eps, 
            min_samples=min_samples,
            next_cluster_id=next_cluster_id
        )
        
        stability_scores.append(churn_rate)
        
        df_res = pd.DataFrame(X_embedded, columns=['x', 'y'], index=current_stocks)
        df_res['Cluster'] = labels
        
        results.append(df_res)
        timestamps.append(df_returns.index[end_idx - 1])
        
        prev_labels = labels
        prev_stocks = current_stocks

    return results, timestamps, stability_scores

def _run_dbscan_step_with_memory(X_embedded, current_stocks, prev_labels, prev_stocks, eps, min_samples, next_cluster_id):
    scaler_tsne = StandardScaler()
    X_embedded_norm = scaler_tsne.fit_transform(X_embedded)
    
    db = DBSCAN(eps=eps, min_samples=min_samples)
    raw_labels = db.fit_predict(X_embedded_norm)
    
    mapping = {}
    mapping[-1] = -1
    
    unique_raw = np.unique(raw_labels)
    unique_raw = unique_raw[unique_raw != -1]
    
    cluster_sizes = [(lbl, np.sum(raw_labels == lbl)) for lbl in unique_raw]
    cluster_sizes.sort(key=lambda x: x[1], reverse=True)
    
    used_old_labels = set()

    if prev_labels is not None:
        for new_lbl, _ in cluster_sizes:
            indices = [idx for idx, lbl in enumerate(raw_labels) if lbl == new_lbl]
            stocks_in_new = [current_stocks[idx] for idx in indices]
            
            old_votes = []
            for stock in stocks_in_new:
                if stock in prev_stocks:
                    prev_idx = prev_stocks.index(stock)
                    old_lbl = prev_labels[prev_idx]
                    if old_lbl != -1:
                        old_votes.append(old_lbl)
            
            assigned = False
            if old_votes:
                most_common_old = max(set(old_votes), key=old_votes.count)
                if most_common_old not in used_old_labels:
                    mapping[new_lbl] = most_common_old
                    used_old_labels.add(most_common_old)
                    assigned = True
            
            if not assigned:
                mapping[new_lbl] = next_cluster_id
                next_cluster_id += 1
    else:
        for new_lbl in unique_raw:
            mapping[new_lbl] = next_cluster_id
            next_cluster_id += 1
            
    aligned_labels = np.array([mapping.get(l, -1) for l in raw_labels])
    
    changes = 0
    common_count = 0
    if prev_labels is not None:
        for idx, stock in enumerate(current_stocks):
            if stock in prev_stocks:
                prev_idx = prev_stocks.index(stock)
                if prev_labels[prev_idx] != aligned_labels[idx]:
                    changes += 1
                common_count += 1
                
    churn_rate = (changes / common_count * 100) if common_count > 0 else 0
        
    return aligned_labels, churn_rate, next_cluster_id