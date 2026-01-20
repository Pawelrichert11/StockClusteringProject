import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Importujemy silnik obliczeniowy
from tsne import compute_tsne_snapshot

def perform_static_dbscan_analysis(df_returns, perplexity=30, eps=0.5, min_samples=5):
    """
    Wykonuje analizę statyczną (Globalny snapshot całego okresu).
    """
    print("--- Running Static t-SNE + DBSCAN ---")
    
    # 1. Obliczamy geometrię (t-SNE)
    print(f"Computing t-SNE Geometry (perplexity={perplexity})...")
    df_coords = compute_tsne_snapshot(df_returns, perplexity=perplexity)
    
    # 2. Skalujemy współrzędne przed DBSCAN (dla spójności eps)
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(df_coords[['x', 'y']])
    
    # 3. Uruchamiamy DBSCAN
    print(f"Clustering with DBSCAN (eps={eps}, min_samples={min_samples})...")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(coords_scaled)
    
    df_result = df_coords.copy()
    df_result['Cluster'] = clusters
    
    # Statystyki szumu
    n_noise = list(clusters).count(-1)
    noise_pct = n_noise / len(clusters) * 100
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    print(f"Static Result: Found {n_clusters} clusters. Noise: {noise_pct:.1f}%")
    
    return df_result

def analyze_rolling_dbscan(df_returns, window_size=60, step_size=20, perplexity=30, eps=0.6, min_samples=3):
    """
    Wykonuje analizę w oknach kroczących (Rolling Window).
    Zarządza stabilnością kolorów (memory).
    """
    results = []      
    timestamps = []   
    stability_scores = [] 
    
    prev_labels = None
    prev_stocks = None
    
    total_steps = (len(df_returns) - window_size) // step_size + 1
    
    for i in tqdm(range(total_steps), desc="Rolling t-SNE+DBSCAN"):
        # 1. Definicja okna
        start_idx = i * step_size
        end_idx = start_idx + window_size
        
        if end_idx > len(df_returns):
            break
            
        df_window = df_returns.iloc[start_idx:end_idx]
        current_stocks = df_window.columns.tolist()
        
        # 2. Geometria (z tsne.py)
        df_coords = compute_tsne_snapshot(df_window, perplexity=perplexity)
        
        # 3. Klastrowanie + Stabilizacja
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
    """
    Pomocnicza funkcja wykonująca jeden krok DBSCAN i dopasowująca kolory do poprzedniego kroku.
    """
    # Standaryzacja WYNIKÓW t-SNE (żeby eps działał tak samo niezależnie od skali wykresu)
    scaler_tsne = StandardScaler()
    X_embedded_norm = scaler_tsne.fit_transform(X_embedded)
    
    # Właściwy DBSCAN
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