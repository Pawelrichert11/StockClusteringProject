import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from pca import run_pca

def cluster_pca_results(df_pca_results, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(df_pca_results[['PC1', 'PC2']])
    df_pca_results['Cluster'] = clusters
    return df_pca_results, kmeans


def analyze_clusters_over_time(df_returns, window_size=60, step_size=20, n_clusters=5, n_components=10):
    scaler = StandardScaler()
    
    results = []
    timestamps = []
    
    prev_pca_df = None
    prev_centers = None

    total_steps = (len(df_returns) - window_size) // step_size + 1
    
    for i in tqdm(range(total_steps), desc="PCA Cluster Evolution"):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        
        if end_idx > len(df_returns):
            break
            
        df_window = df_returns.iloc[start_idx:end_idx]
        
        # 1. Scaling & PCA Calculation (używając funkcji z pca.py)
        # Skalujemy okno tutaj, aby zachować logikę "per window"
        X_scaled = scaler.fit_transform(df_window)
        
        # run_pca zwraca (df_pca, pca_table). Interesuje nas tylko df_pca (loadings)
        # Przekazujemy df_window.columns jako indeks, żeby df_pca miało nazwy spółek
        df_pca_window, _ = run_pca(X_scaled, df_window.columns, n_components=n_components)
        
        # --- PCA AXIS ALIGNMENT (Sign flipping) ---
        # PCA może losowo odwracać znaki wektorów własnych.
        # Sprawdzamy korelację z poprzednim krokiem i odwracamy w razie potrzeby.
        if prev_pca_df is not None:
            # Używamy tylko wspólnych spółek do porównania
            common_stocks = df_pca_window.index.intersection(prev_pca_df.index)
            
            if len(common_stocks) > 0:
                for col in df_pca_window.columns: # Iterujemy po PC1, PC2...
                    curr_vec = df_pca_window.loc[common_stocks, col]
                    prev_vec = prev_pca_df.loc[common_stocks, col]
                    
                    # Jeśli korelacja jest ujemna, odwracamy znak całej kolumny
                    if np.corrcoef(curr_vec, prev_vec)[0, 1] < 0:
                        df_pca_window[col] *= -1
        
        # 2. Clustering with WARM START
        # Bierzemy dwie pierwsze składowe do klastrowania
        data_to_cluster = df_pca_window.iloc[:, :2].values
        
        if prev_centers is not None:
            # init=prev_centers zmusza algorytm do startu tam, gdzie skończył ostatnio
            kmeans = KMeans(n_clusters=n_clusters, init=prev_centers, n_init=1, random_state=42)
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            
        kmeans.fit(data_to_cluster) 
        
        curr_centers = kmeans.cluster_centers_
        current_labels = kmeans.labels_.copy()
        
        # 3. Hungarian Algorithm for Label Matching
        # Dopasowujemy numery klastrów (0, 1, 2...) tak, aby "zielony" klaster pozostawał "zielony"
        if prev_centers is not None:
            dists = cdist(prev_centers, curr_centers)
            row_ind, col_ind = linear_sum_assignment(dists)
            
            remapping = {new_idx: old_idx for old_idx, new_idx in zip(row_ind, col_ind)}
            new_labels = np.array([remapping[label] for label in current_labels])
            
            # Wyrównujemy centra do następnej iteracji
            aligned_centers = np.zeros_like(curr_centers)
            for new_idx, old_idx in remapping.items():
                aligned_centers[old_idx] = curr_centers[new_idx]
            
            current_labels = new_labels
            curr_centers = aligned_centers 
            
        else:
            # Sortowanie początkowe dla spójności (np. od lewej do prawej na osi X)
            sorted_idx = np.argsort(curr_centers[:, 0])
            mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_idx)}
            new_labels = np.array([mapping[label] for label in current_labels])
            curr_centers = curr_centers[sorted_idx]
            current_labels = new_labels

        prev_centers = curr_centers
        prev_pca_df = df_pca_window.copy()
        
        # Przypisujemy ostateczne etykiety do DataFrame
        df_pca_window['Cluster'] = current_labels
        
        results.append(df_pca_window)
        timestamps.append(df_returns.index[end_idx - 1])
    
    return results, timestamps