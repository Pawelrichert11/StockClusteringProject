import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

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
        
        # 1. PCA Calculation
        X_scaled = scaler.fit_transform(df_window)
        pca = PCA(n_components=n_components)
        # Note: PCA in sklearn expects (samples, features). 
        # Here samples=Days, features=Stocks. This is correct for finding market factors.
        # But for clustering STOCKS, we need loadings (Stocks x Components).
        pca.fit(X_scaled)
        
        # Loadings (Stocks x Components)
        loadings = pca.components_.T
        
        # --- PCA AXIS ALIGNMENT (Sign flipping) ---
        if prev_pca_df is not None:
            # Use only common stocks for alignment
            common_stocks = df_window.columns.intersection(prev_pca_df.index)
            
            if len(common_stocks) > 0:
                current_df_temp = pd.DataFrame(loadings, index=df_window.columns)
                curr_common = current_df_temp.loc[common_stocks]
                prev_common = prev_pca_df.loc[common_stocks]
                
                for comp_idx in range(n_components):
                    # Check correlation
                    corr = np.corrcoef(curr_common.iloc[:, comp_idx], prev_common.iloc[:, comp_idx])[0, 1]
                    if corr < 0:
                        loadings[:, comp_idx] *= -1
        
        df_pca_window = pd.DataFrame(loadings, columns=[f'PC{j+1}' for j in range(n_components)], index=df_window.columns)
        
        # 2. Clustering with WARM START
        data_to_cluster = loadings[:, :2] # Cluster on PC1 and PC2
        
        if prev_centers is not None:
            # init=prev_centers forces algorithm to start where it finished last time
            kmeans = KMeans(n_clusters=n_clusters, init=prev_centers, n_init=1, random_state=42)
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            
        kmeans.fit(data_to_cluster) 
        
        curr_centers = kmeans.cluster_centers_
        current_labels = kmeans.labels_.copy()
        
        # 3. Hungarian Algorithm for Label Matching
        if prev_centers is not None:
            dists = cdist(prev_centers, curr_centers)
            row_ind, col_ind = linear_sum_assignment(dists)
            
            remapping = {new_idx: old_idx for old_idx, new_idx in zip(row_ind, col_ind)}
            new_labels = np.array([remapping[label] for label in current_labels])
            
            # Align centers for next iteration
            aligned_centers = np.zeros_like(curr_centers)
            for new_idx, old_idx in remapping.items():
                aligned_centers[old_idx] = curr_centers[new_idx]
            
            current_labels = new_labels
            curr_centers = aligned_centers 
            
        else:
            # Initial sort for consistency
            sorted_idx = np.argsort(curr_centers[:, 0])
            mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_idx)}
            new_labels = np.array([mapping[label] for label in current_labels])
            curr_centers = curr_centers[sorted_idx]
            current_labels = new_labels

        prev_centers = curr_centers
        prev_pca_df = df_pca_window.copy()
        
        df_pca_window['Cluster'] = current_labels
        results.append(df_pca_window)
        timestamps.append(df_returns.index[end_idx - 1])
    
    return results, timestamps