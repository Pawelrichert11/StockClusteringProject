import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def run_pca(X_scaled, tickers_index, n_components=10):
    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)
    explained_var = pca.explained_variance_ratio_
    cum_explained_var = np.cumsum(explained_var)
    pca_table = pd.DataFrame({
        'PC': [f'PC{i+1}' for i in range(len(explained_var))],
        'Explained Variance': explained_var,
        'Cumulative Variance': cum_explained_var
    })
    df_pca = pd.DataFrame(
        pca.components_.T,
        index=tickers_index,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    return df_pca, pca_table