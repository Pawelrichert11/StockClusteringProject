import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



def preprocess_data(df_prices):
    # Log returns
    df_returns = np.log(df_prices / df_prices.shift(1)).dropna()
    
    # Standaryzacja (Z-score)
    # Pandas domyślnie działa na kolumnach (czyli na spółkach) - to jest OK
    X_scaled = (df_returns - df_returns.mean()) / df_returns.std()
    X_scaled = X_scaled.fillna(0)
    
    # UWAGA: USUNĄŁEM LINIĘ: X_scaled = X_scaled.T
    # Teraz X_scaled ma kształt: [Wiersze=Daty, Kolumny=Spółki]
    
    tickers_index = X_scaled.columns.tolist() # Teraz bierzemy kolumny jako tickery
    return X_scaled, tickers_index, df_returns

# --- 2. POPRAWKA: PCA wyciągające cechy spółek z macierzy czasu ---
def run_pca(X_scaled, tickers_index, n_components=10):
    # X_scaled wchodzi jako (Daty x Spółki)
    
    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)
    
    # Explained Variance (ile rynku wyjaśnia dany czynnik)
    explained_var = pca.explained_variance_ratio_
    cum_explained_var = np.cumsum(explained_var)
    
    # Tabela podsumowująca wariancję
    pca_table = pd.DataFrame({
        'PC': [f'PC{i+1}' for i in range(len(explained_var))],
        'Explained Variance': explained_var,
        'Cumulative Variance': cum_explained_var
    })
    
    # KLUCZOWE DLA GRUPOWANIA SPÓŁEK:
    # Chcemy wiedzieć, jak każda spółka (kolumna) reaguje na PC1, PC2 itd.
    # To znajduje się w pca.components_ (wymiar: n_components x n_features)
    # Musimy to obrócić (.T), żeby dostać (Spółki x n_components)
    
    loadings = pca.components_.T 
    
    # Opcjonalnie: Przeskalowanie ładunków przez pierwiastek z wariancji (dla lepszej interpretacji statystycznej)
    # loadings = loadings * np.sqrt(pca.explained_variance_)

    df_pca = pd.DataFrame(
        loadings,
        index=tickers_index,  # Teraz index to nasze Spółki
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    
    return df_pca, explained_var, cum_explained_var, pca_table