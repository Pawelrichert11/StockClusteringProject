import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

def compute_tsne_snapshot(df_window, perplexity=30, random_state=42):
    """
    Oblicza współrzędne t-SNE dla jednego wycinka danych.
    """
    # 1. Przygotowanie danych (transpozycja: wiersze to spółki)
    X = df_window.T.values 
    
    # 2. Skalowanie (lokalne dla tego okna/snapshotu)
    scaler_data = StandardScaler()
    X_scaled = scaler_data.fit_transform(X)
    
    # 3. Uruchomienie t-SNE
    # init='pca' jest zazwyczaj bardziej stabilne dla danych finansowych
    tsne = TSNE(
        n_components=2, 
        perplexity=perplexity, 
        random_state=random_state, 
        init='pca', 
        learning_rate='auto'
    )
    X_embedded = tsne.fit_transform(X_scaled)
    
    # 4. Zwrócenie DataFrame z wynikami
    df_coords = pd.DataFrame(X_embedded, columns=['x', 'y'], index=df_window.columns)
    
    return df_coords