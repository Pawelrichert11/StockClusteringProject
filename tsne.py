import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

# ==========================================
# 1. GENEROWANIE PRZYKŁADOWYCH DANYCH
# ==========================================
def generate_dummy_stock_data(n_stocks=50, n_days=500):
    """Generuje losowe stopy zwrotu dla n_stocks spółek."""
    np.random.seed(42)
    dates = pd.date_range(start='2022-01-01', periods=n_days, freq='B')
    
    # Tworzymy sektory, żeby klastry miały sens
    sector_1 = np.random.normal(0.001, 0.02, (n_days, n_stocks // 3))
    sector_2 = np.random.normal(0.001, 0.02, (n_days, n_stocks // 3)) + np.sin(np.linspace(0, 10, n_days))[:, None] * 0.01
    sector_3 = np.random.normal(0.001, 0.02, (n_days, n_stocks - 2 * (n_stocks // 3)))
    
    data = np.hstack([sector_1, sector_2, sector_3])
    df = pd.DataFrame(data, index=dates, columns=[f'Stock_{i}' for i in range(n_stocks)])
    return df

# ==========================================
# 2. ANALIZA T-SNE W OKNACH CZASOWYCH
# ==========================================
def analyze_tsne_clusters(df_returns, window_size=60, step_size=20, n_clusters=3, perplexity=10):
    """
    Przeprowadza t-SNE + KMeans w oknach kroczących i oblicza stabilność.
    """
    scaler = StandardScaler()
    
    results = []      # Tu trzymamy ramki danych z wynikami dla każdego okna
    timestamps = []   # Daty końcowe okien
    stability_scores = [] # Wyniki zmienności w %
    
    prev_labels = None
    prev_stocks = None
    
    # Obliczamy liczbę kroków
    total_steps = (len(df_returns) - window_size) // step_size + 1
    
    for i in tqdm(range(total_steps), desc="t-SNE Clustering Progress"):
        # Definiowanie okna
        start_idx = i * step_size
        end_idx = start_idx + window_size
        
        if end_idx > len(df_returns):
            break
            
        df_window = df_returns.iloc[start_idx:end_idx]
        current_stocks = df_window.columns.tolist()
        
        # 1. Przygotowanie danych (Standaryzacja)
        # Transponujemy, bo t-SNE ma grupować SPÓŁKI (wiersze), a nie DNI
        X = df_window.T.values 
        X_scaled = scaler.fit_transform(X)
        
        # 2. t-SNE Embedding
        # t-SNE redukuje wymiarowość (np. z 60 dni do 2 wymiarów) bazując na podobieństwie
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca')
        X_embedded = tsne.fit_transform(X_scaled)
        
        # 3. KMeans na wynikach t-SNE
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_embedded)
        
        # 4. STABILIZACJA I DOPASOWANIE KLASTRÓW (Algorytm Węgierski)
        # Musimy dopasować kolory klastrów do poprzedniego okresu, bo KMeans losuje numery.
        if prev_labels is not None:
            # Budujemy macierz pomyłek (Conigency Matrix)
            # Wiersze: stare klastry, Kolumny: nowe klastry
            # Wartość: ile spółek przeszło z klastra A do B
            cost_matrix = np.zeros((n_clusters, n_clusters))
            
            for stock_idx, stock_name in enumerate(current_stocks):
                if stock_name in prev_stocks:
                    prev_idx = prev_stocks.index(stock_name)
                    old_c = prev_labels[prev_idx]
                    new_c = labels[stock_idx]
                    cost_matrix[old_c, new_c] -= 1 # Ujemne, bo algorytm szuka minimum kosztu
            
            # Algorytm węgierski znajduje najlepsze dopasowanie
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # Tworzymy mapę zamiany etykiet (np. Nowy 2 -> Stary 0)
            remapping = {new_c: old_c for old_c, new_c in zip(row_ind, col_ind)}
            
            # Aplikujemy zmianę
            aligned_labels = np.array([remapping.get(l, l) for l in labels])
            
            # --- OBLICZANIE ZMIENNOŚCI (STABILITY METRIC) ---
            changes = 0
            common_count = 0
            for stock_idx, stock_name in enumerate(current_stocks):
                if stock_name in prev_stocks:
                    prev_idx = prev_stocks.index(stock_name)
                    if prev_labels[prev_idx] != aligned_labels[stock_idx]:
                        changes += 1
                    common_count += 1
            
            churn_rate = (changes / common_count * 100) if common_count > 0 else 0
            stability_scores.append(churn_rate)
            
            labels = aligned_labels # Aktualizujemy etykiety na te wyrównane
        else:
            stability_scores.append(0) # Pierwszy okres = 0 zmian

        # Zapisujemy wyniki
        df_res = pd.DataFrame(X_embedded, columns=['x', 'y'], index=current_stocks)
        df_res['Cluster'] = labels
        results.append(df_res)
        timestamps.append(df_returns.index[end_idx - 1])
        
        prev_labels = labels
        prev_stocks = current_stocks

    return results, timestamps, stability_scores