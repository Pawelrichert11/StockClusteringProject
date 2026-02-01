import pandas as pd
from sklearn.manifold import TSNE

def compute_tsne_snapshot(df_window, perplexity=30, random_state=42):
    X = df_window.T.values 
    tsne = TSNE(
        n_components=2, 
        perplexity=perplexity, 
        random_state=random_state, 
        init='pca', 
        learning_rate='auto'
    )
    X_embedded = tsne.fit_transform(X)
    df_coords = pd.DataFrame(X_embedded, columns=['x', 'y'], index=df_window.columns)
    return df_coords