#!/usr/bin/env python3
"""
Cluster business ideas from a CSV file using embeddings + UMAP + KMeans.
Interactive Plotly graph output.
"""

import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import umap
from sklearn.cluster import KMeans
from kneed import KneeLocator
import plotly.express as px
import webbrowser
import argparse

def find_optimal_k(X, max_k=25):
    """Find optimal k using elbow method (kneedle)."""
    inertias = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    kneedle = KneeLocator(range(1, max_k+1), inertias, curve='convex', direction='decreasing')
    optimal_k = kneedle.knee
    
    if optimal_k is None:
        # Fallback to silhouette score if no clear elbow
        from sklearn.metrics import silhouette_score
        best_k = 2
        best_score = -1
        for k in range(2, min(max_k, X.shape[0]-1)):
            labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
            if len(set(labels)) > 1:
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
        optimal_k = best_k
        print(f"No clear elbow found. Using silhouette score: k={optimal_k}")
    else:
        print(f"Elbow detected at k={optimal_k}")
    
    return optimal_k, inertias

def main(csv_path, id_col='id', text_col='idea_text', n_clusters=None, max_k=25):
    # 1. Load data
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    if id_col not in df.columns or text_col not in df.columns:
        raise ValueError(f"CSV must contain '{id_col}' and '{text_col}' columns.")
    texts = df[text_col].fillna('').astype(str).tolist()
    ids = df[id_col].tolist()
    print(f"Loaded {len(texts)} ideas.")

    # 2. Generate embeddings (GPU automatically if available)
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Computing embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    print(f"Embeddings shape: {embeddings.shape}")

    # 3. Dimensionality reduction with UMAP (to 5D for clustering)
    print("Reducing dimensions with UMAP (to 5 components)...")
    reducer = umap.UMAP(n_components=5, random_state=42, n_neighbors=15, min_dist=0.1)
    reduced = reducer.fit_transform(embeddings)
    print(f"Reduced shape: {reduced.shape}")

    # 4. Determine number of clusters
    if n_clusters is None:
        print(f"Finding optimal k (2 to {max_k}) using elbow method...")
        optimal_k, inertias = find_optimal_k(reduced, max_k=max_k)
        n_clusters = optimal_k
    else:
        print(f"Using provided k = {n_clusters}")

    # 5. KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(reduced)
    df['cluster'] = cluster_labels

    # 6. Prepare 2D visualisation (further reduce to 2D with UMAP for plotting)
    print("Preparing 2D plot projection...")
    viz_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    viz_coords = viz_reducer.fit_transform(embeddings)
    df['x'] = viz_coords[:, 0]
    df['y'] = viz_coords[:, 1]

    # 7. Interactive scatter plot
    fig = px.scatter(
        df, x='x', y='y', color='cluster', hover_data={id_col: True, text_col: True},
        title=f'Business Ideas – {n_clusters} Clusters (KMeans on UMAP reduced embeddings)',
        labels={'cluster': 'Cluster', 'x': 'UMAP 1', 'y': 'UMAP 2'},
        color_continuous_scale='Viridis'
    )
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    fig.update_layout(hoverlabel=dict(bgcolor="white", font_size=12))

    # 8. Save outputs to data/processed/ folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.abspath(os.path.join(script_dir, '../../data/processed'))
    os.makedirs(processed_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    output_csv = os.path.join(processed_dir, f'{base_name}_clustered.csv')
    output_html = os.path.join(processed_dir, f'{base_name}_clusters.html')
    
    # Save CSV with only essential columns + cluster
    df_out = df[[id_col, text_col, 'cluster']]
    df_out.to_csv(output_csv, index=False)
    print(f"Clustered data saved to: {output_csv}")
    
    fig.write_html(output_html)
    print(f"Interactive graph saved to: {output_html}")
    
    # Also save elbow plot (optional)
    if n_clusters is None:
        import matplotlib.pyplot as plt
        elbow_plot_path = os.path.join(processed_dir, f'{base_name}_elbow.png')
        plt.figure()
        plt.plot(range(1, max_k+1), inertias, 'bo-')
        plt.xlabel('k')
        plt.ylabel('Inertia')
        plt.title('Elbow Method')
        plt.axvline(x=n_clusters, color='r', linestyle='--')
        plt.savefig(elbow_plot_path)
        plt.close()
        print(f"Elbow plot saved to: {elbow_plot_path}")
    
    print("\nCluster sizes:")
    print(df['cluster'].value_counts().sort_index())
    
    # Open graph in browser
    webbrowser.open(output_html)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cluster text ideas from CSV')
    parser.add_argument('csv_file', help='Path to CSV file (relative or absolute)')
    parser.add_argument('--id-col', default='id', help='Column name for IDs (default: id)')
    parser.add_argument('--text-col', default='idea_text', help='Column name for idea text (default: idea_text)')
    parser.add_argument('--k', type=int, default=None, help='Number of clusters (auto-select if not given)')
    parser.add_argument('--max-k', type=int, default=25, help='Maximum clusters to consider for auto-selection (default: 25)')
    args = parser.parse_args()
    
    main(args.csv_file, args.id_col, args.text_col, args.k, args.max_k)
