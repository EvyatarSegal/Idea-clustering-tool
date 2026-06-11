#!/usr/bin/env python3
"""
Cluster business ideas using the 'summary' column from the intermediate CSV.
Input: ../../data/intermediate/test_ideas_summarized.csv
Output: ../../data/processed/test_ideas_clustered.csv + interactive HTML
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
    inertias = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    kneedle = KneeLocator(range(1, max_k+1), inertias, curve='convex', direction='decreasing')
    optimal_k = kneedle.knee
    if optimal_k is None:
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
        print(f"No clear elbow. Using silhouette: k={optimal_k}")
    else:
        print(f"Elbow detected at k={optimal_k}")
    return optimal_k, inertias

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='../../data/intermediate/test_ideas_summarized.csv',
                        help='Input CSV with summary column')
    parser.add_argument('--text-col', default='summary',
                        help='Column to cluster (default: summary). Falls back to idea_text if not found.')
    parser.add_argument('--output-dir', default='../../data/processed',
                        help='Directory for output files')
    parser.add_argument('--max-k', type=int, default=25,
                        help='Maximum clusters to consider')
    parser.add_argument('--k', type=int, default=None,
                        help='Force number of clusters (skip auto)')
    args = parser.parse_args()

    # Resolve paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, args.input)
    output_dir = os.path.join(script_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")

    # Determine text column
    if args.text_col not in df.columns:
        print(f"Column '{args.text_col}' not found. Falling back to 'idea_text'.")
        text_col = 'idea_text'
    else:
        text_col = args.text_col
    texts = df[text_col].fillna('').astype(str).tolist()
    print(f"Using column: {text_col}")

    # Generate embeddings (GPU if available)
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Computing embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    print(f"Embeddings shape: {embeddings.shape}")

    # Reduce dimensionality for clustering (5D)
    print("Reducing dimensions with UMAP (to 5 components)...")
    reducer = umap.UMAP(n_components=5, random_state=42, n_neighbors=15, min_dist=0.1)
    reduced = reducer.fit_transform(embeddings)

    # Determine k
    if args.k is None:
        optimal_k, inertias = find_optimal_k(reduced, max_k=args.max_k)
        n_clusters = optimal_k
    else:
        n_clusters = args.k
        print(f"Using user-provided k={n_clusters}")

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(reduced)
    df['cluster'] = cluster_labels

    # 2D projection for visualisation
    print("Preparing 2D plot projection...")
    viz_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    viz_coords = viz_reducer.fit_transform(embeddings)
    df['x'] = viz_coords[:, 0]
    df['y'] = viz_coords[:, 1]

    # Interactive plot
    hover_cols = ['id', text_col, 'cluster'] if 'id' in df.columns else [text_col, 'cluster']
    fig = px.scatter(df, x='x', y='y', color='cluster',
                     hover_data={col: True for col in hover_cols},
                     title=f'Idea Clustering – {n_clusters} clusters (using {text_col})',
                     labels={'cluster': 'Cluster', 'x': 'UMAP 1', 'y': 'UMAP 2'},
                     color_continuous_scale='Viridis')
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    fig.update_layout(hoverlabel=dict(bgcolor="white", font_size=12))

    # Save outputs
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_csv = os.path.join(output_dir, f'{base_name}_clustered.csv')
    output_html = os.path.join(output_dir, f'{base_name}_clusters.html')
    df.to_csv(output_csv, index=False)
    fig.write_html(output_html)

    print(f"Clustered CSV saved to: {output_csv}")
    print(f"Interactive graph saved to: {output_html}")
    print("\nCluster sizes:")
    print(df['cluster'].value_counts().sort_index())

    webbrowser.open(output_html)

if __name__ == '__main__':
    main()
