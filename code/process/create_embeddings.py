#!/usr/bin/env python3
"""
Generate embeddings from text column, optionally cluster, and save as:
- CSV with cluster labels
- (Optional) CSV with each embedding dimension as separate column
- (Optional) NumPy .npy file of embeddings
"""

import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import umap
from sklearn.cluster import KMeans
from kneed import KneeLocator
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_file', help='Input CSV file')
    parser.add_argument('--id-col', default='id')
    parser.add_argument('--text-col', default='idea_text')
    parser.add_argument('--cluster', action='store_true', help='Perform clustering and add cluster labels')
    parser.add_argument('--max-k', type=int, default=25)
    parser.add_argument('--export-dimensions', action='store_true', help='Export each embedding dimension as separate column')
    parser.add_argument('--export-npy', action='store_true', help='Export embeddings as .npy file')
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.csv_file)
    texts = df[args.text_col].fillna('').astype(str).tolist()

    # Generate embeddings
    print("Loading model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    print(f"Embeddings shape: {embeddings.shape}")

    # Prepare output directory
    out_dir = os.path.dirname(os.path.abspath(args.csv_file))
    base = os.path.splitext(os.path.basename(args.csv_file))[0]

    # Save embeddings as .npy
    if args.export_npy:
        npy_path = os.path.join(out_dir, f'{base}_embeddings.npy')
        np.save(npy_path, embeddings)
        print(f"Embeddings saved to: {npy_path}")

    # Export each dimension as separate column
    if args.export_dimensions:
        emb_df = pd.DataFrame(embeddings, columns=[f'dim_{i}' for i in range(embeddings.shape[1])])
        output_df = pd.concat([df.reset_index(drop=True), emb_df], axis=1)
        dim_csv = os.path.join(out_dir, f'{base}_with_embeddings.csv')
        output_df.to_csv(dim_csv, index=False)
        print(f"CSV with {embeddings.shape[1]} dimension columns saved to: {dim_csv}")

    # Clustering
    if args.cluster:
        print("Reducing dimensions with UMAP...")
        reduced = umap.UMAP(n_components=5, random_state=42).fit_transform(embeddings)
        
        # Find optimal k
        inertias = []
        for k in range(1, args.max_k+1):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(reduced)
            inertias.append(km.inertia_)
        kneedle = KneeLocator(range(1, args.max_k+1), inertias, curve='convex', direction='decreasing')
        optimal_k = kneedle.knee or 5
        
        print(f"Clustering with k={optimal_k}")
        labels = KMeans(n_clusters=optimal_k, random_state=42, n_init=10).fit_predict(reduced)
        df['cluster'] = labels
        
        # Save clustered CSV (without embedding columns unless requested)
        cluster_csv = os.path.join(out_dir, f'{base}_clustered.csv')
        df.to_csv(cluster_csv, index=False)
        print(f"Clustered data saved to: {cluster_csv}")
        print("\nCluster sizes:")
        print(df['cluster'].value_counts().sort_index())

if __name__ == '__main__':
    main()
