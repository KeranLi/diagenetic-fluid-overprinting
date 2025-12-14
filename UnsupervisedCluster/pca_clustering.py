"""
PCA-Clustering Analysis Pipeline
Performs dimensionality reduction via Principal Component Analysis followed by K-means clustering.

IMPORTANT: KNN = K-Nearest Neighbors (classification algorithm)
           K-means = K-means clustering (clustering algorithm)
This script uses K-means for clustering, which is the standard approach for PCA+clustering workflows.

Author: Keran Li
Date: 2025-10-8
License: MIT
"""

import numpy as np
import pandas as pd
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class PCAClusteringPipeline:
    """
    PCA + Clustering pipeline with automated cluster number optimization.
    Supports K-means (recommended), DBSCAN, and Hierarchical clustering.
    """
    
    def __init__(self, n_components=0.95, clustering_method='kmeans', 
                 n_clusters='auto', random_state=42):
        """
        Initialize the PCA-Clustering pipeline.
        
        Args:
            n_components: Number of PCA components or variance ratio (0-1)
            clustering_method: 'kmeans' (K-means clustering), 'dbscan', or 'hierarchical'
            n_clusters: Number of clusters ('auto' or int)
            random_state: Random seed for reproducibility
        """
        self.n_components = n_components
        self.clustering_method = clustering_method.lower()
        self.n_clusters = n_clusters
        self.random_state = random_state
        
        # Initialize components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self.clusterer = None
        self.labels_ = None
        self.optimal_n_clusters_ = None
        
        # Validation
        valid_methods = ['kmeans', 'dbscan', 'hierarchical']
        if self.clustering_method not in valid_methods:
            raise ValueError(f"clustering_method must be one of {valid_methods}")
    
    def load_data(self, filepath):
        """Load dataset from CSV file."""
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        return df
    
    def preprocess(self, X):
        """Scale data and apply PCA."""
        print("\nScaling data...")
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"\nApplying PCA (n_components={self.n_components})...")
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Print PCA results
        n_components_selected = self.pca.n_components_
        explained_var = np.sum(self.pca.explained_variance_ratio_)
        print(f"Components selected: {n_components_selected}")
        print(f"Total explained variance: {explained_var:.4f}")
        
        return X_pca
    
    def find_optimal_clusters(self, X_pca, max_clusters=10):
        """Determine optimal number of clusters using elbow and silhouette methods."""
        if self.clustering_method == 'dbscan':
            print("DBSCAN does not require optimal cluster number (uses eps and min_samples)")
            return None
        
        print(f"\nTesting cluster numbers from 2 to {max_clusters}...")
        
        inertias = []
        silhouette_scores = []
        
        for k in tqdm(range(2, max_clusters + 1), desc="Evaluating clusters"):
            clusterer = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = clusterer.fit_predict(X_pca)
            
            inertias.append(clusterer.inertia_)
            silhouette_scores.append(silhouette_score(X_pca, labels))
        
        # Plot evaluation metrics
        self._plot_cluster_metrics(range(2, max_clusters + 1), inertias, silhouette_scores)
        
        # Determine optimal k (max silhouette score)
        optimal_n = range(2, max_clusters + 1)[np.argmax(silhouette_scores)]
        self.optimal_n_clusters_ = optimal_n
        
        print(f"\nOptimal number of clusters: {optimal_n}")
        print(f"Best Silhouette score: {max(silhouette_scores):.4f}")
        
        return optimal_n
    
    def _plot_cluster_metrics(self, k_range, inertias, silhouette_scores):
        """Plot cluster evaluation metrics."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Elbow method
        axes[0].plot(k_range, inertias, 'bo-')
        axes[0].set_xlabel('Number of Clusters (k)')
        axes[0].set_ylabel('Inertia (Within-cluster SSE)')
        axes[0].set_title('Elbow Method')
        axes[0].grid(True, alpha=0.3)
        
        # Silhouette score
        axes[1].plot(k_range, silhouette_scores, 'ro-')
        axes[1].set_xlabel('Number of Clusters (k)')
        axes[1].set_ylabel('Silhouette Score')
        axes[1].set_title('Silhouette Analysis (higher is better)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('cluster_evaluation_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def perform_clustering(self, X_pca):
        """Perform clustering on PCA-transformed data."""
        print(f"\nPerforming {self.clustering_method.upper()} clustering...")
        
        if self.clustering_method == 'kmeans':
            if self.n_clusters == 'auto':
                n_clusters = self.find_optimal_clusters(X_pca)
            else:
                n_clusters = self.n_clusters
            
            print(f"K-means with n_clusters={n_clusters}...")
            self.clusterer = KMeans(
                n_clusters=n_clusters, 
                random_state=self.random_state,
                n_init=10,
                init='k-means++'
            )
            self.labels_ = self.clusterer.fit_predict(X_pca)
            print(f"Inertia: {self.clusterer.inertia_:.2f}")
            
        elif self.clustering_method == 'dbscan':
            print(f"DBSCAN (eps=0.5, min_samples=5)...")
            self.clusterer = DBSCAN(eps=0.5, min_samples=5, metric='euclidean')
            self.labels_ = self.clusterer.fit_predict(X_pca)
            
            n_clusters = len(np.unique(self.labels_[self.labels_ != -1]))
            n_noise = np.sum(self.labels_ == -1)
            print(f"Clusters found: {n_clusters}, Noise points: {n_noise}")
            
        elif self.clustering_method == 'hierarchical':
            if self.n_clusters == 'auto':
                n_clusters = self.find_optimal_clusters(X_pca)
            else:
                n_clusters = self.n_clusters
            
            print(f"Hierarchical (linkage=ward, n_clusters={n_clusters})...")
            self.clusterer = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage='ward'
            )
            self.labels_ = self.clusterer.fit_predict(X_pca)
        
        return self.labels_
    
    def evaluate_clustering(self, X_pca):
        """Evaluate clustering quality metrics."""
        print("\nEvaluating clustering quality...")
        
        if self.clustering_method == 'dbscan':
            core_samples = self.labels_ != -1
            X_eval = X_pca[core_samples]
            labels_eval = self.labels_[core_samples]
        else:
            X_eval = X_pca
            labels_eval = self.labels_
        
        if len(np.unique(labels_eval)) < 2:
            print("Warning: Only 1 cluster found. Cannot compute metrics.")
            return None
        
        silhouette_avg = silhouette_score(X_eval, labels_eval)
        db_score = davies_bouldin_score(X_eval, labels_eval)
        
        print(f"\nMetrics:")
        print(f"  Silhouette Score: {silhouette_avg:.4f} (higher is better)")
        print(f"  Davies-Bouldin Index: {db_score:.4f} (lower is better)")
        
        return {'silhouette': silhouette_avg, 'davies_bouldin': db_score}
    
    def plot_results(self, X_pca, original_data=None, save_path=None):
        """Generate comprehensive clustering visualizations."""
        print("\nGenerating visualizations...")
        
        n_components = X_pca.shape[1]
        
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle(f'PCA-Clustering Results ({self.clustering_method.upper()})', 
                    fontsize=16, fontweight='bold')
        
        # 1. PCA scatter plot
        ax1 = plt.subplot(2, 3, 1)
        scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=self.labels_, 
                             cmap='viridis', alpha=0.7, s=50)
        ax1.set_xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]:.2%} variance)')
        ax1.set_ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]:.2%} variance)')
        ax1.set_title('PCA Cluster Visualization')
        plt.colorbar(scatter, ax=ax1, label='Cluster ID')
        ax1.grid(True, alpha=0.3)
        
        # 2. Cumulative variance
        ax2 = plt.subplot(2, 3, 2)
        cumulative_var = np.cumsum(self.pca.explained_variance_ratio_)
        ax2.plot(range(1, len(cumulative_var) + 1), cumulative_var, 'bo-')
        ax2.axhline(y=0.95, color='r', linestyle='--')
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Explained Variance')
        ax2.set_title('PCA Variance Explained')
        ax2.grid(True, alpha=0.3)
        
        # 3. Feature importance
        ax3 = plt.subplot(2, 3, 3)
        if hasattr(self.pca, 'components_'):
            feature_importance = np.abs(self.pca.components_[0])
            ax3.bar(range(len(feature_importance)), feature_importance, color='steelblue')
            ax3.set_xlabel('Feature Index')
            ax3.set_ylabel('Absolute Loading')
            ax3.set_title('PC1 Feature Importance')
            ax3.grid(True, alpha=0.3)
        
        # 4. Cluster size distribution
        ax4 = plt.subplot(2, 3, 4)
        cluster_counts = pd.Series(self.labels_).value_counts().sort_index()
        ax4.bar(cluster_counts.index.astype(str), cluster_counts.values, color='darkcyan')
        ax4.set_xlabel('Cluster ID')
        ax4.set_ylabel('Number of Samples')
        ax4.set_title('Cluster Size Distribution')
        ax4.grid(True, alpha=0.3)
        
        # 5. t-SNE visualization
        if len(X_pca) < 5000:
            ax5 = plt.subplot(2, 3, 5)
            print("Performing t-SNE...")
            tsne = TSNE(n_components=2, random_state=self.random_state, perplexity=min(30, len(X_pca)-1))
            X_tsne = tsne.fit_transform(X_pca)
            
            scatter = ax5.scatter(X_tsne[:, 0], X_tsne[:, 1], c=self.labels_, 
                                 cmap='viridis', alpha=0.7, s=30)
            ax5.set_xlabel('t-SNE 1')
            ax5.set_ylabel('t-SNE 2')
            ax5.set_title('t-SNE Visualization')
            plt.colorbar(scatter, ax=ax5, label='Cluster ID')
            ax5.grid(True, alpha=0.3)
        
        # 6. Silhouette plot (for K-means/hierarchical)
        if self.clustering_method in ['kmeans', 'hierarchical']:
            ax6 = plt.subplot(2, 3, 6)
            from sklearn.metrics import silhouette_samples
            sample_values = silhouette_samples(X_pca, self.labels_)
            
            y_lower = 0
            for i in sorted(np.unique(self.labels_)):
                ith_cluster_values = sample_values[self.labels_ == i]
                ith_cluster_values.sort()
                size_cluster_i = ith_cluster_values.shape[0]
                y_upper = y_lower + size_cluster_i
                color = plt.cm.viridis(float(i) / len(np.unique(self.labels_)))
                ax6.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_values,
                                 facecolor=color, edgecolor=color, alpha=0.7)
                y_lower = y_upper + 10
            ax6.set_xlabel("Silhouette coefficient")
            ax6.set_ylabel("Cluster")
            ax6.set_title("Silhouette Plot")
            ax6.axvline(x=sample_values.mean(), color="red", linestyle="--")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def export_results(self, X_pca, original_data=None, prefix='clustering_results'):
        """Export clustering results and model."""
        print("\nExporting results...")
        
        results_df = pd.DataFrame({
            'sample_id': range(len(X_pca)),
            'cluster_id': self.labels_
        })
        
        for i in range(X_pca.shape[1]):
            results_df[f'PC{i+1}'] = X_pca[:, i]
        
        if original_data is not None:
            results_df = pd.concat([results_df, original_data.reset_index(drop=True)], axis=1)
        
        results_df.to_csv(f"{prefix}_results.csv", index=False)
        print(f"Results saved to {prefix}_results.csv")
        
        # Save model
        joblib.dump({
            'scaler': self.scaler,
            'pca': self.pca,
            'clusterer': self.clusterer,
            'labels': self.labels_
        }, f"{prefix}_model.pkl")
        print(f"Model saved to {prefix}_model.pkl")
        
        return results_df


def main():
    parser = argparse.ArgumentParser(
        description='PCA-Clustering Analysis Pipeline (K-means, DBSCAN, Hierarchical)',
        epilog='Note: KNN is for classification. For clustering, use K-means (recommended), DBSCAN, or Hierarchical.'
    )
    
    parser.add_argument('--file', type=str, required=True, help='Path to CSV dataset')
    parser.add_argument('--method', type=str, default='kmeans', 
                       choices=['kmeans', 'dbscan', 'hierarchical'],
                       help='Clustering algorithm (K-means is standard PCA+clustering)')
    parser.add_argument('--components', type=float, default=0.95,
                       help='PCA components (int) or variance ratio (0-1)')
    parser.add_argument('--clusters', type=str, default='auto',
                       help='Number of clusters ("auto" or int)')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--output_prefix', type=str, default='clustering_results',
                       help='Output file prefix')
    
    args = parser.parse_args()
    
    # Parse cluster number
    n_clusters = 'auto' if args.clusters == 'auto' else int(args.clusters)
    
    pipeline = PCAClusteringPipeline(
        n_components=args.components,
        clustering_method=args.method,
        n_clusters=n_clusters
    )
    
    # Load data
    df = pipeline.load_data(args.file)
    X = df.select_dtypes(include=[np.number])
    
    # Run pipeline
    results_df, metrics, X_pca = pipeline.run_pipeline(
        X, original_data=df, prefix=args.output_prefix, plot=args.plot
    )
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()