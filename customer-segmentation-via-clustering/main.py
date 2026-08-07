from src.data_ingestion import load_data

from src.preprocessing import (
    clean_data,
    transform_data,
    scale_data
)

from src.clustering import (
    calculate_wcss,
    calculate_silhouette_scores,
    train_kmeans,
    evaluate_clustering
)

from src.visualization import (
    plot_histograms,
    plot_boxplots,
    plot_correlation,
    plot_elbow,
    plot_silhouette,
    plot_cluster_distribution,
    plot_cluster_profile,
    plot_pca
)


DATA_PATH = (
    r"C:\Users\rahul\Documents\Programming\Github Projects"
    r"\Classic-ML-Projects\customer-segmentation-via-clustering"
    r"\data\customer_data.csv"
)


def main():

    # 1. Load data

    data = load_data(DATA_PATH)

    print("Original shape:", data.shape)

    data = clean_data(data)

    print("Shape after cleaning:", data.shape)

    # 3. EDA

    print("\nSummary statistics:")
    print(data.describe().T)

    print("\nSkewness:")
    print(data.skew())

    plot_correlation(data)

    plot_histograms(
        data,
        title="Original Feature Distributions"
    )

    plot_boxplots(
        data,
        title="Original Feature Boxplots"
    )

    # 4. Log transformation

    data_log = transform_data(data)

    print("\nSkewness after log transformation:")
    print(data_log.skew())

    plot_histograms(
        data_log,
        title="Log-Transformed Distributions"
    )

    plot_boxplots(
        data_log,
        title="Log-Transformed Boxplots"
    )

    # 5. Scaling

    data_scaled, scaler = scale_data(data_log)

    # 6. Elbow Method

    k_values, wcss = calculate_wcss(
        data_scaled,
        range(1, 11)
    )

    plot_elbow(k_values, wcss)

    # 7. Silhouette Analysis

    k_values, silhouette_scores = (
        calculate_silhouette_scores(
            data_scaled,
            range(2, 11)
        )
    )

    plot_silhouette(
        k_values,
        silhouette_scores
    )

    best_k = k_values[
        silhouette_scores.index(
            max(silhouette_scores)
        )
    ]

    print(f"\nBest K according to silhouette score: {best_k}")

    # 8. Train final K-Means

    kmeans, labels = train_kmeans(
        data_scaled,
        n_clusters=best_k
    )

    # 9. Final evaluation

    final_score = evaluate_clustering(
        data_scaled,
        labels
    )

    print(
        f"Final Silhouette Score: "
        f"{final_score:.4f}"
    )

    # 10. Add cluster labels

    data_clustered = data.copy()

    data_clustered["Cluster"] = labels

    # 11. Cluster distribution

    print("\nCustomers per cluster:")
    print(
        data_clustered["Cluster"]
        .value_counts()
        .sort_index()
    )

    print("\nCluster percentages:")
    print(
        data_clustered["Cluster"]
        .value_counts(
            normalize=True
        )
        .sort_index()
        .mul(100)
    )

    plot_cluster_distribution(
        data_clustered
    )

    # 12. Cluster profiling

    cluster_profile = (
        data_clustered
        .groupby("Cluster")
        .mean()
    )

    print("\nCluster Profile:")
    print(cluster_profile)

    plot_cluster_profile(
        cluster_profile
    )

    # 13. PCA visualization

    plot_pca(
        data_scaled,
        labels
    )


if __name__ == "__main__":
    main()