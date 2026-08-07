import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


def plot_histograms(data, title="Feature Distributions"):

    data.hist(
        figsize=(15, 10),
        bins=30
    )

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_boxplots(data, title="Boxplots"):

    plt.figure(figsize=(15, 8))

    data.boxplot(rot=45)

    plt.title(title)
    plt.show()


def plot_correlation(data):

    plt.figure(figsize=(10, 7))

    sns.heatmap(
        data.corr(),
        annot=True,
        fmt=".2f"
    )

    plt.title("Correlation Heatmap")
    plt.show()


def plot_elbow(k_values, wcss):

    plt.figure(figsize=(8, 5))

    plt.plot(
        k_values,
        wcss,
        marker="o"
    )

    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("WCSS (Inertia)")
    plt.title("Elbow Method")

    plt.show()


def plot_silhouette(k_values, scores):

    plt.figure(figsize=(8, 5))

    plt.plot(
        k_values,
        scores,
        marker="o"
    )

    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score")

    plt.show()


def plot_cluster_distribution(data):

    plt.figure(figsize=(7, 5))

    sns.countplot(
        x="Cluster",
        data=data
    )

    plt.title("Number of Customers in Each Cluster")

    plt.show()


def plot_cluster_profile(cluster_profile):

    plt.figure(figsize=(10, 6))

    sns.heatmap(
        cluster_profile,
        annot=True,
        fmt=".0f"
    )

    plt.title("Average Spending by Cluster")

    plt.show()


def plot_pca(data_scaled, labels):

    pca = PCA(n_components=2)

    data_pca = pca.fit_transform(data_scaled)

    plt.figure(figsize=(8, 6))

    sns.scatterplot(
        x=data_pca[:, 0],
        y=data_pca[:, 1],
        hue=labels,
        palette="deep"
    )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("Customer Clusters using PCA")

    plt.show()

    explained_variance = pca.explained_variance_ratio_

    print(
        f"PC1 variance: {explained_variance[0]:.2%}"
    )

    print(
        f"PC2 variance: {explained_variance[1]:.2%}"
    )