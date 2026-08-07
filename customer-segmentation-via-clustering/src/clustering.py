from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def calculate_wcss(data_scaled, k_range=range(1, 11)):
    """
    Calculate WCSS/Inertia for different values of K.
    """

    wcss = []

    for k in k_range:

        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        kmeans.fit(data_scaled)

        wcss.append(kmeans.inertia_)

    return list(k_range), wcss


def calculate_silhouette_scores(
    data_scaled,
    k_range=range(2, 11)
):
    """
    Calculate silhouette score for different K values.
    """

    scores = []

    for k in k_range:

        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = kmeans.fit_predict(data_scaled)

        score = silhouette_score(
            data_scaled,
            labels
        )

        scores.append(score)

    return list(k_range), scores


def train_kmeans(data_scaled, n_clusters=2):
    """
    Train final K-Means model.
    """

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(data_scaled)

    return kmeans, labels


def evaluate_clustering(data_scaled, labels):

    return silhouette_score(
        data_scaled,
        labels
    )