import umap
import matplotlib.pyplot as plt

def run_umap(embeddings, labels):

    # Create UMAP reducer
    reducer = umap.UMAP(
        n_neighbors=20,
        min_dist=0.1,
        n_components=2,
        random_state=42
    )

    # Project embeddings → 2D
    embedding_2d = reducer.fit_transform(embeddings)

    # Plot
    plt.figure(figsize=(8,8))

    plt.scatter(
        embedding_2d[labels == 0, 0],
        embedding_2d[labels == 0, 1],
        s=20,
        label="Good"
    )

    plt.scatter(
        embedding_2d[labels == 1, 0],
        embedding_2d[labels == 1, 1],
        s=20,
        label="Defect"
    )

    plt.scatter(
        embedding_2d[labels == 2, 0],
        embedding_2d[labels == 2, 1],
        s=20,
        label="OOD"
    )

    plt.legend()
    plt.title("UMAP projection of embedding space")
    plt.show()
