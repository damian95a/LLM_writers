from typing import Dict, List, Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    PointStruct,
    ScoredPoint,
)
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt


def plot_tsne_by_author_allowed(
    qdrant_client, collection_name, model_name, max_points=1000
):
    print(f"Pobieranie embeddingów do t-SNE (max {max_points})...")
    embeddings = []
    authors = []
    allowed_authors = {"sienkiewicz_henryk", "słowacki_juliusz"}
    scroll_filter_conditions = [
        FieldCondition(key="model_name", match=MatchValue(value=model_name))
    ]
    current_offset = None
    scroll_limit = 250
    total = 0

    while total < max_points:
        points, next_page_offset = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(must=scroll_filter_conditions),
            limit=min(scroll_limit, max_points - total),
            offset=current_offset,
            with_payload=["author"],
            with_vectors=True,
        )
        if not points:
            break
        for point in points:
            author = point.payload.get("author", "unknown")
            if author not in allowed_authors:
                continue
            vector_data = point.vector
            if isinstance(vector_data, dict):
                vec = np.array(vector_data.get("", []), dtype=np.float32)
            else:
                vec = np.array(vector_data, dtype=np.float32)
            if vec.size == 0:
                continue
            embeddings.append(vec)
            authors.append(author)
            total += 1
            if total >= max_points:
                break
        current_offset = next_page_offset
        if current_offset is None:
            break

    X = np.array(embeddings)
    if X.shape[0] == 0:
        print("Brak embeddingów do wizualizacji.")
        return

    print(f"Redukcja wymiarów t-SNE ({X.shape[0]} punktów)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_2d = tsne.fit_transform(X)

    unique_authors = list(sorted(set(authors)))
    color_map = {author: idx for idx, author in enumerate(unique_authors)}
    colors = [color_map[author] for author in authors]

    plt.figure(figsize=(12, 9))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, cmap="tab20", alpha=0.7)

    handles = []
    for author, idx in color_map.items():
        handles.append(
            plt.Line2D(
                [],
                [],
                marker="o",
                color="w",
                markerfacecolor=plt.cm.tab20(idx / max(1, len(unique_authors) - 1)),
                label=author,
                markersize=8,
            )
        )
    plt.legend(
        handles=handles,
        title="author",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=8,
    )

    for i in range(min(30, len(authors))):
        plt.annotate(authors[i], (X_2d[i, 0], X_2d[i, 1]), fontsize=8, alpha=0.7)
    plt.title(
        f"t-SNE embeddingów z Qdrant ({model_name}) wg autora (tylko Sienkiewicz i Słowacki)"
    )
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.show()


def fetch_all_embeddings_for_tsne(
    qdrant_client, collection_name, model_name, max_points=1000
):
    embeddings = []
    labels = []
    scroll_filter_conditions = [
        FieldCondition(key="model_name", match=MatchValue(value=model_name))
    ]
    current_offset = None
    scroll_limit = 250
    total = 0

    while total < max_points:
        points, next_page_offset = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(must=scroll_filter_conditions),
            limit=min(scroll_limit, max_points - total),
            offset=current_offset,
            with_payload=["source_file"],
            with_vectors=True,
        )
        if not points:
            break
        for point in points:
            vector_data = point.vector
            if isinstance(vector_data, dict):
                vec = np.array(vector_data.get("", []), dtype=np.float32)
            else:
                vec = np.array(vector_data, dtype=np.float32)
            if vec.size == 0:
                continue
            embeddings.append(vec)
            labels.append(point.payload.get("source_file", "unknown"))
            total += 1
            if total >= max_points:
                break
        current_offset = next_page_offset
        if current_offset is None:
            break
    return np.array(embeddings), labels


def plot_tsne_of_embeddings(
    qdrant_client, collection_name, model_name, max_points=1000
):
    print(f"Pobieranie embeddingów do t-SNE (max {max_points})...")
    X, labels = fetch_all_embeddings_for_tsne(
        qdrant_client, collection_name, model_name, max_points
    )
    if X.shape[0] == 0:
        print("Brak embeddingów do wizualizacji.")
        return
    print(f"Redukcja wymiarów t-SNE ({X.shape[0]} punktów)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_2d = tsne.fit_transform(X)

    # Przypisz kolor do każdego unikalnego source_file
    unique_labels = list(sorted(set(labels)))
    color_map = {label: idx for idx, label in enumerate(unique_labels)}
    colors = [color_map[label] for label in labels]

    plt.figure(figsize=(12, 9))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, cmap="tab20", alpha=0.7)

    # Dodaj legendę z nazwami plików
    handles = []
    for label, idx in color_map.items():
        handles.append(
            plt.Line2D(
                [],
                [],
                marker="o",
                color="w",
                markerfacecolor=plt.cm.tab20(idx / max(1, len(unique_labels) - 1)),
                label=label,
                markersize=8,
            )
        )
    plt.legend(
        handles=handles,
        title="source_file",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=8,
    )

    # Opcjonalnie: wyświetl etykiety plików dla pierwszych N punktów
    for i in range(min(30, len(labels))):
        plt.annotate(labels[i], (X_2d[i, 0], X_2d[i, 1]), fontsize=8, alpha=0.7)
    plt.title(f"t-SNE embeddingów z Qdrant ({model_name})")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.show()


def plot_tsne_by_author(qdrant_client, collection_name, model_name, max_points=1000):
    print(f"Pobieranie embeddingów do t-SNE (max {max_points})...")
    embeddings = []
    authors = []
    scroll_filter_conditions = [
        FieldCondition(key="model_name", match=MatchValue(value=model_name))
    ]
    current_offset = None
    scroll_limit = 250
    total = 0

    while total < max_points:
        points, next_page_offset = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(must=scroll_filter_conditions),
            limit=min(scroll_limit, max_points - total),
            offset=current_offset,
            with_payload=["author"],
            with_vectors=True,
        )
        if not points:
            break
        for point in points:
            vector_data = point.vector
            if isinstance(vector_data, dict):
                vec = np.array(vector_data.get("", []), dtype=np.float32)
            else:
                vec = np.array(vector_data, dtype=np.float32)
            if vec.size == 0:
                continue
            embeddings.append(vec)
            authors.append(point.payload.get("author", "unknown"))
            total += 1
            if total >= max_points:
                break
        current_offset = next_page_offset
        if current_offset is None:
            break

    X = np.array(embeddings)
    if X.shape[0] == 0:
        print("Brak embeddingów do wizualizacji.")
        return

    print(f"Redukcja wymiarów t-SNE ({X.shape[0]} punktów)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_2d = tsne.fit_transform(X)

    # Przypisz kolor do każdego unikalnego autora
    unique_authors = list(sorted(set(authors)))
    color_map = {author: idx for idx, author in enumerate(unique_authors)}
    colors = [color_map[author] for author in authors]

    plt.figure(figsize=(12, 9))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, cmap="tab20", alpha=0.7)

    # Dodaj legendę z nazwami autorów
    handles = []
    for author, idx in color_map.items():
        handles.append(
            plt.Line2D(
                [],
                [],
                marker="o",
                color="w",
                markerfacecolor=plt.cm.tab20(idx / max(1, len(unique_authors) - 1)),
                label=author,
                markersize=8,
            )
        )
    plt.legend(
        handles=handles,
        title="author",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=8,
    )

    # Opcjonalnie: wyświetl etykiety autorów dla pierwszych N punktów
    for i in range(min(30, len(authors))):
        plt.annotate(authors[i], (X_2d[i, 0], X_2d[i, 1]), fontsize=8, alpha=0.7)
    plt.title(f"t-SNE embeddingów z Qdrant ({model_name}) wg autora")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.show()


def plot_tsne_of_authors_avg(
    qdrant_client, collection_name, model_name, max_points=1000
):
    """
    Średnia embeddingów dla każdego autora i wizualizacja t-SNE (każdy punkt to autor).
    """
    print(f"Pobieranie embeddingów do t-SNE (max {max_points})...")
    # 1. Pobierz embeddingi i autorów
    author_vectors: Dict[str, list] = {}
    scroll_filter_conditions = [
        FieldCondition(key="model_name", match=MatchValue(value=model_name))
    ]
    current_offset = None
    scroll_limit = 250
    total = 0

    while total < max_points:
        points, next_page_offset = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(must=scroll_filter_conditions),
            limit=min(scroll_limit, max_points - total),
            offset=current_offset,
            with_payload=["author"],
            with_vectors=True,
        )
        if not points:
            break
        for point in points:
            vector_data = point.vector
            if isinstance(vector_data, dict):
                vec = np.array(vector_data.get("", []), dtype=np.float32)
            else:
                vec = np.array(vector_data, dtype=np.float32)
            if vec.size == 0:
                continue
            author = point.payload.get("author", "unknown")
            if author not in author_vectors:
                author_vectors[author] = []
            author_vectors[author].append(vec)
            total += 1
            if total >= max_points:
                break
        current_offset = next_page_offset
        if current_offset is None:
            break

    # 2. Oblicz średnie embeddingi dla każdego autora
    avg_embeddings = []
    author_labels = []
    for author, vectors in author_vectors.items():
        stacked = np.stack(vectors)
        avg_vec = np.mean(stacked, axis=0)
        avg_embeddings.append(avg_vec)
        author_labels.append(author)

    X = np.array(avg_embeddings)
    if X.shape[0] == 0:
        print("Brak embeddingów do wizualizacji.")
        return

    print(f"Redukcja wymiarów t-SNE ({X.shape[0]} autorów)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, X.shape[0] - 1))
    X_2d = tsne.fit_transform(X)

    # Kolory dla autorów
    unique_authors = list(sorted(set(author_labels)))
    color_map = {author: idx for idx, author in enumerate(unique_authors)}
    colors = [color_map[author] for author in author_labels]

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        X_2d[:, 0], X_2d[:, 1], c=colors, cmap="tab20", alpha=0.8, s=120
    )

    # Legenda
    handles = []
    for author, idx in color_map.items():
        handles.append(
            plt.Line2D(
                [],
                [],
                marker="o",
                color="w",
                markerfacecolor=plt.cm.tab20(idx / max(1, len(unique_authors) - 1)),
                label=author,
                markersize=10,
            )
        )
    plt.legend(
        handles=handles,
        title="author",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=10,
    )

    # Etykiety autorów
    for i, author in enumerate(author_labels):
        plt.annotate(author, (X_2d[i, 0], X_2d[i, 1]), fontsize=10, alpha=0.8)
    plt.title(f"t-SNE średnich embeddingów autorów ({model_name})")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.show()


def get_filtered_data_after_tsne(
    qdrant_client: QdrantClient,
    collection_name: str,
    model_name: str,
    authors_filter: set = None,
    max_points: int = 5000,
) -> List[Dict[str, Any]]:
    embeddings = []
    authors = []
    scroll_filter_conditions = [
        FieldCondition(key="model_name", match=MatchValue(value=model_name))
    ]
    current_offset = None
    scroll_limit = 250
    total = 0

    while total < max_points:
        points, next_page_offset = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(must=scroll_filter_conditions),
            limit=min(scroll_limit, max_points - total),
            offset=current_offset,
            with_payload=["author"],
            with_vectors=True,
        )
        if not points:
            break
        for point in points:
            author = point.payload.get("author", "unknown")
            if authors_filter is not None and author not in authors_filter:
                continue
            vector_data = point.vector
            if isinstance(vector_data, dict):
                vec = np.array(vector_data.get("", []), dtype=np.float32)
            else:
                vec = np.array(vector_data, dtype=np.float32)
            if vec.size == 0:
                continue
            embeddings.append(vec)
            authors.append(author)
            total += 1
            if total >= max_points:
                break
        current_offset = next_page_offset
        if current_offset is None:
            break

    x = np.array(embeddings)
    if x.shape[0] == 0:
        print("Brak embeddingów do wizualizacji.")
        return

    # print(f"Redukcja wymiarów t-SNE ({x.shape[0]} punktów)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    x_2d = tsne.fit_transform(x)

    results = [
        {"x": float(x), "y": float(y), "author": author}
        for (x, y), author in zip(x_2d, authors)
    ]
    return results


def get_tsne_of_embeddings(qdrant_client, collection_name, model_name, max_points=1000):
    print(f"Pobieranie embeddingów do t-SNE (max {max_points})...")
    X, labels = fetch_all_embeddings_for_tsne(
        qdrant_client, collection_name, model_name, max_points
    )
    if X.shape[0] == 0:
        print("Brak embeddingów do wizualizacji.")
        return
    print(f"Redukcja wymiarów t-SNE ({X.shape[0]} punktów)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    x_2d = tsne.fit_transform(X)

    results = [
        {"x": float(x), "y": float(y), "label": label}
        for (x, y), label in zip(x_2d, labels)
    ]
    return results


def get_tsne_of_authors_avg(
    qdrant_client, collection_name, model_name, max_points=1000
):
    """
    Średnia embeddingów dla każdego autora i wizualizacja t-SNE (każdy punkt to autor).
    """
    print(f"Pobieranie embeddingów do t-SNE (max {max_points})...")
    # 1. Pobierz embeddingi i autorów
    author_vectors: Dict[str, list] = {}
    scroll_filter_conditions = [
        FieldCondition(key="model_name", match=MatchValue(value=model_name))
    ]
    current_offset = None
    scroll_limit = 250
    total = 0

    while total < max_points:
        points, next_page_offset = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(must=scroll_filter_conditions),
            limit=min(scroll_limit, max_points - total),
            offset=current_offset,
            with_payload=["author"],
            with_vectors=True,
        )
        if not points:
            break
        for point in points:
            vector_data = point.vector
            if isinstance(vector_data, dict):
                vec = np.array(vector_data.get("", []), dtype=np.float32)
            else:
                vec = np.array(vector_data, dtype=np.float32)
            if vec.size == 0:
                continue
            author = point.payload.get("author", "unknown")
            if author not in author_vectors:
                author_vectors[author] = []
            author_vectors[author].append(vec)
            total += 1
            if total >= max_points:
                break
        current_offset = next_page_offset
        if current_offset is None:
            break

    # 2. Oblicz średnie embeddingi dla każdego autora
    avg_embeddings = []
    author_labels = []
    for author, vectors in author_vectors.items():
        stacked = np.stack(vectors)
        avg_vec = np.mean(stacked, axis=0)
        avg_embeddings.append(avg_vec)
        author_labels.append(author)

    X = np.array(avg_embeddings)
    if X.shape[0] == 0:
        print("Brak embeddingów do wizualizacji.")
        return

    print(f"Redukcja wymiarów t-SNE ({X.shape[0]} autorów)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, X.shape[0] - 1))
    x_2d = tsne.fit_transform(X)

    results = [
        {"x": float(x), "y": float(y), "author": author}
        for (x, y), author in zip(x_2d, author_labels)
    ]
    return results

    # Kolory dla autorów
    # unique_authors = list(sorted(set(author_labels)))
    # color_map = {author: idx for idx, author in enumerate(unique_authors)}
    # colors = [color_map[author] for author in author_labels]
    #
    # plt.figure(figsize=(10, 8))
    # scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, cmap='tab20', alpha=0.8, s=120)
    #
    # # Legenda
    # handles = []
    # for author, idx in color_map.items():
    #     handles.append(plt.Line2D([], [], marker="o", color='w', markerfacecolor=plt.cm.tab20(idx / max(1, len(unique_authors)-1)), label=author, markersize=10))
    # plt.legend(handles=handles, title="author", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    #
    # # Etykiety autorów
    # for i, author in enumerate(author_labels):
    #     plt.annotate(author, (X_2d[i, 0], X_2d[i, 1]), fontsize=10, alpha=0.8)
    # plt.title(f"t-SNE średnich embeddingów autorów ({model_name})")
    # plt.xlabel("t-SNE 1")
    # plt.ylabel("t-SNE 2")
    # plt.tight_layout()
    # plt.show()
