"""
Image Search using Qdrant Vector Database
Search for similar images in the malaria collection.
"""

import argparse
import os
from qdrant_client import QdrantClient
from embedding import ImageEmbedder


def search_image(image_path: str, top_k: int = 5, host: str = "localhost", port: int = 6333):
    """
    Search for similar images in Qdrant database.
    
    Args:
        image_path: Path to the query image
        top_k: Number of similar images to return
        host: Qdrant server host
        port: Qdrant server port
    """
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return
    
    # Initialize embedder
    embedder = ImageEmbedder("models/resnet34-v2-7.onnx")
    
    # Initialize Qdrant client
    client = QdrantClient(host=host, port=port)
    
    # Collection name
    collection_name = "malaria"
    
    # Check if collection exists
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]
    
    if collection_name not in collection_names:
        print(f"Error: Collection '{collection_name}' not found in Qdrant database.")
        return
    
    # Generate embedding for query image
    print(f"Generating embedding for: {image_path}")
    query_embedding = embedder.get_embedding(image_path)
    
    # Search in Qdrant
    results = client.query_points(
        collection_name=collection_name,
        query=query_embedding.tolist(),
        limit=top_k
    ).points
    
    # Display results
    print("\n" + "=" * 60)
    print(f"Search Results for: {os.path.basename(image_path)}")
    print("=" * 60)
    
    if not results:
        print("No similar images found.")
        return
    
    print(f"\nTop {len(results)} similar images:\n")
    print(f"{'Rank':<6} {'Filename':<25} {'Score':<10}")
    print("-" * 45)
    
    for rank, result in enumerate(results, 1):
        filename = result.payload.get("filename", "Unknown")
        score = result.score
        print(f"{rank:<6} {filename:<25} {score:.4f}")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Search for similar images in Qdrant malaria collection"
    )
    parser.add_argument(
        "image",
        type=str,
        help="Path to the query image"
    )
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=5,
        help="Number of similar images to return (default: 5)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Qdrant server host (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=6333,
        help="Qdrant server port (default: 6333)"
    )
    
    args = parser.parse_args()
    
    search_image(
        image_path=args.image,
        top_k=args.top_k,
        host=args.host,
        port=args.port
    )


if __name__ == "__main__":
    main()
