import os
from qdrant_client import QdrantClient, models
from utils import get_embedding_model


def get_qdrant_client():
    """
    Initializes and returns the Qdrant client.
    Expects QDRANT_HOST and QDRANT_API_KEY to be set in the environment.
    """
    host = os.getenv("QDRANT_HOST")
    api_key = os.getenv("QDRANT_API_KEY")

    if not host:
        raise ValueError("QDRANT_HOST not found in environment variables.")

    # You can connect to a local Qdrant instance or Qdrant Cloud
    client = QdrantClient(url=host, api_key=api_key)
    return client


def create_collection_if_not_exists(client: QdrantClient, collection_name: str):
    """
    Creates a new collection in Qdrant if it doesn't already exist.

    Args:
        client (QdrantClient): The Qdrant client instance.
        collection_name (str): The name of the collection to create.
    """
    try:
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]

        if collection_name not in collection_names:
            print(f"Creating collection: {collection_name}")
            # The vector size for OpenAI's text-embedding-ada-002 is 1536
            # client.recreate_collection(
            #     collection_name=collection_name,
            #     vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
            # )
            client.recreate_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=3072, distance=models.Distance.COSINE),
            )
            print(f"Collection '{collection_name}' created successfully.")
        else:
            print(f"Collection '{collection_name}' already exists.")

    except Exception as e:
        print(f"An error occurred while creating collection: {e}")

# --- DATA UPSERTION ---

def upsert_chunks(client: QdrantClient, collection_name: str, chunks: list[str], metadatas: list[dict]):
    """
    Embeds text chunks and upserts them into a Qdrant collection.

    Args:
        client (QdrantClient): The Qdrant client instance.
        collection_name (str): The name of the collection.
        chunks (list[str]): The list of text chunks to embed and store.
        metadatas (list[dict]): A list of metadata dictionaries, one for each chunk.
    """
    if not chunks:
        print("No chunks to upsert.")
        return

    print(f"Upserting {len(chunks)} chunks to '{collection_name}'...")
    embedding_model = get_embedding_model()
    
    try:
        # Embed all chunks in a single call for efficiency
        vectors = embedding_model.embed_documents(chunks)

        # Prepare points for Qdrant
        points = [
            models.PointStruct(
                id=i,  # Using a simple integer ID. For production, consider UUIDs.
                vector=vector,
                payload={"text": chunk, **metadata}
            )
            for i, (vector, chunk, metadata) in enumerate(zip(vectors, chunks, metadatas))
        ]

        # Upsert in batches
        client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True  # Wait for the operation to complete
        )
        print(f"Successfully upserted {len(chunks)} chunks.")
    except Exception as e:
        print(f"An error occurred during upsertion: {e}")


# --- SEMANTIC SEARCH ---

def semantic_search(client: QdrantClient, collection_name: str, query_text: str, limit: int = 5):
    """
    Performs a semantic search in a Qdrant collection.

    Args:
        client (QdrantClient): The Qdrant client instance.
        collection_name (str): The name of the collection to search in.
        query_text (str): The user's query text.
        limit (int): The maximum number of results to return.

    Returns:
        list: A list of search results (hit objects).
    """
    if not query_text:
        return []
        
    embedding_model = get_embedding_model()
    
    try:
        # Create a vector embedding for the query
        query_vector = embedding_model.embed_query(query_text)

        # Perform the search
        search_results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=True  # Include the metadata in the results
        )
        
        return search_results
    except Exception as e:
        print(f"An error occurred during semantic search: {e}")
        return []