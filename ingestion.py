import os
import pypdf
import json

# Import our custom modules
import utils
import vector_db
import knowledge_graph


def _extract_text_from_pdf(file_path: str) -> str:
    """Extracts text content from a PDF file."""
    try:
        reader = pypdf.PdfReader(file_path)
        text = "".join(page.extract_text() for page in reader.pages)
        return text
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
        return ""

def _extract_text_from_txt(file_path: str) -> str:
    """Extracts text content from a .txt file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading TXT {file_path}: {e}")
        return ""

def _extract_text_from_image(file_path: str) -> str:
    """Extracts text from an image using OCR."""
    return utils.ocr_from_image(file_path)

def _transcribe_audio(file_path: str) -> str:
    """Transcribes an audio file."""
    # Note: This uses the simple speech_recognition library. For higher accuracy
    # with various audio formats, integrating a local Whisper model would be better.
    # This might require converting the input audio to .wav format first.
    return utils.transcribe_audio(file_path)


# --- MAIN INGESTION ORCHESTRATOR ---

def ingest_file(file_path: str, collection_name: str, qdrant_client, neo4j_driver):
    """
    Orchestrates the ingestion process for a single file.

    1.  Determines the file type.
    2.  Extracts raw text content.
    3.  Chunks the text.
    4.  Upserts text chunks and their embeddings to Qdrant.
    5.  Extracts entities and relationships from each chunk.
    6.  Updates the Neo4j knowledge graph with the extracted data.

    Args:
        file_path (str): The path to the file to be ingested.
        collection_name (str): The name of the Qdrant collection.
        qdrant_client: An active Qdrant client instance.
        neo4j_driver: An active Neo4j driver instance.
    """
    filename = os.path.basename(file_path)
    print(f"\n--- Starting ingestion for: {filename} ---")

    # 1. & 2. Determine file type and extract text
    file_extension = os.path.splitext(filename)[1].lower()
    raw_text = ""
    
    if file_extension == '.pdf':
        raw_text = _extract_text_from_pdf(file_path)
    elif file_extension == '.txt':
        raw_text = _extract_text_from_txt(file_path)
    elif file_extension in ['.jpg', '.jpeg', '.png']:
        raw_text = _extract_text_from_image(file_path)
    elif file_extension in ['.mp3', '.wav']:
        # Note: You might need to handle file conversion for some audio formats
        raw_text = _transcribe_audio(file_path)
    else:
        print(f"Unsupported file type: {file_extension}. Skipping.")
        return

    if not raw_text or raw_text.isspace():
        print(f"No text extracted from {filename}. Skipping further processing.")
        return

    print(f"Successfully extracted {len(raw_text)} characters of text.")

    # 3. Chunk the text
    text_chunks = utils.chunk_text(raw_text)
    print(f"Text split into {len(text_chunks)} chunks.")

    # 4. Upsert chunks to Qdrant (Vector Database)
    # We create metadata to link each chunk back to its source file
    metadatas = [{"source_file": filename, "chunk_index": i} for i in range(len(text_chunks))]
    vector_db.upsert_chunks(qdrant_client, collection_name, text_chunks, metadatas)

    # 5. & 6. Extract entities and update Knowledge Graph
    print("Extracting entities and updating knowledge graph...")
    llm = utils.get_llm()
    for i, chunk in enumerate(text_chunks):
        print(f"Processing chunk {i+1}/{len(text_chunks)} for KG...")
        # Get structured data (entities, relationships) from the LLM
        structured_data_str = utils.extract_entities_and_relationships(chunk, llm)
        
        # Update the knowledge graph with the new data
        knowledge_graph.update_graph_from_json(neo4j_driver, structured_data_str, filename)
    
    print(f"--- Finished ingestion for: {filename} ---")


# --- Example usage for standalone testing ---
if __name__ == '__main__':
    # This block allows you to test the ingestion script directly.
    # Make sure your .env file is set up correctly.
    utils.load_environment()
    
    # --- Configuration ---
    QDRANT_COLLECTION_NAME = "multimodal_rag_collection"
    # Create a dummy file for testing
    TEST_FILE_PATH = "dummy_document.txt"
    with open(TEST_FILE_PATH, "w") as f:
        f.write("John Doe is the CEO of Global Tech Inc., a company based in Silicon Valley. ")
        f.write("Global Tech Inc. announced a new partnership with AI Solutions LLC. ")
        f.write("This partnership was announced in a press release last Tuesday.")

    # --- Initialize clients ---
    qdrant_client = None
    neo4j_driver = None
    try:
        qdrant_client = vector_db.get_qdrant_client()
        neo4j_driver = knowledge_graph.get_neo4j_driver()

        # Ensure the Qdrant collection exists
        vector_db.create_collection_if_not_exists(qdrant_client, QDRANT_COLLECTION_NAME)

        # --- Run the ingestion process ---
        ingest_file(
            file_path=TEST_FILE_PATH,
            collection_name=QDRANT_COLLECTION_NAME,
            qdrant_client=qdrant_client,
            neo4j_driver=neo4j_driver
        )

        print("\nTest ingestion complete. Check your Qdrant and Neo4j databases.")

    except Exception as e:
        print(f"An error occurred during the test run: {e}")
    finally:
        # --- Clean up ---
        if neo4j_driver:
            knowledge_graph.close_driver(neo4j_driver)
        if os.path.exists(TEST_FILE_PATH):
            os.remove(TEST_FILE_PATH)