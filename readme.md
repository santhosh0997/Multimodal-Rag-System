# Multimodal Enterprise RAG System

This project is a prototype of an Enterprise Retrieval-Augmented Generation (RAG) system, developed as part of a 72-hour technical challenge. It ingests multimodal data (text, images), builds a knowledge graph and a vector database, and uses a hybrid search mechanism to provide accurate, context-aware answers to user queries.

## Key Features

-   **Multimodal Ingestion:** Processes `.pdf`, `.txt`, and image files (`.jpg`, `.png`) using OCR.
-   **Dual Data Storage:**
    -   **Knowledge Graph (Neo4j):** Stores structured entities and their relationships for precise, factual lookups.
    -   **Vector Database (Qdrant):** Stores text embeddings for fast and scalable semantic search.
-   **Hybrid Search:** Combines the strengths of graph traversal and semantic search to retrieve the most relevant context.
-   **LLM Integration:** Uses Google's Gemini models for entity extraction and final answer generation.
-   **Interactive UI:** A user-friendly web interface built with Streamlit for file uploads and Q&A.
-   **Evaluation-First Dashboard:** Includes an automated evaluation suite using DeepEval, with results visualized in a dedicated dashboard tab to measure system quality and prevent hallucinations.

## Tech Stack

-   **Backend:** Python, LangChain, Streamlit
-   **Databases:** Qdrant, Neo4j
-   **LLM & Embeddings:** Google Gemini
-   **Evaluation:** DeepEval
-   **Containerization:** Docker

---

##  Prerequisites

Before you begin, ensure you have the following installed on your system:

1.  **Python (3.9+):** [Download Python](https://www.python.org/downloads/)
2.  **Docker & Docker Desktop:** Required to run the Qdrant and Neo4j databases. [Install Docker](https://www.docker.com/products/docker-desktop/)
3.  **Tesseract OCR Engine:** The system needs this for processing text from images.
    -   **macOS:** `brew install tesseract`
    -   **Ubuntu/Debian:** `sudo apt-get install tesseract-ocr`
    -   **Windows:** Follow the installation guide at [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
4.  **Google AI API Key:** You need an API key with billing enabled.
    -   Get it from [Google AI Studio](https://aistudio.google.com/).
    -   Ensure you have **enabled billing** and the **Vertex AI API** in your Google Cloud project.

---

## 🚀 Step-by-Step Setup Instructions


### 1. Create and Activate a Virtual Environment

It's highly recommended to use a virtual environment to manage project dependencies.

-   **Create the environment:**
    ```bash
    python -m venv venv
    ```
-   **Activate the environment:**
    -   **macOS / Linux:**
        ```bash
        source venv/bin/activate
        ```
    -   **Windows:**
        ```bash
        .\venv\Scripts\activate
        ```

### 3. Install Dependencies

Install all the required Python packages from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Now, open the `.env` file and fill in your credentials:

```dotenv
# .env Configuration File

# --- Google Gemini API Configuration ---
GOOGLE_API_KEY="Your-Google-AI-API-Key"

# --- Qdrant Vector Database Configuration ---
# Default for a local Docker container
QDRANT_HOST="http://localhost:6333"
QDRANT_API_KEY=""

# --- Neo4j Knowledge Graph Configuration ---
# Default for a local Docker container or Neo4j Desktop
NEO4J_URI="neo4j://localhost:7687"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="YourStrongNeo4jPassword" # Use the password you set
```

---

## 🏃‍♀️ Running the System

### 1. Start the Backend Databases

The Qdrant and Neo4j databases must be running before you launch the application. Open your terminal and run the following Docker commands.

-   **Start Qdrant:**
    This command will start Qdrant and create a `qdrant_storage` folder in your project directory to persist data.
    ```bash
    docker run -p 6333:6333 \
        --name qdrant-db \
        -v "$(pwd)/qdrant_storage:/qdrant/storage" \
        qdrant/qdrant
    ```

-   **Start Neo4j:**
    Replace `YourStrongNeo4jPassword` with the same password you set in your `.env` file.
    ```bash
    docker run \
        --name neo4j-db \
        -p 7474:7474 -p 7687:7687 \
        -d \
        -e NEO4J_AUTH=neo4j/YourStrongNeo4jPassword \
        neo4j:latest
    ```
    *Alternatively, you can use Neo4j Desktop.*

### 2. Run the Streamlit Application

Ensure your virtual environment is still active.

```bash
streamlit run app.py
```

Your web browser will automatically open a new tab with the application running at `http://localhost:8501`.

### 3. Run the Evaluation Suite

To populate the Evaluation Dashboard, you need to run the evaluation script. This will run a series of tests and generate an `eval_results.json` file.

Open a **new terminal**, activate the virtual environment again, and run:

```bash
python evaluation.py
```
