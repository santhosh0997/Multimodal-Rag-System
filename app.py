import os
import streamlit as st
import tempfile

import utils
import vector_db
import knowledge_graph
import ingestion
import hybrid_search

st.set_page_config(
    page_title="Multimodal Enterprise RAG",
    page_icon="🤖",
    layout="wide"
)

QDRANT_COLLECTION_NAME = "multimodal_rag"
UPLOAD_DIR = "temp_uploads"

utils.load_environment()

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

@st.cache_resource
def get_db_connections():
    """
    Initializes and caches the database connections.
    Streamlit's cache_resource ensures this function runs only once.
    """
    print("Initializing database connections...")
    qdrant_client = vector_db.get_qdrant_client()
    neo4j_driver = knowledge_graph.get_neo4j_driver()
    vector_db.create_collection_if_not_exists(qdrant_client, QDRANT_COLLECTION_NAME)
    return qdrant_client, neo4j_driver

try:
    qdrant_client, neo4j_driver = get_db_connections()
    st.success("Successfully connected to databases (Qdrant & Neo4j).")
except Exception as e:
    st.error(f"Failed to connect to databases. Please check your .env configuration and ensure services are running. Error: {e}")
    st.stop()


st.title("🧠 Multimodal Enterprise RAG System")
st.markdown("Leveraging Knowledge Graphs and Hybrid Search for advanced Q&A")

# --- SIDEBAR FOR FILE INGESTION ---
with st.sidebar:
    st.header("Upload New Knowledge")
    uploaded_files = st.file_uploader(
        "Upload Files (.pdf, .txt, .jpg, .png, .mp3, .wav)",
        type=['pdf', 'txt', 'jpg', 'jpeg', 'png', 'mp3', 'wav'],
        accept_multiple_files=True
    )

    if st.button("Process Uploaded Files"):
        if uploaded_files:
            with st.spinner("Ingesting files... This may take a moment."):
                for uploaded_file in uploaded_files:
                    # Save the file to a temporary location
                    temp_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    try:
                        # Call the main ingestion function
                        ingestion.ingest_file(
                            file_path=temp_path,
                            collection_name=QDRANT_COLLECTION_NAME,
                            qdrant_client=qdrant_client,
                            neo4j_driver=neo4j_driver
                        )
                        st.success(f"Successfully ingested: {uploaded_file.name}")
                    except Exception as e:
                        st.error(f"Failed to ingest {uploaded_file.name}. Error: {e}")
                    finally:
                        # Clean up the temporary file
                        os.remove(temp_path)
        else:
            st.warning("Please upload at least one file.")


# --- MAIN CHAT INTERFACE ---
st.header("Ask a Question")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Perform hybrid retrieval
            retrieved_context = hybrid_search.hybrid_retrieval(
                prompt,
                QDRANT_COLLECTION_NAME,
                qdrant_client,
                neo4j_driver
            )
            
            # Generate the final response
            final_answer = hybrid_search.generate_response(prompt, retrieved_context)
            
            st.markdown(final_answer)
            
            # Optional: Show the retrieved context for transparency
            with st.expander("View Retrieved Context"):
                st.text(retrieved_context)
                
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": final_answer})