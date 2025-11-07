import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pytesseract
from PIL import Image
import speech_recognition as sr

def load_environment():
    """Loads environment variables from a .env file."""
    load_dotenv()

def get_google_api_key():
    """Fetches the Google API key from environment variables."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")
    return api_key

def get_llm():
    """Initializes and returns the Google Gemini LLM."""
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=get_google_api_key(), temperature=0)

def get_embedding_model():
    """Initializes and returns the Google embedding model."""
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=get_google_api_key())


# --- TEXT PROCESSING ---

def chunk_text(text, chunk_size=1000, chunk_overlap=100):
    """
    Splits a long text into smaller chunks.
    
    Args:
        text (str): The input text.
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The number of characters to overlap between chunks.
        
    Returns:
        list[str]: A list of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# --- MULTIMODAL EXTRACTION ---

def ocr_from_image(image_path):
    """
    Performs Optical Character Recognition (OCR) on an image file.
    
    Requires Tesseract OCR engine to be installed on the system.
    
    Args:
        image_path (str): The path to the image file.
        
    Returns:
        str: The extracted text from the image.
    """
    try:
        with Image.open(image_path) as img:
            text = pytesseract.image_to_string(img)
            return text
    except Exception as e:
        print(f"Error during OCR: {e}")
        return ""

def transcribe_audio(audio_path):
    """
    Transcribes an audio file to text using Google's Web Speech API.
    
    Note: This is a simple implementation. For production, consider robust
    services like OpenAI's Whisper, either via API or a local model.
    
    Args:
        audio_path (str): The path to the audio file (e.g., .wav).
        
    Returns:
        str: The transcribed text.
    """
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            return text
    except Exception as e:
        print(f"Error during audio transcription: {e}")
        return ""

# --- ENTITY EXTRACTION (LLM-based) ---

def extract_entities_and_relationships(text, llm):
    """
    Uses an LLM to extract entities and their relationships from a text chunk.
    
    Args:
        text (str): The text chunk to process.
        llm: An initialized language model instance.
        
    Returns:
        str: A string containing the extracted entities and relationships,
             often in a structured format like JSON (as requested in the prompt).
    """
    prompt = f"""
    From the following text, extract named entities (like people, organizations, locations)
    and the relationships between them. Structure the output as a list of JSON objects,
    where each object has 'source', 'relationship', and 'target'.
    
    Example:
    Text: "John Smith works for Acme Corp, which is located in New York."
    Output:
    [
      {{
        "source": "John Smith",
        "relationship": "WORKS_FOR",
        "target": "Acme Corp"
      }},
      {{
        "source": "Acme Corp",
        "relationship": "LOCATED_IN",
        "target": "New York"
      }}
    ]

    Now, analyze this text:
    ---
    {text}
    ---
    """
    
    try:
        response = llm.invoke(prompt)
        return response
    except Exception as e:
        print(f"Error during entity extraction: {e}")
        return "[]" # Return empty list as a string

# You would call load_environment() at the start of your main script (app.py)
# load_environment()