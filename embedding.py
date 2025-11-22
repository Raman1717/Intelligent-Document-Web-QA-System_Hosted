import re 
import os 
import pickle 
import requests 
import json 
import faiss 
import numpy as np 
import logging 
from docx import Document 
from typing import List, Tuple, Optional, Union, Dict, Any
from sentence_transformers import SentenceTransformer 
import nltk 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from bs4 import BeautifulSoup 
from urllib.parse import urlparse 
import PyPDF2
import fitz  # PyMuPDF
import mysql.connector
from mysql.connector import Error
import uuid
from datetime import datetime
from PIL import Image
import pytesseract
import io
import psycopg2
import psycopg2.extras
# ------------------ NLTK DOWNLOAD CHECK ------------------ 
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

# Set up logging 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration 
API_KEY = "AIzaSyDL4T66vw6uN0UgsGBxxuTFqVE9Nes84sQ"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={API_KEY}"
DEFAULT_TOP_K = 3

# Tesseract OCR Configuration
# Update this path based on your Tesseract installation
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Database configuration
# DB_CONFIG = {
#     'host': 'localhost',
#     'user': 'root',
#     'password': 'Raman@mysql',
#     'database': 'rag_chat_user'
# }

DB_CONFIG = {
    "host": "ninja-saiga-18700.j77.aws-ap-south-1.cockroachlabs.cloud",
    "user": "raman",
    "password": "meN4ZNAwDyxLKDAy086omA",
    "database": "rag_chat_user",
    "port":26257
}

# ------------------ Image OCR Functions ------------------

def extract_text_from_image(image_data: Union[str, bytes, Image.Image]) -> str:
    """
    Extract text from an image using OCR (Tesseract).
    
    Args:
        image_data: Can be file path, bytes, or PIL Image object
        
    Returns:
        Extracted text from the image
    """
    try:
        # Handle different input types
        if isinstance(image_data, str):
            # File path
            image = Image.open(image_data)
        elif isinstance(image_data, bytes):
            # Bytes data
            image = Image.open(io.BytesIO(image_data))
        elif isinstance(image_data, Image.Image):
            # Already a PIL Image
            image = image_data
        else:
            logger.warning(f"Unsupported image data type: {type(image_data)}")
            return ""
        
        # Convert to RGB if necessary (for better OCR results)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Extract text using Tesseract
        extracted_text = pytesseract.image_to_string(image, lang='eng')
        
        # Clean the extracted text
        cleaned_text = clean_paragraph(extracted_text)
        
        if cleaned_text:
            logger.info(f"Successfully extracted {len(cleaned_text)} characters from image via OCR")
            return cleaned_text
        else:
            logger.warning("No text extracted from image")
            return ""
            
    except Exception as e:
        logger.error(f"Error extracting text from image: {str(e)}")
        return ""

def extract_images_and_text_from_pdf(filename: str) -> Tuple[str, List[str]]:
    """
    Extract both text and images from PDF, then OCR the images.
    
    Returns:
        Tuple of (combined_text, list of image texts)
    """
    text_content = ""
    image_texts = []
    
    try:
        # Open PDF with PyMuPDF for better image extraction
        doc = fitz.open(filename)
        
        for page_num, page in enumerate(doc):
            # Extract regular text
            page_text = page.get_text()
            if page_text:
                cleaned = clean_paragraph(page_text)
                if cleaned:
                    text_content += cleaned + "\n"
            
            # Extract images from page
            image_list = page.get_images(full=True)
            
            for img_index, img_info in enumerate(image_list):
                try:
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Perform OCR on the image
                    image_text = extract_text_from_image(image_bytes)
                    
                    if image_text and len(image_text.strip()) > 10:  # Only add if substantial text found
                        image_texts.append(image_text)
                        logger.info(f"Extracted text from image {img_index + 1} on page {page_num + 1}")
                    
                except Exception as img_error:
                    logger.warning(f"Could not process image {img_index + 1} on page {page_num + 1}: {str(img_error)}")
                    continue
        
        doc.close()
        logger.info(f"Extracted text from {len(image_texts)} images in PDF")
        
    except Exception as e:
        logger.error(f"Error extracting images from PDF: {str(e)}")
    
    return text_content, image_texts

def extract_images_and_text_from_docx(filename: str) -> Tuple[str, List[str]]:
    """
    Extract both text and images from DOCX file, then OCR the images.
    
    Returns:
        Tuple of (combined_text, list of image texts)
    """
    text_content = ""
    image_texts = []
    
    try:
        doc = Document(filename)
        
        # Extract text from paragraphs
        for para in doc.paragraphs:
            cleaned = clean_paragraph(para.text)
            if cleaned:
                text_content += cleaned + " "
        
        # Extract images from document
        try:
            from docx.oxml import parse_xml
            from docx.oxml.ns import qn
            
            # Get all image relationships
            for rel in doc.part.rels.values():
                if "image" in rel.target_ref:
                    try:
                        image_data = rel.target_part.blob
                        
                        # Perform OCR on the image
                        image_text = extract_text_from_image(image_data)
                        
                        if image_text and len(image_text.strip()) > 10:
                            image_texts.append(image_text)
                            logger.info(f"Extracted text from image in DOCX")
                        
                    except Exception as img_error:
                        logger.warning(f"Could not process image in DOCX: {str(img_error)}")
                        continue
        except Exception as e:
            logger.warning(f"Could not extract images from DOCX: {str(e)}")
        
        logger.info(f"Extracted text from {len(image_texts)} images in DOCX")
        
    except Exception as e:
        logger.error(f"Error extracting images from DOCX: {str(e)}")
    
    return text_content, image_texts

def extract_images_from_url(url: str) -> List[str]:
    """
    Download and OCR images from a web URL.
    
    Returns:
        List of extracted texts from images
    """
    image_texts = []
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return image_texts
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Find all image tags
        images = soup.find_all('img')
        
        for img_tag in images[:10]:  # Limit to first 10 images to avoid overload
            try:
                img_url = img_tag.get('src')
                
                if not img_url:
                    continue
                
                # Handle relative URLs
                if img_url.startswith('//'):
                    img_url = 'https:' + img_url
                elif img_url.startswith('/'):
                    from urllib.parse import urljoin
                    img_url = urljoin(url, img_url)
                elif not img_url.startswith('http'):
                    continue
                
                # Download image
                img_response = requests.get(img_url, timeout=5)
                if img_response.status_code == 200:
                    # Perform OCR
                    image_text = extract_text_from_image(img_response.content)
                    
                    if image_text and len(image_text.strip()) > 10:
                        image_texts.append(image_text)
                        logger.info(f"Extracted text from web image: {img_url}")
                
            except Exception as img_error:
                logger.warning(f"Could not process web image: {str(img_error)}")
                continue
        
        logger.info(f"Extracted text from {len(image_texts)} images from URL")
        
    except Exception as e:
        logger.error(f"Error extracting images from URL: {str(e)}")
    
    return image_texts

# ------------------ Database Functions ------------------

def get_db_connection():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("✅ Connected to CockroachDB successfully!")
        return conn
    except Exception as e:
        print("❌ Failed to connect to CockroachDB:", e)
        raise

def create_chat_session(document_source: str, chunks: List[str], user_id: int = None) -> str:
    """Create a new chat session and store chunks in database"""
    session_id = str(uuid.uuid4())
    connection = get_db_connection()
    if not connection:
        logger.error("Failed to connect to database")
        return None
    
    try:
        cursor = connection.cursor()
        
        cursor.execute(
            "INSERT INTO chat_sessions (session_id, user_id, document_source, chunk_count) VALUES (%s, %s, %s, %s)",
            (session_id, user_id, document_source, len(chunks))
        )
        
        chunk_data = [(session_id, user_id, i, chunk) for i, chunk in enumerate(chunks)]
        cursor.executemany(
            "INSERT INTO session_chunks (session_id, user_id, chunk_index, chunk_text) VALUES (%s, %s, %s, %s)",
            chunk_data
        )
        
        connection.commit()
        logger.info(f"Created chat session {session_id} with {len(chunks)} chunks for user {user_id}")
        return session_id
        
    except Error as e:
        logger.error(f"Error creating chat session: {e}")
        return None
    finally:
            cursor.close()
            connection.close()

def save_chat_message(session_id: str, message_type: str, content: str, user_id: int = None, retrieved_chunks: List[Dict] = None):
    """Save a chat message to database"""
    connection = get_db_connection()
    if not connection:
        return False
    
    try:
        cursor = connection.cursor()
        
        chunks_json = json.dumps(retrieved_chunks) if retrieved_chunks else None
        
        cursor.execute(
            "INSERT INTO chat_messages (session_id, user_id, message_type, content, retrieved_chunks) VALUES (%s, %s, %s, %s, %s)",
            (session_id, user_id, message_type, content, chunks_json)
        )
        
        connection.commit()
        logger.info(f"Saved {message_type} message to session {session_id}")
        return True
        
    except Error as e:
        logger.error(f"Error saving chat message: {e}")
        return False
    finally: 
     cursor.close()
     connection.close()


def get_chat_sessions(user_id: int = None) -> List[Dict]:
    """Get all chat sessions for a specific user ordered by latest first"""
    connection = get_db_connection()
    if not connection:
        return []
    
    try:
        cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        if user_id:
            cursor.execute(
                "SELECT session_id, document_source, chunk_count, created_at, updated_at "
                "FROM chat_sessions WHERE user_id = %s ORDER BY updated_at DESC",
                (user_id,)
            )
        else:
            cursor.execute(
                "SELECT session_id, document_source, chunk_count, created_at, updated_at "
                "FROM chat_sessions ORDER BY updated_at DESC"
            )
            
        sessions = cursor.fetchall()
        
        for session in sessions:
            session['created_at'] = session['created_at'].isoformat() if session['created_at'] else None
            session['updated_at'] = session['updated_at'].isoformat() if session['updated_at'] else None
            
        return sessions
        
    except Error as e:
        logger.error(f"Error fetching chat sessions: {e}")
        return []
    finally: 
      cursor.close()
      connection.close()

def get_chat_history(session_id: str, user_id: int = None) -> List[Dict]:
    """Get chat history for a specific session"""
    connection = get_db_connection()
    if not connection:
        return []
    
    try:
        cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        if user_id:
            cursor.execute(
                """
                SELECT message_type, content, retrieved_chunks, created_at
                FROM chat_messages 
                WHERE session_id = %s AND user_id = %s 
                ORDER BY created_at ASC
                """,
                (session_id, user_id)
            )
        else:
            cursor.execute(
                """
                SELECT message_type, content, retrieved_chunks, created_at
                FROM chat_messages 
                WHERE session_id = %s 
                ORDER BY created_at ASC
                """,
                (session_id,)
            )
            
        messages = cursor.fetchall()
        
        for message in messages:
            if message["retrieved_chunks"]:
                message["retrieved_chunks"] = json.loads(message["retrieved_chunks"])
            message["created_at"] = (
                message["created_at"].isoformat() if message["created_at"] else None
            )
            
        return messages
        
    except Exception as e:
        logger.error(f"Error fetching chat history: {e}")
        return []
    
    finally:
        try:
            cursor.close()
        except Exception:
            pass

        try:
            connection.close()
        except Exception:
            pass


def get_session_chunks(session_id: str, user_id: int = None) -> List[str]:
    """Get chunks for a specific session"""
    connection = get_db_connection()
    if not connection:
        return []
    
    try:
        cursor = connection.cursor()

        if user_id:
            cursor.execute(
                """
                SELECT chunk_text
                FROM session_chunks
                WHERE session_id = %s AND user_id = %s
                ORDER BY chunk_index ASC
                """,
                (session_id, user_id)
            )
        else:
            cursor.execute(
                """
                SELECT chunk_text
                FROM session_chunks
                WHERE session_id = %s
                ORDER BY chunk_index ASC
                """,
                (session_id,)
            )
            
        chunks = [row[0] for row in cursor.fetchall()]
        return chunks
        
    except Exception as e:
        logger.error(f"Error fetching session chunks: {e}")
        return []
    
    finally:
        try:
            cursor.close()
        except Exception:
            pass

        try:
            connection.close()
        except Exception:
            pass

def delete_chat_session(session_id: str, user_id: int = None) -> bool:
    """Delete a chat session and all related data"""
    connection = get_db_connection()
    if not connection:
        return False
    
    try:
        cursor = connection.cursor()

        if user_id:
            cursor.execute(
                "DELETE FROM chat_sessions WHERE session_id = %s AND user_id = %s",
                (session_id, user_id)
            )
        else:
            cursor.execute(
                "DELETE FROM chat_sessions WHERE session_id = %s",
                (session_id,)
            )
            
        connection.commit()
        logger.info(f"Deleted chat session {session_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error deleting chat session: {e}")
        return False
    
    finally:
        try:
            cursor.close()
        except Exception:
            pass

        try:
            connection.close()
        except Exception:
            pass

# ------------------- Phase 1 — Document Ingestion & Chunking with OCR ----------------------------

def process_source(source: str, session_id: str = None, user_id: int = None) -> Tuple[List[str], str]:
    """ 
    Complete source processing pipeline for file or URL WITH IMAGE OCR.
    Returns chunks and session_id.
    """
    print("🔄 Processing document/URL with image extraction...")
    chunks = extract_chunks(source)
    
    if not session_id:
        session_id = create_chat_session(source, chunks, user_id)
        if not session_id:
            logger.error("Failed to create chat session, using local storage only")
    
    save_chunks_to_file(chunks)
    print(f"✅ Extracted {len(chunks)} chunks (including text from images) and saved to chunks.pkl")
    if session_id:
        print(f"📝 Session ID: {session_id}")
    return chunks, session_id

def extract_chunks(source: str) -> List[str]:
    """ 
    Extract chunks from file or URL WITH IMAGE OCR.
    """
    if is_valid_url(source):
        return extract_chunks_from_url(source)
    else:
        if source.lower().endswith('.docx'):
            return extract_chunks_from_docx(source)
        elif source.lower().endswith(('.txt', '.pdf')):
            return extract_chunks_from_other_formats(source)
        else:
            raise ValueError(f"Unsupported file format: {source}")

def is_valid_url(url: str) -> bool:
    """ 
    Check if the given string is a valid URL.
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def extract_chunks_from_url(url: str) -> List[str]:
    """ 
    Extract and chunk text from URL INCLUDING images via OCR.
    """
    # Extract text content
    text_content = scrape_web_content(url)
    cleaned_text = clean_paragraph(text_content)
    text_content = cleaned_text if cleaned_text else ""
    
    # Extract text from images on the page
    image_texts = extract_images_from_url(url)
    
    # Combine all text
    combined_text = text_content
    if image_texts:
        combined_text += "\n\n[IMAGE CONTENT]\n" + "\n\n".join(image_texts)
        logger.info(f"Added {len(image_texts)} image texts to URL content")

    preprocessed_text = preprocess_text(combined_text)
    words = preprocessed_text.split()
    total_words = len(words)
    logger.info(f"Total words from URL (with images): {total_words}")

    if total_words <= 500:
        chunk_size = 100
    elif total_words <= 2000:
        chunk_size = 250
    else:
        chunk_size = 500
    logger.info(f"Dynamic chunk size set to {chunk_size} words")

    chunks = []
    for i in range(0, total_words, chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    logger.info(f"Created {len(chunks)} chunks from URL")
    return chunks

def scrape_web_content(url: str) -> str:
    """ 
    Scrape text content from a web page.
    """
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text(separator="\n", strip=True)
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split(" "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            logger.info(f"Successfully scraped {len(text)} characters from {url}")
            return text
        else:
            logger.error(f"Failed to fetch page, status code: {response.status_code}")
            raise Exception(f"Failed to fetch web page: HTTP {response.status_code}")
    except Exception as e:
        logger.error(f"Error scraping web content: {str(e)}")
        raise

def clean_paragraph(text: str) -> str:
    """ 
    Clean and normalize paragraph text.
    """
    if not text or not isinstance(text, str):
        return None
    cleaned = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned if cleaned else None

def preprocess_text(text: str) -> str:
    """ 
    Preprocess text by lowercasing, removing stopwords, and keeping only alphanumeric tokens.
    """
    if not text or not isinstance(text, str):
        return ""
    words = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    filtered_words = [
        word.lower() for word in words if word.lower() not in stop_words and word.isalnum()
    ]
    return " ".join(filtered_words)

def extract_chunks_from_docx(filename: str) -> List[str]:
    """ 
    Extract and chunk text from DOCX INCLUDING images via OCR.
    """
    validate_file(filename)
    
    # Extract text and images
    text_content, image_texts = extract_images_and_text_from_docx(filename)
    
    # Combine text and image content
    combined_text = text_content
    if image_texts:
        combined_text += "\n\n[IMAGE CONTENT]\n" + "\n\n".join(image_texts)
        logger.info(f"Added {len(image_texts)} image texts to DOCX content")
    
    preprocessed_text = preprocess_text(combined_text)
    words = preprocessed_text.split()
    total_words = len(words)
    logger.info(f"Total words in DOCX (with images): {total_words}")

    if total_words <= 500:
        chunk_size = 100
    elif total_words <= 2000:
        chunk_size = 250
    else:
        chunk_size = 500
    logger.info(f"Dynamic chunk size set to {chunk_size} words")

    chunks = []
    for i in range(0, total_words, chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    logger.info(f"Created {len(chunks)} chunks from DOCX")
    return chunks

def validate_file(filename: str) -> None:
    """ 
    Validate that the file exists and is a supported format.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    
    supported_formats = ('.docx', '.txt', '.pdf')
    if not filename.lower().endswith(supported_formats):
        raise ValueError(f"File must be one of {supported_formats}: {filename}")

def extract_chunks_from_other_formats(filename: str) -> List[str]:
    """
    Extract and chunk text from TXT or PDF INCLUDING images via OCR.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    
    file_ext = filename.lower().split('.')[-1]
    
    if file_ext == 'txt':
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            text_content = f.read()
        image_texts = []
    elif file_ext == 'pdf':
        # Extract text and images from PDF
        text_content, image_texts = extract_images_and_text_from_pdf(filename)
    else:
        raise ValueError(f"Unsupported file format: {filename}")
    
    # Combine text and image content
    if file_ext == 'txt':
        cleaned_text = clean_paragraph(text_content)
        combined_text = cleaned_text if cleaned_text else ""
    else:
        combined_text = text_content
        if image_texts:
            combined_text += "\n\n[IMAGE CONTENT]\n" + "\n\n".join(image_texts)
            logger.info(f"Added {len(image_texts)} image texts to PDF content")

    preprocessed_text = preprocess_text(combined_text)
    words = preprocessed_text.split()
    total_words = len(words)
    logger.info(f"Total words from {file_ext.upper()} (with images): {total_words}")
    
    if total_words <= 500:
        chunk_size = 100
    elif total_words <= 2000:
        chunk_size = 250
    else:
        chunk_size = 500
        
    logger.info(f"Dynamic chunk size set to {chunk_size} words")
    
    chunks = []
    for i in range(0, total_words, chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        
    logger.info(f"Created {len(chunks)} chunks from {file_ext.upper()}")
    return chunks

def save_chunks_to_file(chunks: List[str], output_file: str = "chunks.pkl") -> None:
    """ 
    Save extracted chunks to a pickle file.
    """
    with open(output_file, "wb") as f:
        pickle.dump(chunks, f)
    logger.info(f"Saved {len(chunks)} chunks to {output_file}")

#   -------------------------------- Phase 2 — Embedding & Indexing ----------------------------------

def initialize_qa_system() -> Tuple[SentenceTransformer, faiss.Index, List[str]]:
    """ 
    Initialize the QA system by loading chunks and creating embeddings.
    """
    print("Loading and embedding chunks...")
    chunks = load_chunks_from_file()
    model, index, embeddings = embed_and_store(chunks)
    print(f"Stored embeddings for {len(chunks)} chunks in FAISS")
    return model, index, chunks

def initialize_qa_system_from_session(session_id: str, user_id: int = None) -> Tuple[SentenceTransformer, faiss.Index, List[str]]:
    """Initialize QA system from a specific session"""
    print(f"Loading session {session_id}...")
    
    chunks = get_session_chunks(session_id, user_id)
    if not chunks:
        raise ValueError(f"No chunks found for session {session_id}")
    
    model, embeddings = create_embeddings(chunks)
    index = create_faiss_index(embeddings)
    
    print(f"Loaded {len(chunks)} chunks from session {session_id}")
    return model, index, chunks

def load_chunks_from_file(input_file: str = "chunks.pkl") -> List[str]:
    """ 
    Load chunks from a pickle file.
    """
    with open(input_file, "rb") as f:
        chunks = pickle.load(f)
    logger.info(f"Loaded {len(chunks)} chunks from {input_file}")
    return chunks

def embed_and_store(chunks: List[str]) -> Tuple[SentenceTransformer, faiss.Index, np.ndarray]:
    """ 
    Complete pipeline: embed chunks and store them in a FAISS index.
    """
    model, embeddings = create_embeddings(chunks)
    index = create_faiss_index(embeddings)
    return model, index, embeddings

def create_embeddings(chunks: List[str]) -> Tuple[SentenceTransformer, np.ndarray]:
    """ 
    Create embeddings for text chunks using SentenceTransformer.
    """
    try:
        model = SentenceTransformer("BAAI/bge-base-en-v1.5")
        embeddings = model.encode(
            chunks, convert_to_numpy=True, show_progress_bar=True, batch_size=32, normalize_embeddings=True
        ).astype("float32")
        logger.info(f"Successfully embedded {len(chunks)} chunks with dimension {embeddings.shape[1]}")
        return model, embeddings
    except Exception as e:
        logger.error(f"Error in create_embeddings: {str(e)}")
        raise

def create_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """ 
    Create a FAISS index from embeddings.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info(f"Created FAISS index with dimension {dim}")
    return index

#   ------------------------------ Phase 3 — Retrieval & Answer Generation ---------------------------

def answer_question(query: str, model: SentenceTransformer, index: faiss.Index, chunks: List[str], session_id: str = None, user_id: int = None) -> str:
    """ 
    Complete QA pipeline for a single question with session tracking.
    """
    retrieved = retrieve_relevant_chunks(query, model, index, chunks)
    
    print(f"\n🔍 Retrieved {len(retrieved)} relevant chunks:")
    for i, (chunk, score) in enumerate(retrieved, 1):
        print(f"{i}. [Score: {score:.3f}] {chunk[:100]}...")
    print("---")
    
    if session_id and user_id:
        save_chat_message(session_id, 'user', query, user_id)
    
    print("\n🤖 Generating answer...")
    answer = generate_answer(query, retrieved)
    enhanced_answer = enhance_answer_quality(answer, query)
    
    if session_id and user_id:
        retrieved_data = [{"text": chunk, "score": float(score)} for chunk, score in retrieved]
        save_chat_message(session_id, 'bot', enhanced_answer, user_id, retrieved_data)
    
    return enhanced_answer

def retrieve_relevant_chunks(query: str, model: SentenceTransformer, index: faiss.Index, chunks: List[str], top_k: int = DEFAULT_TOP_K) -> List[Tuple[str, float]]:
    """ 
    Retrieve the most relevant chunks for a query with their similarity scores.
    """
    try:
        preprocessed_query = preprocess_text(query)
        query_vec = model.encode([preprocessed_query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        D, I = index.search(query_vec, top_k)
        
        retrieved = []
        for idx, score in zip(I[0], D[0]):
            if score > 0.15:
                retrieved.append((chunks[idx], float(score)))
        
        retrieved.sort(key=lambda x: x[1], reverse=True)
        
        if not retrieved and I.size > 0:
            for i in range(min(3, len(I[0]))):
                retrieved.append((chunks[I[0][i]], float(D[0][i])))
        
        return retrieved[:top_k]
    except Exception as e:
        logger.error(f"Error in retrieve_relevant_chunks: {str(e)}")
        return []
    
def generate_answer(query: str, retrieved_chunks: List[Tuple[str, float]]) -> str:
    """ 
    Generate an answer using the Gemini API based on retrieved context.
    """
    prompt = construct_enhanced_prompt(query, retrieved_chunks)
    return call_gemini_api(prompt)

def construct_enhanced_prompt(query: str, retrieved_chunks: List[Tuple[str, float]]) -> str:
    """
    Construct a prompt for the LLM that allows mixed formatting with better structure.
    """
    if not retrieved_chunks:
        return f"""Answer the following question based on your general knowledge.

Question: {query}

FORMAT RULES:
- Use simple bullet points with • symbol
- Write in clear, concise paragraphs when needed
- Use numbered lists (1., 2., 3.) for steps or sequential information
- Use bullet points (•) for features, advantages, or non-sequential lists
- Bold important terms using **term**
- Keep formatting simple and clean
"""

    context_parts = []
    for i, (chunk, score) in enumerate(retrieved_chunks):
        clean_chunk = re.sub(r'\s+', ' ', chunk).strip()
        context_parts.append(f"[Source {i+1}, Relevance: {score:.3f}] {clean_chunk}")
    
    context = "\n\n".join(context_parts)
    
    # Detect if query requires list/point format
    list_indicators = [
        "list", "steps", "ways", "methods", "types", "advantages", 
        "disadvantages", "benefits", "features", "points", "factors",
        "arguments", "grounds", "reasons", "differences", "explain in points",
        "bullet points", "enumerate", "outline"
    ]
    
    requires_list_format = any(word in query.lower() for word in list_indicators)
    
    prompt = f"""You are an expert assistant. Use the retrieved chunks to answer the question accurately and in a well-formatted manner.

CRITICAL INSTRUCTIONS:
1. Merge all retrieved chunks smoothly and logically
2. Maintain factual accuracy from the source material
3. Text marked as [IMAGE CONTENT] is equally important
4. Use SIMPLE formatting - NO markdown headings (##, ###)
5. Use bullet points with • symbol for lists
6. Use numbered lists for sequential steps

FORMATTING REQUIREMENTS:
"""

    if requires_list_format:
        prompt += """
- Start with a brief introduction (1-2 sentences)
- Use numbered lists for sequential information:
  1. First point with explanation
  2. Second point with explanation
- Use bullet points (•) for non-sequential lists
- Each point should have 2-4 sentences of explanation
- Bold key terms using **term**
- Add a concluding sentence if appropriate

Example format:
Brief introduction explaining the context.

• **First Point**: Detailed explanation of the first point with relevant information from the chunks.

• **Second Point**: Detailed explanation with supporting details.

• **Third Point**: Clear explanation with context.

Concluding remarks if needed.
"""
    else:
        prompt += """
- Write in clear paragraphs (3-5 sentences each)
- Bold important terms using **term**
- Use proper spacing between paragraphs
- Maintain a natural, flowing narrative
- Use bullet points when listing multiple items

Example format:
First paragraph introducing the topic with key information from the retrieved chunks.

Second paragraph diving deeper into specific aspects with supporting details.

Additional paragraphs providing comprehensive coverage of the topic.
"""

    prompt += f"""

CONTEXT (with relevance scores):
{context}

QUESTION: {query}

Now provide a well-formatted answer following the above rules. Remember:
- NO markdown headings (##, ###)
- Use simple bullet points with • symbol
- Use numbered lists for steps
- Ensure proper spacing between elements
"""

    return prompt

def call_gemini_api(prompt: str) -> str:
    """ 
    Make API call to Gemini LLM.
    """
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [
            {
                "parts": [{"text": prompt}],
                "role": "user"
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "topK": 20,
            "topP": 0.8,
            "maxOutputTokens": 4096,
        },
        "safetySettings": [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]
    }
    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(data), timeout=60)
        response.raise_for_status()
        output = response.json()
        try:
            answer = output["candidates"][0]["content"]["parts"][0]["text"].strip()
            if not answer or len(answer) < 10:
                return "I couldn't generate a satisfactory answer based on the available information. Please try rephrasing your question."
            return answer
        except (KeyError, IndexError):
            logger.warning("Unexpected response format from Gemini API")
            return "I received an unexpected response format from the AI service. Please try again."
    except requests.exceptions.Timeout:
        logger.error("Request to Gemini API timed out")
        return "The AI service is taking too long to respond. Please try again later."
    except requests.exceptions.RequestException as e:
        logger.error(f"Request to Gemini API failed: {str(e)}")
        return f"Sorry, I encountered an error connecting to the AI service: {str(e)}"

def enhance_answer_quality(answer: str, query: str) -> str:
    """
    Post-process the generated answer to ensure clean, consistent formatting.
    """
    # Remove common LLM preambles
    patterns_to_remove = [
        r"Based on the provided context,?\s*",
        r"According to the context,?\s*",
        r"As mentioned in the sources,?\s*",
        r"The context (indicates|shows|states|says) that\s*",
        r"Based on my analysis,?\s*",
        r"According to the information provided,?\s*",
        r"Here is the answer:?\s*",
        r"Here's the answer:?\s*"
    ]
    
    for pattern in patterns_to_remove:
        answer = re.sub(pattern, "", answer, flags=re.IGNORECASE)
    
    # Remove markdown headings completely
    answer = re.sub(r'#{1,6}\s+', '', answer)
    
    # Fix common formatting issues
    # Convert various bullet styles to simple •
    answer = re.sub(r'^[\s]*[-*●]\s+', '• ', answer, flags=re.MULTILINE)
    
    # Ensure proper bullet formatting
    answer = re.sub(r'^•\s*', '• ', answer, flags=re.MULTILINE)
    
    # Fix numbered lists
    answer = re.sub(r'^\s*(\d+)\.\s*\*\*', r'\1. **', answer, flags=re.MULTILINE)
    
    # Normalize whitespace
    answer = re.sub(r' +', ' ', answer)  # Multiple spaces to single
    answer = re.sub(r'\n\s*\n\s*\n+', '\n\n', answer)  # Max 2 newlines
    
    # Ensure proper spacing after list items
    answer = re.sub(r'(^\d+\..*?[.!?])\n(?=\d+\.)', r'\1\n\n', answer, flags=re.MULTILINE)
    answer = re.sub(r'(^•.*?[.!?])\n(?=•)', r'\1\n\n', answer, flags=re.MULTILINE)
    
    # Fix spacing around bold text
    answer = re.sub(r'\*\*\s+', '**', answer)
    answer = re.sub(r'\s+\*\*', '**', answer)
    
    # Clean up any remaining issues
    answer = answer.strip()
    
    # Ensure answer doesn't start with formatting artifacts
    answer = re.sub(r'^[\s\*#\-●]+', '', answer)
    
    return answer

def display_formatted_answer(answer: str) -> None:
    """
    Display the answer with proper formatting in terminal.
    """
    lines = answer.split('\n')
    
    for line in lines:
        # Numbered lists
        if re.match(r'^\d+\.\s+', line):
            print(f"\n{line}")
        # Bullet points
        elif line.strip().startswith('•'):
            print(f"  {line.strip()}")
        # Regular text
        else:
            if line.strip():
                print(line)
            else:
                print()  # Preserve blank lines

# ----------------------------------------------------------------------------------------------------- 

def main():
    """Enhanced main function with session management"""
    try:
        import sys
        session_id = None
        
        # Check for session argument
        if len(sys.argv) > 1 and sys.argv[1] == "--session":
            if len(sys.argv) > 2:
                session_id = sys.argv[2]
                print(f"🔄 Resuming session: {session_id}")
                
                model, index, chunks = initialize_qa_system_from_session(session_id)
            else:
                sessions = get_chat_sessions()
                if sessions:
                    print("\n📚 Available chat sessions:")
                    for i, session in enumerate(sessions):
                        print(f"{i+1}. {session['document_source']} ({session['chunk_count']} chunks) - {session['updated_at']}")
                    
                    choice = input("\nEnter session number to resume or 'n' for new session: ")
                    if choice.isdigit() and 1 <= int(choice) <= len(sessions):
                        session_id = sessions[int(choice)-1]['session_id']
                        model, index, chunks = initialize_qa_system_from_session(session_id)
                    else:
                        session_id = None
                else:
                    print("No previous sessions found.")
                    session_id = None
        
        if not session_id:
            if not os.path.exists("chunks.pkl"):
                source = input("📄 Enter the path to your .docx/.txt/.pdf file or a URL: ").strip()
                if not source:
                    source = "sample.docx"
                chunks, session_id = process_source(source)
            else:
                chunks = load_chunks_from_file()
                session_id = create_chat_session("Existing chunks.pkl", chunks)
            
            model, index, chunks = initialize_qa_system()

        print(f"\n✅ Session ID: {session_id}")
        print("💬 Start chatting with the system! (type 'stop chat' to exit)")
        print("📋 Commands: 'list sessions' | 'switch session' | 'stop chat'\n")
        
        while True:
            query = input("❓ Enter your query: ").strip()
            
            if query.lower() in ['stop chat', 'exit', 'quit']:
                print("👋 Chat ended.")
                break
            elif query.lower() == 'list sessions':
                sessions = get_chat_sessions()
                if sessions:
                    print("\n📚 Previous chat sessions:")
                    for i, session in enumerate(sessions):
                        current_indicator = " (CURRENT)" if session['session_id'] == session_id else ""
                        print(f"{i+1}. {session['document_source']} - {session['updated_at']}{current_indicator}")
                else:
                    print("No previous sessions found.")
                continue
            elif query.lower() == 'switch session':
                sessions = get_chat_sessions()
                if sessions:
                    print("\n📚 Available sessions:")
                    for i, session in enumerate(sessions):
                        current_indicator = " (CURRENT)" if session['session_id'] == session_id else ""
                        print(f"{i+1}. {session['document_source']} - {session['updated_at']}{current_indicator}")
                    
                    choice = input("\nEnter session number to switch to: ")
                    if choice.isdigit() and 1 <= int(choice) <= len(sessions):
                        new_session_id = sessions[int(choice)-1]['session_id']
                        if new_session_id != session_id:
                            print(f"🔄 Switching to session {new_session_id}...")
                            os.execv(sys.executable, [sys.executable] + sys.argv + ['--session', new_session_id])
                    else:
                        print("❌ Invalid choice.")
                else:
                    print("No sessions available.")
                continue
                
            if not query:
                print("⚠️ Please enter a valid query.")
                continue

            answer = answer_question(query, model, index, chunks, session_id)

            print("\n✨ Final Answer:\n")
            display_formatted_answer(answer)
            print("-" * 60)

    except FileNotFoundError:
        logger.error("Document file or chunks.pkl not found")
        print("❌ Error: File not found. Please check the file path.")
    except Exception as e:
        logger.error(f"Unexpected error in main: {str(e)}")
        print(f"❌ An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()