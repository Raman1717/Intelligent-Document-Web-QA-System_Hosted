from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os, uuid, logging
import embedding 
import psycopg2
from psycopg2.extras import RealDictCursor

# =============================
# CONFIGURATION
# =============================
app = Flask(__name__, static_folder="static")
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_CONFIG = {
    "host": "ninja-saiga-18700.j77.aws-ap-south-1.cockroachlabs.cloud",
    "user": "raman",
    "password": "meN4ZNAwDyxLKDAy086omA",
    "database": "rag_chat_user",
    "port":26257
}

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {"txt", "pdf", "docx"}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size


model = None
index = None
chunks = None
current_session_id = None
is_initialized = False

# =============================
# HELPERS
# =============================
def get_db_connection():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("✅ Connected to CockroachDB successfully!")
        return conn
    except Exception as e:
        print("❌ Failed to connect to CockroachDB:", e)
        raise

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def verify_user_token(token):
    """Check user auth token"""
    if not token:
        return None
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute("SELECT * FROM users WHERE auth_token=%s", (token,))
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    return user

# =============================
# FRONTEND ROUTES
# =============================
@app.route("/")
def serve_index():
    return send_from_directory("static", "index.html")

@app.route("/<path:path>")
def serve_static_files(path):
    return send_from_directory("static", path)

# =============================
# AUTHENTICATION
# =============================
@app.route("/api/signup", methods=["POST"])
def signup():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"status": "error", "message": "Missing username or password"}), 400

    conn = get_db_connection()   
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute("SELECT * FROM users WHERE username=%s", (username,))
    if cursor.fetchone():
        cursor.close(); conn.close()
        return jsonify({"status": "error", "message": "Username already exists"}), 400

    password_hash = generate_password_hash(password)
    token = str(uuid.uuid4())

    cursor.execute("INSERT INTO users (username, password_hash, auth_token) VALUES (%s, %s, %s)", 
                   (username, password_hash, token))
    conn.commit()
    cursor.close(); conn.close()

    return jsonify({"status": "success", "user": {"username": username, "token": token}}), 200


@app.route("/api/signin", methods=["POST"])
def signin():
    data = request.get_json()
    username, password = data.get("username"), data.get("password")
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute("SELECT * FROM users WHERE username=%s", (username,))
    user = cursor.fetchone()

    if not user or not check_password_hash(user["password_hash"], password):
        return jsonify({"status": "error", "message": "Invalid username or password"}), 401

    token = str(uuid.uuid4())
    cursor.execute("UPDATE users SET auth_token=%s WHERE user_id=%s", (token, user["user_id"]))
    conn.commit(); cursor.close(); conn.close()
    return jsonify({"status": "success", "user": {"username": username, "token": token}}), 200


@app.route("/api/logout", methods=["POST"])
def logout():
    data = request.get_json()
    token = data.get("auth_token")
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET auth_token=NULL WHERE auth_token=%s", (token,))
    conn.commit()
    cursor.close(); conn.close()
    return jsonify({"status": "success", "message": "Logged out"}), 200


@app.route("/api/verify-token", methods=["POST"])
def verify_token():
    data = request.get_json()
    token = data.get("auth_token")
    user = verify_user_token(token)
    if not user:
        return jsonify({"status": "error", "message": "Invalid token"}), 401
    return jsonify({"status": "success", "username": user["username"]}), 200

# =============================
# SYSTEM STATUS & MANAGEMENT
# =============================
@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current system status"""
    try:
        token = request.headers.get("Authorization")
        user = verify_user_token(token)
        if not user:
            return jsonify({"status": "error", "message": "Unauthorized"}), 401

        chunk_count = len(chunks) if chunks else 0
        return jsonify({
            'status': 'success',
            'initialized': is_initialized,
            'chunk_count': chunk_count,
            'session_id': current_session_id,
            'chunks_file_exists': os.path.exists('chunks.pkl')
        })
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route("/api/reset", methods=["POST"])
def reset_system():
    """Reset the system and clear all data"""
    global model, index, chunks, current_session_id, is_initialized
    
    token = request.headers.get("Authorization")
    user = verify_user_token(token)
    if not user:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    try:
        model = None
        index = None
        chunks = None
        current_session_id = None
        is_initialized = False
        
        # Delete chunks.pkl if exists
        if os.path.exists('chunks.pkl'):
            os.remove('chunks.pkl')
        
        # Clear uploads folder
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        logger.info("System reset successfully")
        
        return jsonify({
            'status': 'success',
            'message': 'System reset successfully!'
        })
        
    except Exception as e:
        logger.error(f"Error resetting system: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# =============================
# RAG DOCUMENT PROCESSING
# =============================
@app.route("/api/process-file", methods=["POST"])
def process_file():
    global is_initialized, current_session_id, model, index, chunks
    
    token = request.headers.get("Authorization")
    user = verify_user_token(token)
    if not user:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"status": "error", "message": "Invalid file type"}), 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_{filename}")
        file.save(filepath)

        logger.info(f"Processing file: {filepath}")
        
        # Pass user_id to process_source
        chunks, session_id = embedding.process_source(filepath, user_id=user["user_id"])
        current_session_id = session_id
        
        # Initialize the QA system immediately after processing
        try:
            model, index, chunks = embedding.initialize_qa_system()
            is_initialized = True
            logger.info(f"QA system initialized with {len(chunks)} chunks")
        except Exception as init_error:
            logger.error(f"Failed to initialize QA system: {str(init_error)}")
            is_initialized = False

        return jsonify({
            "status": "success",
            "message": f"File processed into {len(chunks)} chunks.",
            "chunk_count": len(chunks),
            "filename": filename,
            "session_id": session_id,
            "initialized": is_initialized
        })
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route("/api/process-url", methods=["POST"])
def process_url():
    global is_initialized, current_session_id, model, index, chunks
    
    token = request.headers.get("Authorization")
    user = verify_user_token(token)
    if not user:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    data = request.get_json()
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"status": "error", "message": "No URL provided"}), 400

    try:
        logger.info(f"Processing URL: {url}")
        
        # Pass user_id to process_source
        chunks, session_id = embedding.process_source(url, user_id=user["user_id"])
        current_session_id = session_id
        
        # Initialize the QA system immediately after processing
        try:
            model, index, chunks = embedding.initialize_qa_system()
            is_initialized = True
            logger.info(f"QA system initialized with {len(chunks)} chunks")
        except Exception as init_error:
            logger.error(f"Failed to initialize QA system: {str(init_error)}")
            is_initialized = False
        
        return jsonify({
            "status": "success",
            "message": f"URL processed into {len(chunks)} chunks.",
            "chunk_count": len(chunks),
            "url": url,
            "session_id": session_id,
            "initialized": is_initialized
        })
    except Exception as e:
        logger.error(f"Error processing URL: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route("/api/initialize", methods=["POST"])
def initialize_system():
    """Initialize the QA system"""
    global model, index, chunks, is_initialized
    
    token = request.headers.get("Authorization")
    user = verify_user_token(token)
    if not user:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    try:
        # Check if chunks.pkl exists
        if not os.path.exists('chunks.pkl'):
            return jsonify({
                'status': 'error',
                'message': 'Please process a document first!'
            }), 400
        
        logger.info("Initializing QA system...")
        
        model, index, chunks = embedding.initialize_qa_system()
        is_initialized = True
        
        logger.info(f"System initialized with {len(chunks)} chunks")
        
        return jsonify({
            "status": "success", 
            "message": "System initialized", 
            "chunk_count": len(chunks),
            "message": f'System ready with {len(chunks)} chunks!'
        })
    except Exception as e:
        logger.error(f"Error initializing system: {str(e)}")
        is_initialized = False
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/query", methods=["POST"])
def query():
    """Answer a question"""
    global model, index, chunks, current_session_id, is_initialized
    
    token = request.headers.get("Authorization")
    user = verify_user_token(token)
    if not user:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    data = request.get_json()
    question = data.get("question", "").strip()
    session_id = data.get("session_id", current_session_id)

    if not question:
        return jsonify({"status": "error", "message": "Empty question"}), 400
    
    # Check if system is initialized
    if not is_initialized or model is None or index is None or chunks is None:
        return jsonify({
            'status': 'error',
            'message': 'System not initialized. Please process a document first!'
        }), 400

    try:
        logger.info(f"Processing query: {question}")
        
        # Pass user_id to answer_question
        answer = embedding.answer_question(question, model, index, chunks, 
                                         session_id=session_id, user_id=user["user_id"])
        
        logger.info("Answer generated successfully")
        
        return jsonify({
            "status": "success",
            "answer": answer,
            "session_id": session_id
        })
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

# =============================
# SESSION MANAGEMENT
# =============================
@app.route("/api/sessions", methods=["GET"])
def get_sessions():
    """Return list of all user sessions."""
    logger.info("GET /api/sessions called")

    token = request.headers.get("Authorization")
    logger.info(f"Authorization token received: {token}")

    user = verify_user_token(token)
    if not user:
        logger.warning("Unauthorized access attempt: invalid token")
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    logger.info(f"Authenticated user: user_id={user['user_id']}, username={user.get('username')}")

    try:
        logger.info(f"Fetching sessions for user_id={user['user_id']}")
        sessions = embedding.get_chat_sessions(user_id=user["user_id"])
        
        logger.info(f"Found {len(sessions)} session(s) for user_id={user['user_id']}")

        return jsonify({
            "status": "success",
            "sessions": sessions,
            "current_session": current_session_id
        })

    except Exception as e:
        logger.error("Error getting sessions", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route("/api/sessions/<session_id>", methods=["GET"])
def get_session_history(session_id):
    """Return chat history for a specific session."""
    token = request.headers.get("Authorization")
    user = verify_user_token(token)
    if not user:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    try:
        # Pass user_id to get_chat_history
        history = embedding.get_chat_history(session_id, user_id=user["user_id"])
        return jsonify({
            "status": "success", 
            "session_id": session_id, 
            "history": history
        })
    except Exception as e:
        logger.error(f"Error getting session history: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/sessions/<session_id>/load', methods=['POST'])
def load_session(session_id):
    """Load a specific chat session"""
    global model, index, chunks, current_session_id, is_initialized
    
    token = request.headers.get("Authorization")
    user = verify_user_token(token)
    if not user:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    try:
        logger.info(f"Loading session: {session_id}")
        
        # Load session from database with user_id
        model, index, chunks = embedding.initialize_qa_system_from_session(session_id, user_id=user["user_id"])
        current_session_id = session_id
        is_initialized = True
        
        # Get chat history with user_id
        history = embedding.get_chat_history(session_id, user_id=user["user_id"])
        
        logger.info(f"Session {session_id} loaded with {len(chunks)} chunks and {len(history)} messages")
        
        return jsonify({
            'status': 'success',
            'message': f'Session loaded!',
            'chunk_count': len(chunks),
            'session_id': session_id,
            'history': history,
            'initialized': True
        })
        
    except Exception as e:
        logger.error(f"Error loading session: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/sessions/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """Delete a chat session"""
    global current_session_id, is_initialized, model, index, chunks
    
    token = request.headers.get("Authorization")
    user = verify_user_token(token)
    if not user:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    try:
        # Pass user_id to delete_chat_session
        success = embedding.delete_chat_session(session_id, user_id=user["user_id"])
        if success:
            # If current session is deleted, reset system
            if session_id == current_session_id:
                model = None
                index = None
                chunks = None
                current_session_id = None
                is_initialized = False
            return jsonify({'status': 'success', 'message': 'Session deleted successfully'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to delete session'}), 500
    except Exception as e:
        logger.error(f"Error deleting session: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# =============================
# RUN SERVER
# =============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render provides PORT automatically
    print(f"Server starting at: http://0.0.0.0:{port}")
    app.run(debug=False, host="0.0.0.0", port=port)