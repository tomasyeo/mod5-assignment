import os
from dotenv import load_dotenv

load_dotenv()

# --- Application Settings ---
MEMORY_LIMIT = int(os.getenv("MEMORY_LIMIT", "5"))
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Logging Settings ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# --- Model Settings ---
# Switched back to 8B-instant for reliable tool support and high rate limits
GROQ_MODEL = "llama-3.1-8b-instant"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")
APPLE_DATA_DIR = os.path.join(DATA_DIR, "apple_data")
STATIC_DIR = os.path.join(BASE_DIR, "static")
MODEL_PATH = os.path.join(DATA_DIR, "salary_model.joblib")
