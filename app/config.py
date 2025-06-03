# app/config.py
import os
import re # For more robust comment removal
from dotenv import load_dotenv

# Determine project root first
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env file located in the project root
dotenv_path = os.path.join(PROJECT_ROOT, '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
else:
    load_dotenv()


def clean_env_value(value: str) -> str:
    """Cleans a value read from .env: strips whitespace, quotes, and comments."""
    if value is None:
        return None
    
    # 1. Remove inline comments (anything after #)
    cleaned_value = re.split(r'\s*#', value, 1)[0]
    
    # 2. Strip leading/trailing whitespace
    cleaned_value = cleaned_value.strip()
    
    # 3. Remove surrounding quotes (single or double)
    if (cleaned_value.startswith('"') and cleaned_value.endswith('"')) or \
       (cleaned_value.startswith("'") and cleaned_value.endswith("'")):
        cleaned_value = cleaned_value[1:-1]
        
    return cleaned_value if cleaned_value else None # Return None if it becomes empty

# API Keys
GEMINI_API_KEY = clean_env_value(os.getenv("GEMINI_API_KEY")) # Clean API key too, just in case

# --- Path Configuration ---
DEFAULT_XLSX_FILENAME = "data/train_with_text.xlsx"
DEFAULT_CHROMA_DB_DIRNAME = "chroma_db_store_default"
DEFAULT_LOG_FILENAME = "logs/api_default.log"

# XLSX Path
xlsx_file_path_env = clean_env_value(os.getenv("XLSX_FILE_PATH"))
if xlsx_file_path_env and os.path.isabs(xlsx_file_path_env):
    XLSX_FILE_PATH = xlsx_file_path_env
else:
    XLSX_FILE_PATH = os.path.join(PROJECT_ROOT, xlsx_file_path_env or DEFAULT_XLSX_FILENAME)

# ChromaDB Path (This is the directory for ChromaDB)
chroma_db_path_env = clean_env_value(os.getenv("CHROMA_DB_PATH"))
if chroma_db_path_env and os.path.isabs(chroma_db_path_env):
    CHROMA_DB_PATH = chroma_db_path_env
elif chroma_db_path_env:
    CHROMA_DB_PATH = os.path.join(PROJECT_ROOT, chroma_db_path_env)
else:
    CHROMA_DB_PATH = os.path.join(PROJECT_ROOT, DEFAULT_CHROMA_DB_DIRNAME)

# Log File Path
log_file_path_env = clean_env_value(os.getenv("LOG_FILE_PATH"))
if log_file_path_env and os.path.isabs(log_file_path_env):
    LOG_FILE_PATH = log_file_path_env
else:
    LOG_FILE_PATH = os.path.join(PROJECT_ROOT, log_file_path_env or DEFAULT_LOG_FILENAME)


# --- Other Configurations ---
COLLECTION_NAME = clean_env_value(os.getenv("COLLECTION_NAME")) or "contracts_persistent_api_v1_default"
EMBEDDING_MODEL_NAME = clean_env_value(os.getenv("EMBEDDING_MODEL_NAME")) or "models/text-embedding-004"
GENERATIVE_MODEL_NAME = clean_env_value(os.getenv("GENERATIVE_MODEL_NAME")) or "gemini-1.5-flash-latest"
LOG_LEVEL = (clean_env_value(os.getenv("LOG_LEVEL")) or "INFO").upper()
TOP_K_RETRIEVAL = int(clean_env_value(os.getenv("TOP_K_RETRIEVAL")) or "3")
EMBEDDING_BATCH_SIZE = int(clean_env_value(os.getenv("EMBEDDING_BATCH_SIZE")) or "50")


def validate_config():
    """Validates critical configurations."""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set or empty after cleaning. Please check your .env file.")
    if not CHROMA_DB_PATH or CHROMA_DB_PATH == PROJECT_ROOT or CHROMA_DB_PATH == os.path.abspath("."):
        raise ValueError(f"CHROMA_DB_PATH ('{CHROMA_DB_PATH}') is invalid. It must be a specific directory, not the project root or current directory, and not empty.")
    if not os.path.exists(os.path.dirname(XLSX_FILE_PATH)):
        print(f"WARNING: Directory for XLSX_FILE_PATH '{os.path.dirname(XLSX_FILE_PATH)}' does not exist. File: '{XLSX_FILE_PATH}'")
    if TOP_K_RETRIEVAL <= 0:
        raise ValueError("TOP_K_RETRIEVAL must be a positive integer.")
    if EMBEDDING_BATCH_SIZE <=0 or EMBEDDING_BATCH_SIZE > 100:
        raise ValueError("EMBEDDING_BATCH_SIZE must be between 1 and 100.")

try:
    validate_config()
    print(f"--- Resolved Config Paths ---")
    print(f"Project Root:     {PROJECT_ROOT}")
    print(f"XLSX Path:        {XLSX_FILE_PATH}")
    print(f"ChromaDB Path:    {CHROMA_DB_PATH}")
    print(f"Log Path:         {LOG_FILE_PATH}")
    print(f"-----------------------------")
except ValueError as e:
    print(f"CONFIGURATION ERROR: {e}")
    # raise SystemExit(f"Configuration Error: {e}")