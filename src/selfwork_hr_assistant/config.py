# config.py
import os
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    OPENAI_KEY = os.getenv("OPENAI_KEY")
    DOCUMENTS_DIR = "resumes"
    COLLECTION_NAME = "CVs"
    PERSISTENT_DIR = "data/chromadb"
    MODEL_NAME = "text-embedding-3-small"
    LLM_MODEL = "llama3.2"
    AI_API_URL = "http://localhost:11434/v1"
    AI_API_KEY = os.getenv("OPENAI_KEY")
   
    LLM_MODEL = "gpt-4o-mini"
    AI_API_URL = "https://api.openai.com/v1/"
