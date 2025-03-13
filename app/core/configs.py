from pydantic_settings import BaseSettings
from typing import Optional

class Configurations(BaseSettings):
    
    # Model Configurations
    embedding_model: str = "models/text-embedding-004"
    
    genai2_flash_model: str = "models/gemini-2.0-flash"
    genai2_flash_lite_model: str = "models/gemini-2.0-flash-lite"
    groq_llama3_3_70b_model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.2
    max_tokens: Optional[int] = None
    # max_retries: int = 3
    GOOGLE_API_KEY: str
    GROQ_API_KEY: str
    
    # Langfuse Tracking
    # LANGFUSE_PUBLIC_KEY: str
    # LANGFUSE_SECRET_KEY: str
    # LANGFUSE_HOST: str
    
    # Database Connection
    database_name: str = "myDatabase"
    vector_store_collection: str = "mySource"
    history_collection: str = "myHistory"
    vector_index_name: str = "vector_index"
    fulltext_index_name: str = "search_index"
    top_k: int = 5
    fulltext_penalty: int = 50
    vector_penalty: int = 50
    MONGO_USERNAME: str
    MONGO_PASSWORD: str
    MONGO_HOST: str

    
    class Config:
        env_file = ".env"
        case_sensitive = True


configs = Configurations()