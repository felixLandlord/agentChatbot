from langchain_community.document_loaders import JSONLoader
import json
import logging
from pathlib import Path
from typing import List, Optional
from langchain.docstore.document import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from app.core.configs import configs
# from pymongo import MongoClient
from pymongo import AsyncMongoClient


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MONGO_URI = f"mongodb+srv://{configs.MONGO_USERNAME}:{configs.MONGO_PASSWORD}@{configs.MONGO_HOST}/{configs.database_name}"

# mongo_client = MongoClient(MONGO_URI)
mongo_client = AsyncMongoClient(MONGO_URI)

db = mongo_client[configs.database_name]
source_collection = db[configs.vector_store_collection]
history_collection = db[configs.history_collection]

embedding_model = GoogleGenerativeAIEmbeddings(
            model=configs.embedding_model,
            api_key=configs.GOOGLE_API_KEY
        )


async def load_json(file_path: str) -> Optional[List[Document]]:
    """
    Load and parse a JSON file using LangChain's JSONLoader.
    
    Args:
        file_path: Path to the JSON file
    
    Returns:
        List of Document objects containing the parsed content or None if error occurs
    """
    try:
        # Ensure file path is absolute
        abs_path = Path(file_path).resolve()
        
        # Check if file exists
        if not abs_path.exists():
            logger.error(f"❌ File not found: {abs_path}")
            return None
            
        # Check if file is JSON
        if abs_path.suffix.lower() != '.json':
            logger.error(f"❌ File is not a JSON file: {abs_path}")
            return None
            
        logger.info(f"📂 Loading JSON file: {abs_path}")
        loader = JSONLoader(
            file_path=str(abs_path),
            jq_schema='.[]',  # Load all items from root array
            text_content=False
        )
        
        docs = await loader.aload()
        logger.info(f"✅ Successfully loaded {len(docs)} documents")
        return docs
        
    except json.JSONDecodeError as e:
        logger.error(f"❌ Invalid JSON format: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"❌ Error loading JSON file: {str(e)}")
        return None
   
    
async def create_embeddings_from_json(file_path: str) -> Optional[MongoDBAtlasVectorSearch]:
    """
    Load JSON data and create embeddings for MongoDB Vector Search.
    
    Args:
        file_path: Path to the JSON file
    
    Returns:
        MongoDBAtlasVectorSearch instance or None if error occurs
    """
    try:
        docs = await load_json(file_path)
        if not docs:
            return None

        # Process each document
        processed_docs = []
        for doc in docs:
            data = json.loads(doc.page_content)
            
            # Create combined question and answer as main content
            qa_content = f"Question: {data['question']}\nAnswer: {data['answer']}"
            
            # Create document with the correct field structure
            processed_docs.append(
                Document(
                    page_content=qa_content,  # Store question and answer as the main content
                    metadata={
                        "question": data['question'],  # Store question separately in metadata
                        "answer": data['answer'],  # Store answer separately in metadata
                        "category": data['category'],
                        "tags": data['tags']
                    }
                )
            )

        logger.info(f"🧠 Creating vector store with {len(processed_docs)} documents")
        
        vector_store = await MongoDBAtlasVectorSearch.afrom_documents(
            documents=processed_docs,
            embedding=embedding_model,
            collection=source_collection,
            index_name=configs.vector_index_name,
            embedding_key = "embedding"
        )
        
        logger.info("✅ Successfully created vector store")
        return vector_store

    except Exception as e:
        logger.error(f"❌ Error creating embeddings: {str(e)}")
        return None


async def vector_store_initializer():
    return MongoDBAtlasVectorSearch(
        collection=source_collection,
        embedding=embedding_model,
        index_name=configs.vector_index_name
    )