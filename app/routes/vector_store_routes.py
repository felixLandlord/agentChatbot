from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import tempfile
import os
import logging
from app.services.vector_store_service import create_embeddings_from_json


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


vector_store_router = APIRouter()


@vector_store_router.post("/upload")
async def upload_json_and_create_embeddings(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload a JSON file and create embeddings in the background.
    
    Args:
        background_tasks: FastAPI background tasks
        file: Uploaded JSON file
    
    Returns:
        JSON response with status
    """
    if not file.filename.endswith('.json'):
        raise HTTPException(status_code=400, detail="Only JSON files are allowed")
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as temp_file:
            temp_file_path = temp_file.name
            # Write uploaded file content to temp file
            content = await file.read()
            temp_file.write(content)
        
        try:
            logger.info(f"📂 Processing uploaded file: {file.filename}")
            
            vector_store = await create_embeddings_from_json(temp_file_path)
            
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
                logger.info(f"🗑️ Temporary file {temp_file_path} removed.")
            except Exception as e:
                logger.error(f"❌ Error removing temporary file {temp_file_path}: {str(e)}")
                
            if vector_store:
                logger.info(f"✅ Successfully created embeddings from {file.filename}")
                return JSONResponse(
                    status_code=200,
                    content={
                        "message": f"File '{file.filename}' processed successfully. Embeddings created."
                    }
                )
            else:
                logger.error(f"❌ Failed to create embeddings from {file.filename}")
                raise HTTPException(status_code=500, detail="Failed to create embeddings")
                
        except Exception as e:
            logger.error(f"❌ Error processing {file.filename}: {str(e)}")
            # Clean up temporary file in case of error
            try:
                os.unlink(temp_file_path)
                logger.info(f"🗑️ Temporary file {temp_file_path} removed.")
            except:
                pass
            raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    
    except Exception as e:
        logger.error(f"❌ Error processing upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing upload: {str(e)}")