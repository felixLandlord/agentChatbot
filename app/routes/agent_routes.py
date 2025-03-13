from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse, StreamingResponse
import logging
from app.services.agent_service import process_question_stream, process_question
from pydantic import BaseModel
from typing import Optional
import uuid
from app.services.helper_services import get_dynamic_greeting

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


agent_router = APIRouter()

    
class QuestionRequest(BaseModel):
    question: str
    thread_id: Optional[str] = None


@agent_router.get("/init")
async def initiate_chat():
    """
    Initialize a new chat session and return a thread ID
    """
    try:
        thread_id = str(uuid.uuid4())
        logger.info(f"🚀 Initiating new chat session with thread ID: {thread_id}")
        greeting = get_dynamic_greeting()
        logger.info(f"💬 Sending greeting: {greeting}")
        
        return JSONResponse(
            status_code=200,
            content={"thread_id": thread_id, "greeting": greeting}
        )
    except Exception as e:
        logger.error(f"❌ Error initiating chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error initiating chat: {str(e)}")
    

@agent_router.post("/chat")
async def chat_with_agent(question_request: QuestionRequest, request: Request):
    """
    Send a question to the AI Aku agent and get a response.
    
    Args:
        request: Question request containing the user's question
    
    Returns:
        JSON response with the agent's answer
    """
    try:
        if not question_request.thread_id:
            raise HTTPException(status_code=400, detail="thread_id is required")
            
        logger.info(f"🔍 Processing question for thread {question_request.thread_id}")
        
        logger.info(f"❓ Received question: {question_request.question}")
        
        headers = request.headers
        accept = headers.get("accept", "")
        
        if "application/json" in accept:
            # Regular JSON response for Swagger UI
            logger.info(f"📡 Sending JSON response for thread {question_request.thread_id}")
            result = await process_question(question_request.question, question_request.thread_id)
            return JSONResponse(content=result)
        else:
            # Streaming response for SSE
            logger.info(f"📡 Sending streaming response for thread {question_request.thread_id}")
            return StreamingResponse(
                (chunk async for chunk in process_question_stream(question_request.question, question_request.thread_id)),
                media_type='text/event-stream'
            )
        
    except Exception as e:
        logger.error(f"❌ Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")