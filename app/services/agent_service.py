# from langchain_mongodb.retrievers.hybrid_search import MongoDBAtlasHybridSearchRetriever
from app.services.vector_store_service import vector_store_initializer, source_collection, history_collection, embedding_model
from app.core.configs import configs
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser   
from typing_extensions import TypedDict
from typing import List, Optional, Dict, Any
from langgraph.graph import END, StateGraph
import logging
import uuid
from datetime import datetime, timedelta
# from pydantic import BaseModel
import asyncio
from app.core.custom_hybrid_retriever import CustomMongoDBAtlasHybridSearchRetriever
import json


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


vector_store = None
async def initialize_vector_store():
    vector_store = await vector_store_initializer()
    return vector_store


genai_flash_llm = ChatGoogleGenerativeAI(
    model=configs.genai2_flash_model,
    temperature=configs.temperature,
    max_tokens=configs.max_tokens,
    api_key=configs.GOOGLE_API_KEY,
    streaming=True
)

genai_flash_lite_llm = ChatGoogleGenerativeAI(
    model=configs.genai2_flash_lite_model,
    temperature=configs.temperature,
    max_tokens=configs.max_tokens,
    api_key=configs.GOOGLE_API_KEY,
    streaming=True
)

groq_llama_llm = ChatGroq(
    model=configs.groq_llama3_3_70b_model, 
    temperature=configs.temperature, 
    api_key=configs.GROQ_API_KEY,
    streaming=True
)


@tool("hybrid_search")
async def get_hybrid_search_results(query: str) -> List[str]:
    """
    Perform a hybrid search using MongoDB Atlas and return relevant documents.
    
    Args:
        query: The search query string
        top_k: Number of results to return (default: 5)
        
    Returns:
        List of documents matching the query
    """    
    if not query:
        logger.error("❌ Hybrid search received an empty query.")
        return []
    
    try:
        vector_store = await initialize_vector_store()
    

        retriever = CustomMongoDBAtlasHybridSearchRetriever(
            vectorstore=vector_store,
            search_index_name=configs.fulltext_index_name,
            top_k=configs.top_k,
            fulltext_penalty=configs.fulltext_penalty,
            vector_penalty=configs.vector_penalty
        )
        
        documents = await retriever.ainvoke(query)
        
        
        if not documents:
            return []
            
        # Extract only the page_content from each document
        page_contents = [doc.page_content for doc in documents if hasattr(doc, 'page_content')]
        
        return page_contents
    
    except Exception as e:
        logger.error(f"❌ Error in hybrid search: {str(e)}")
        import traceback
        logger.error(f"🛠️ Traceback: {traceback.format_exc()}")
        return []

tools = [get_hybrid_search_results]
tool_node = ToolNode(tools)


# Defining graph states
class AgentState(TypedDict):
    question: str
    thread_id: Optional[str]
    reformulated_question: Optional[str]
    needs_retrieval: Optional[bool]
    retrieved_documents: Optional[List[str]]
    useful_documents: Optional[List[str]]
    response: Optional[str]


# Step 1: Reformulate the question for better search
async def reformulate_question(state: AgentState) -> AgentState:
    """Reformulate the question to enhance search quality"""
    logger.info("🔄 Reformulating question for better search")
    
    history_text = ""
    if state.get("thread_id"):
        history = await get_conversation_history(state["thread_id"])
        history_text = await format_conversation_history(history)
    
    prompt = PromptTemplate(
        template="""You are an expert at reformulating questions for better search results.
        Given a chat history and the latest user's question, formulate a standalone question.
        Which can be understood without the chat history. Do NOT answer the question.
        Just reformulate it if needed and otherwise return it as is.
        Keep the reformulation concise and focused on the key information needs.
        
        User conversation history: {history}
        
        Original question: {question}
        
        Reformulated question:""",
        input_variables=["history", "question"]
    )
    
    reformulated_question = await genai_flash_lite_llm.ainvoke(prompt.format(history=history_text, question=state["question"]))
    logger.info(f"🔄 Reformulated question: {reformulated_question.content}")
    
    return {
        **state,
        "reformulated_question": reformulated_question.content
    }


# Step 2: Decide if retrieval is needed
async def decide_retrieval(state: AgentState) -> str:
    """Decide whether to use retrieval or directly answer"""
    logger.info("🔍 Deciding if retrieval is needed")
    
    prompt = PromptTemplate(
        template="""Determine if the following question requires looking up information from knowledge base.
        Answer with 'yes' for ANY question that requires retrieval from a knowledge base.

        Answer with 'no' ONLY if the question is:
        1. A simple greeting (exactly "hi", "hello", "good morning")
        2. A direct question about your capabilities as an AI (exactly "who are you", "what can you do")

        For ALL other questions, including ambiguous ones, answer with 'yes'.     
        
        Question: {question}
        
        Does this question require retrieval from a knowledge base? (yes/no)""",
        input_variables=["question"]
    )
    
    decision = await groq_llama_llm.ainvoke(prompt.format(question=state["reformulated_question"]))
    logger.info(f"🔍 Retrieval decision: {decision.content}")
    needs_retrieval = "yes" in decision.content.lower()
    
    state["needs_retrieval"] = needs_retrieval
    
    if needs_retrieval:
        logger.info("📚 Retrieval needed")
        return "retrieve"
    else:
        logger.info("🚫 No retrieval needed")
        return "direct_answer"


# Step 3: Retrieve documents
async def retrieve_documents(state: AgentState) -> AgentState:
    """Retrieve documents using the hybrid search tool"""
    logger.info("📚 Retrieving documents")
    
    query = str(state["reformulated_question"]) if state.get("reformulated_question") else ""
    logger.info(f"🔍 Query for retrieval: {query}")
    
    try:
        retrieved_docs = await get_hybrid_search_results.ainvoke(query)
        logger.info(f"📚 Retrieved documents: {retrieved_docs}")
        
        return {
            **state,
            "retrieved_documents": retrieved_docs
        }
    except Exception as e:
        logger.error(f"❌ Error in document retrieval: {str(e)}")
        return {
            **state,
            "retrieved_documents": []
        }
    
# Step 4: Filter useful documents
async def filter_documents(state: AgentState) -> AgentState:
    """Filter retrieved documents to keep only the useful ones"""
    logger.info("🔍 Filtering useful documents")
    
    if not state.get("retrieved_documents"):
        return {**state, "useful_documents": []}
    
    prompt = PromptTemplate(
        template="""Determine if each document is relevant to answering the question.
        
        Question: {question}
        
        Documents:
        {documents}
        
        For each document, indicate if it's useful (yes/no) in a JSON format with the structure:
        {{
            "useful_indices": [list of indices of useful documents, starting from 0]
        }}""",
        input_variables=["question", "documents"]
    )
    
    documents_text = "\n\n".join([f"Document {i}: {doc}" for i, doc in enumerate(state["retrieved_documents"])])
    
    result = await genai_flash_llm.ainvoke(prompt.format(
        question=state["question"],
        documents=documents_text
    ))
    logger.info("🔍 Filtering result: %s", result.content)
    
    # Extract useful document indices
    try:
        # Try to parse as JSON
        import json
        import re
        
        # Find JSON-like content in the response
        json_match = re.search(r'\{.*\}', result.content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            useful_indices = json.loads(json_str).get("useful_indices", [])
            
            useful_docs = [state["retrieved_documents"][i] for i in useful_indices if i < len(state["retrieved_documents"])]
            logger.info("📄 Useful docs: %s", useful_docs)
        else:
            # Fallback: use all documents
            useful_docs = state["retrieved_documents"]
            logger.info("🚫 No JSON found in response, using all documents")
    except:
        # If parsing fails, use all documents
        useful_docs = state["retrieved_documents"]
        logger.info("🚫 Failed to parse JSON, using all documents")
    
    return {
        **state,
        "useful_documents": useful_docs
    }


# Step 5: Generate direct answer (no retrieval)
async def generate_direct_answer(state: AgentState) -> AgentState:
    """Generate a response without using retrieval"""
    logger.info("💬 Generating direct answer")
    
    # Get conversation history if thread_id is available
    history_text = ""
    if state.get("thread_id"):
        history = await get_conversation_history(state["thread_id"])
        history_text = await format_conversation_history(history)
    
    prompt = PromptTemplate(
        template="""You are called felixLandlord, an expert assistant dedicated to helping users with their questions.
        Your tone should be friendly, approachable, and professional, ensuring users feel supported and valued.
        Greet users naturally and warmly, without mentioning that you're an AI model.
        When handling sensitive or potentially negative information, express empathy and offer solutions where possible.
        If a question is unclear, kindly ask for clarification before responding.
        Transition smoothly between topics, especially when dealing with follow-up questions.
        Keep your responses short, relevant, and straight to the point, avoiding any ambiguity or unnecessary details as well as unnecessarily long responses.
        Limit your responses to at most 2 sentences, ensuring concise outputs.
        Acknowledge user feedback, whether positive or negative, and respond appropriately.
        Always end by politely asking if there's anything else they need help with or suggesting a related topic if contextually appropriate.
        Respond to the user's query as polite as you can, no matter the number of times the user asks the same question without mentioning to the user that you have already answered the question.
        For greetings and simple questions about your capabilities, respond in a friendly manner.
        Do not make up information - only respond with general pleasantries for questions that don't require specific the knowledge.
        
        User conversation history: {history}
                
        User question: {question}
        
        Your response:""",
        input_variables=["history", "question"]
    )
    
    response = await genai_flash_lite_llm.ainvoke(prompt.format(history=history_text, question=state["question"]))
    logger.info("💬 Response: %s", response.content)
    
    # If thread_id is available, add the response to conversation history
    if state.get("thread_id"):
        await add_to_conversation_history(state["thread_id"], "assistant", response.content)
    
    return {
        **state,
        "response": response.content
    }


# Step 6: Generate answer with retrieved documents
async def generate_answer_with_docs(state: AgentState) -> AgentState:
    """Generate a response using the filtered documents"""
    logger.info("📄 Generating answer with documents")
    
    useful_docs = state.get("useful_documents", [])
    logger.info("📄 Useful docs: %s", useful_docs)
    
    # Get conversation history if thread_id is available
    history_text = ""
    if state.get("thread_id"):
        history = await get_conversation_history(state["thread_id"])
        history_text = await format_conversation_history(history)
    
    if not useful_docs:
        prompt = PromptTemplate(
            template="""You are called felixLandlord, an expert assistant dedicated to helping users with their questions.
            Your tone should be friendly, approachable, and professional, ensuring users feel supported and valued.
            
            IMPORTANT: Since no relevant documents were found, you should:
            - Acknowledge that you don't have the specific information requested

            
            Keep your responses short, relevant, and straight to the point.
            Limit your responses to at most 3 sentences, ensuring concise outputs.
            Be empathetic but clear that you cannot provide information that isn't in your knowledge base.
            Do not make up or invent any information that isn't explicitly stated in the documents provided.
            
            User Conversation history: {history}
            
            User question: {question}
            
            Your response:""",
            input_variables=["history", "question"]
        )
        
        response = await genai_flash_lite_llm.ainvoke(prompt.format(history=history_text, question=state["question"]))
        logger.info("💬 Response: %s", response.content)
    else:
        prompt = PromptTemplate(
            template="""You are called felixLandlord dedicated to helping users with their questions.
            Your tone should be friendly, approachable, and professional, ensuring users feel supported and valued.
            Provide clear, concise, and accurate answers based **solely** on the information from the document,
            focusing **strictly** on the provided document's content and avoiding any unrelated topics.
            When handling sensitive or potentially negative information, express empathy and offer solutions where possible.
            If a question is unclear, kindly ask for clarification before responding.
            Transition smoothly between topics, especially when dealing with follow-up questions.
            Only suggest additional questions if you notice a specific pattern in what the user is asking from the conversation history,
            but not after every question asked by the user.
            Keep your responses short, relevant, and straight to the point, avoiding any ambiguity or unnecessary details as well as unnecessarily long responses.
            Limit your responses to at most 3 sentences, ensuring concise outputs.
            If the information doesn't fully answer the question, acknowledge what you know and what you don't.
            If a question is outside your expertise or not related to the provided document, state firmly, 'I do not have information on the question asked',
            and guide the user to contact a company representative for further assistance.
            If a topic is complex, simplify your explanation while maintaining accuracy.
            Acknowledge user feedback, whether positive or negative, and respond appropriately.
            Always end by politely asking if there's anything else they need help with or suggesting a related topic if contextually appropriate.
            Respond to the user's query as polite as you can, no matter the number of times the user asks the same question without mentioning to the user that you have already answered the question.
            
            User Conversation history: {history}
                        
            User question: {question}
            
            Relevant information:
            {documents}
            
            Your response:""",
            input_variables=["history", "question", "documents"]
        )
        
        documents_text = "\n\n".join(useful_docs)
        logger.info("📄 Documents text: %s", documents_text)
        
        response = await genai_flash_llm.ainvoke(prompt.format(
            history=history_text,
            question=state["question"],
            documents=documents_text
        ))
        logger.info("💬 Response: %s", response.content)
    
    # If thread_id is available, add the response to conversation history
    if state.get("thread_id"):
        await add_to_conversation_history(state["thread_id"], "assistant", response.content)
    
    return {
        **state,
        "response": response.content
    }

# Create the workflow
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("reformulate", reformulate_question)
workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("filter", filter_documents)
workflow.add_node("direct_answer", generate_direct_answer)
workflow.add_node("answer_with_docs", generate_answer_with_docs)

# Add edges
workflow.add_conditional_edges(
    "reformulate",
    decide_retrieval,
    {
        "retrieve": "retrieve",
        "direct_answer": "direct_answer"
    }
)
workflow.add_edge("retrieve", "filter")
workflow.add_edge("filter", "answer_with_docs")
workflow.add_edge("direct_answer", END)
workflow.add_edge("answer_with_docs", END)

# Set entry point
workflow.set_entry_point("reformulate")

# Compile the graph
agent_bot = workflow.compile()


async def process_question(question: str, user_id: str = None) -> Dict[str, Any]:
    """Process a user question through the agent workflow"""
        
    thread_id = user_id if user_id else str(uuid.uuid4())
    
    # Add the question to conversation history
    await add_to_conversation_history(thread_id, "user", question)
    
    config = {"configurable": {"thread_id": thread_id}}
    
    result = await agent_bot.ainvoke(
        {
            "question": question,
            "thread_id": thread_id
        }, 
        config
    )
    logger.info(f"🤖 Result for thread {thread_id}: {result}")
    return {"response": result["response"]}


async def process_question_stream(question: str, user_id: str = None):
    thread_id = user_id if user_id else str(uuid.uuid4())

    await add_to_conversation_history(thread_id, "user", question)

    config = {"configurable": {"thread_id": thread_id}}

    async for chunk in agent_bot.astream(
        {"question": question, "thread_id": thread_id},
        config=config
    ):
        logger.info(f"🔄 Received chunk: {chunk}")  # Log entire chunk

        # Extract response safely
        # response_text = chunk.get("answer_with_docs", {}).get("response", "")
        response_text = chunk.get("direct_answer", {}).get("response") or chunk.get("answer_with_docs", {}).get("response")

        if response_text:
            logger.info(f"✅ Streaming chunk: {response_text}")
            for word in response_text.split():
                yield f"{word} "
                await asyncio.sleep(0.12)
        else:
            # logger.warning("⚠️ Received empty or invalid chunk")
            logger.debug("⚠️ Skipping intermediate chunk") 

    yield " \n"  # Ensure proper termination


# Add conversation history management functions
async def get_conversation_history(thread_id: str, limit: int = 10) -> List[Dict[str, str]]:
    """
    Retrieve conversation history for a thread from MongoDB
    
    Args:
        thread_id: The unique thread identifier
        limit: Maximum number of messages to retrieve
        
    Returns:
        List of conversation messages
    """
    try:
        conversation = await history_collection.find_one({"thread_id": thread_id})
        if not conversation or "messages" not in conversation:
            return []
            
        # Get the most recent messages up to the limit
        messages = conversation["messages"]
        if limit and len(messages) > limit:
            messages = messages[-limit:]
            
        # Return only the role and content
        return [{"role": msg["role"], "content": msg["content"]} for msg in messages]
    except Exception as e:
        logger.error(f"❌ Error retrieving conversation history: {str(e)}")
        return []
    
async def add_to_conversation_history(thread_id: str, role: str, content: str) -> None:
    """
    Add a message to the conversation history in MongoDB
    
    Args:
        thread_id: The unique thread identifier
        role: Message role (user or assistant)
        content: Message content
    """
    try:
        # Check if conversation exists
        existing = await history_collection.find_one({"thread_id": thread_id})
        
        if existing:
            # Add message to existing conversation
            await history_collection.update_one(
                {"thread_id": thread_id},
                {
                    "$push": {
                        "messages": {
                            "role": role,
                            "content": content,
                            "timestamp": datetime.now()
                        }
                    },
                    "$set": {"updated_at": datetime.now()}
                }
            )
        else:
            # Create new conversation with first message
            await history_collection.insert_one({
                "thread_id": thread_id,
                "messages": [{
                    "role": role,
                    "content": content,
                    "timestamp": datetime.now()
                }],
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            })
        
        logger.info(f"📝 Added message to conversation history for thread: {thread_id}")
    except Exception as e:
        logger.error(f"❌ Error adding to conversation history: {str(e)}")

async def format_conversation_history(history: List[Dict[str, str]]) -> str:
    """
    Format conversation history as text for prompts
    
    Args:
        history: List of conversation messages
        
    Returns:
        Formatted conversation history text
    """
    if not history:
        return ""
    
    formatted = "Previous conversation:\n"
    for msg in history:
        prefix = "User: " if msg["role"] == "user" else "AI Aku: "
        formatted += f"{prefix}{msg['content']}\n"
    return formatted


async def cleanup_old_conversations(days: int = 30) -> int:
    """
    Remove conversation history older than specified days
    
    Args:
        days: Number of days to keep
        
    Returns:
        Number of records deleted
    """
    try:
        cutoff_date = datetime.now() - timedelta(days=days)
        result = await history_collection.delete_many({"updated_at": {"$lt": cutoff_date}})
        return result.deleted_count
    except Exception as e:
        logger.error(f"❌ Error cleaning up old conversations: {str(e)}")
        return 0