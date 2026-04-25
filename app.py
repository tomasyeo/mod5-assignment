import os
import sys

# CRITICAL: Inject absolute path to ensure absolute imports (config, agent, logger)
# work correctly regardless of how uvicorn or conda run is executed.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

# Local module imports (now safe due to sys.path injection)
from config import GROQ_API_KEY, MEMORY_LIMIT, STATIC_DIR
from logger import logger
from agent import create_agent_graph

if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY not found in environment variables.")

# Initialize FastAPI
app = FastAPI(title="Apple Multi-Agent RAG System")

# Add CORS Middleware (SECURITY FIX: allow_credentials must be False when using wildcard origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolve static path dynamically
static_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/static", StaticFiles(directory=static_path), name="static")

# Compile the LangGraph agent
logger.info("Initializing LangGraph Multi-Agent System...")
agent = create_agent_graph(GROQ_API_KEY)

class ChatRequest(BaseModel):
    message: str
    history: list = []

@app.get("/")
async def get_index():
    return FileResponse(os.path.join(static_path, "index.html"))

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if not GROQ_API_KEY:
        logger.error("Request failed: GROQ_API_KEY missing.")
        raise HTTPException(status_code=500, detail="An internal server error occurred while processing your request. Please check server configuration.")
    
    try:
        logger.info(f"Incoming message: {request.message[:50]}...")
        messages = []
        limited_history = request.history[-(MEMORY_LIMIT * 2):] if request.history else []
        
        for msg in limited_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        
        messages.append(HumanMessage(content=request.message))
        
        inputs = {"messages": messages, "feedback": []}
        
        # Invoke multi-agent graph with a strict recursion limit
        config = {"recursion_limit": 15}
        result = agent.invoke(inputs, config=config)
        
        # Extract the final answer and diagnostic feedback
        final_answer = result["messages"][-1].content
        feedback = result.get("feedback", [])
        
        logger.info("Response generated successfully.")
        return {
            "response": final_answer,
            "feedback": feedback
        }
    except Exception as e:
        # SECURITY FIX: Data Leakage Prevention
        # Log the actual stack trace server-side ONLY. Do not expose `str(e)` to the client UI.
        logger.exception(f"Error processing chat: {str(e)}")
        raise HTTPException(status_code=500, detail="An internal server error occurred while processing your request. Please try again later.")

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting uvicorn server...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
