from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, Tool
from langchain.chains import RetrievalQA
from langchain_community.tools.tavily_search import TavilySearchResults
import retrieval

# Load environment variables
load_dotenv()

# Get API keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Validate API keys
if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY in .env")
if not TAVILY_API_KEY:
    raise ValueError("Missing TAVILY_API_KEY in .env")

# Initialize LLM model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.5, google_api_key=GOOGLE_API_KEY)

# Ensure retriever is available
if not hasattr(retrieval, "retriever"):
    raise ValueError("Retrieval module does not have 'retriever'. Ensure it is properly defined.")

# Create the RAG-based Q&A system
rag_qa = RetrievalQA.from_chain_type(llm=llm, retriever=retrieval.retriever)

# Initialize web search tool
search_tool = TavilySearchResults(tavily_api_key=TAVILY_API_KEY)

# Define tools
tools = [
    Tool(name="VectorDB Retrieval", func=rag_qa.run, description="Retrieve from internal knowledge base."),
    Tool(name="Web Search", func=search_tool.run, description="Search the web for real-time information."),
]

# Create the agent
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

# Initialize FastAPI
app = FastAPI()

# Define request model
class ChatRequest(BaseModel):
    query: str

# API route for chatbot interaction
@app.post("/chatbot")
def chatbot(request: ChatRequest):
    user_query = request.query
    try:
        response = agent.run(user_query)  # Use the chatbot agent to generate a response
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check route
@app.get("/")
def home():
    return {"message": "Chatbot is working"}

# Run FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


