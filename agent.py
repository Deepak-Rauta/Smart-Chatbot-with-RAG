from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, Tool
from langchain.chains import RetrievalQA
from langchain_community.tools.tavily_search import TavilySearchResults
import retrieval
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
tvly_key = os.getenv("TAVILY_API_KEY")

# Initialize the LLm model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.5, google_api_key=api_key)

# Create the RAG Based Q&A
rag_qa = RetrievalQA.from_chain_type(llm=llm, retriever=retrieval.retriever)

# Initialize the SerpAPI for web search
search_tool = TavilySearchResults(tevly_key=tvly_key)

tools = [
    Tool(name="VectorDB Retrieval", func=rag_qa.run, description="Retrieve from internal knowledge base."),
    Tool(name="Web Search", func=search_tool.run, description="Search the web for real-time information."),
]

# Create an agent 
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

# Function to interact with the chatbot
def chatbot_response(query):
    return agent.run(query)

if __name__ == "__main__":
    while True:
        user_input = input("Ask a questions")
        if user_input.lower() == "exit":
            break
        print(chatbot_response(user_input))


# from langchain_community.tools import tavily_search
# print(dir(tavily_search))
