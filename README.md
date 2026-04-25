# Apple Expert Squad: Multi-Agent RAG System

A production-ready AI assistant tailored for Apple product information, support policies, technical troubleshooting, and an integrated Machine Learning salary predictor. 

This project orchestrates a "Squad of Experts" using **LangGraph**, powered by Groq (`llama-3.1-8b-instant`) for high-speed inference, and relies on a local **ChromaDB** vector database for Retrieval-Augmented Generation (RAG).

## 🚀 Key Features & Tech Stack
*   **Frameworks:** FastAPI, LangGraph, LangChain
*   **LLM Engine:** Groq API (`llama-3.1-8b-instant`)
*   **Vector Database:** ChromaDB with `all-MiniLM-L6-v2` local embeddings
*   **Integrated Tools:** DuckDuckGo Web Search, Pre-trained ML Salary Predictor (`joblib`)
*   **Deployment:** Dockerized for Render.com or any containerized hosting service
*   **Frontend:** Custom HTML/JS Chat UI (Dark Theme) served natively via FastAPI

---

## 🧠 LangGraph Architecture & Workflow

The application uses a directed graph to process, route, and fulfill user queries efficiently:

### 1. The Gateway
The entry point of the system. A specialized agent acts as a guardrail, evaluating the user's initial message. If the query is relevant to Apple products, policies, tech support, or salary predictions, it passes the query into the network. If the query is entirely off-topic (e.g., "Give me a recipe for cookies"), it politely rejects it, saving backend processing tokens.

### 2. The Semantic Router
Once a query passes the Gateway, the Router evaluates the semantic intent of the message and classifies it into one of four categories, directing it to the appropriate Specialist Agent:
*   **Policy Specialist:** Handles delivery, returns, and warranties.
*   **Product Specialist:** Handles hardware specs and features.
*   **Tech Specialist:** Handles troubleshooting and diagnostics.
*   **Salary Analyst:** Handles age-based salary predictions.

### 3. The RAG Specialists (Deterministic Pipeline)
The Policy, Product, and Tech agents do not use a standard autonomous loop. Instead, they follow a highly optimized, 5-step deterministic workflow designed for speed and accuracy:
1.  **Contextualize:** The agent reads the conversation history and rewrites the user's message into a standalone query (e.g., resolving "How much is it?" to "How much is the iPhone 16 Pro?").
2.  **Retrieve:** It fetches highly relevant chunks from the ChromaDB vector store using the standalone query.
3.  **Evaluate:** The LLM decides if the retrieved RAG context contains enough information to answer the core question.
4.  **Web Search Fallback:** If the local database lacks the answer (e.g., for unreleased products or future dates), the agent automatically searches DuckDuckGo for live information.
5.  **Synthesize:** The agent combines the RAG data (or Web Search results) into a final, formatted response, clearly differentiating between confirmed facts and unconfirmed rumors.

### 4. The Salary Analyst (ReAct Agent)
The Salary Analyst utilizes a standard ReAct (Reasoning + Acting) loop. It evaluates the user's age, invokes a custom Python tool that loads a pre-trained machine learning model (`salary_model.joblib`), executes a prediction, and returns the estimated salary.

---

## 📂 Project Structure

```text
mod5-assignment/
├── app.py              # FastAPI server, CORS configuration, and UI routing
├── agent.py            # LangGraph logic (Gateway, Router, Specialist workflows)
├── init_db.py          # Ingestion script to build ChromaDB from text files
├── rag_engine.py       # Advanced Retrieval logic (Multi-Query Expansion)
├── tools.py            # Custom Python tools (Salary Predictor, DDGS Web Search)
├── prompts.py          # Agent personas, evaluation logic, and system prompts
├── logger.py           # Custom persistent logging configuration (outputs to app.log)
├── config.py           # Environment variables and path configurations
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container configuration for deployment
├── README.md           # Project documentation
├── static/
│   └── index.html      # Integrated web chat interface
└── data/
    ├── salary_model.joblib # Pre-trained ML model for salary prediction
    ├── apple_data/         # Raw text files for the knowledge base
    └── chroma_db/          # Generated vector database (created by init_db.py)
```

---

## 🔧 Installation & Local Setup

### Prerequisites
*   Python 3.10+ (A dedicated conda or venv environment is recommended)
*   A Groq API Key

### Step-by-Step Setup
1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd mod5-assignment
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set Environment Variables:**
    Create a `.env` file in the root of the `mod5-assignment/` directory:
    ```env
    GROQ_API_KEY=gsk_your_groq_api_key_here
    MEMORY_LIMIT=5
    LOG_LEVEL=INFO
    ```

4.  **Initialize the Vector Database:**
    Run the ingestion script to process the raw `.txt` files in `data/apple_data/` and build the ChromaDB database.
    ```bash
    python init_db.py
    ```

5.  **Start the Application:**
    Run the provided shell script or start Uvicorn directly:
    ```bash
    ./run_app.sh
    # OR
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
    ```
    Open your browser and navigate to `http://127.0.0.1:8000` to interact with the squad.

---

## 🌐 Deployment (Docker / Render.com)

This application is fully containerized and ready for cloud deployment. The included `Dockerfile` handles everything from installing dependencies to pre-computing the vector database during the build phase.

1.  **Create a New Web Service** on Render.com and connect your repository.
2.  **Environment:** Select `Docker`.
3.  **Environment Variables:** Add your `GROQ_API_KEY` and `MEMORY_LIMIT` directly into the Render dashboard. *(Note: Do not commit your local `.env` file).*
4.  **Build Command:** Render will automatically execute the `Dockerfile`. The Docker build process will run `python init_db.py` to bake the ChromaDB database directly into the container image, ensuring lightning-fast startup times.
5.  **Start Command:** The container will automatically execute `uvicorn app:app --host 0.0.0.0 --port 8000` when launched.
