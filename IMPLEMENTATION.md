# LangGraph Multi-Agent RAG System: Implementation Details

## Objective
This document outlines the architectural design and technical implementation of the Apple Expert Squad application. The system is a high-performance, deterministic multi-agent network designed to maximize the reliability and accuracy of small-parameter language models (specifically `llama-3.1-8b-instant`) without relying on brittle autonomous loops.

---

## 🏛️ AI Architecture Paradigm

This project completely Abandons the "God Prompt" anti-pattern in favor of a highly structured, decentralized intelligence network. It achieves this through three foundational AI architecture patterns:

### 1. The Multi-Agent System (MAS) / Actor Pattern
Instead of a single monolithic LLM prompt attempting to manage policies, products, and technical support simultaneously, the system is broken down into specialized agents. 
Each agent (Policy, Product, Tech, Salary) functions as an independent "Actor." They receive a message, process it strictly according to their internal logic (their specific RAG domain or ML tool), and pass the result back to the central state. This enforces a strict **Separation of Concerns**, preventing context dilution and persona drift.

### 2. Finite State Machine (FSM) via LangGraph
The core orchestration of the AI workflow is modeled as a Directed Acyclic Graph (DAG) acting as a Finite State Machine.
The system transitions through distinct, predictable states: `gateway` -> `router` -> `specialist_node` -> `END`. The flow of execution (the Conditional Edges) is determined dynamically by the output of the Semantic Router LLM, explicitly defining the exact path a user's query takes through the system based on its intent.

### 3. The Chain of Responsibility (Deterministic RAG Pipeline)
Autonomous ReAct (Reasoning + Acting) loops are notoriously unstable on 8B parameter models, often resulting in infinite tool-calling loops. To solve this, the RAG specialists implement a strict, linear **Chain of Responsibility**:
*Contextualizer -> Retriever -> Evaluator -> Web Scout Fallback -> Synthesizer.*
This design guarantees that local RAG data is always prioritized before any external web search is authorized, physically preventing infinite loop hallucinations and drastically simplifying the cognitive load on the LLM.

---

## 💻 Software Code Patterns (Python)

The Python implementation is not a procedural script; it is a highly structured, Object-Oriented architecture that heavily utilizes structural and creational design patterns to ensure modularity, DRY (Don't Repeat Yourself) compliance, and testability.

### 1. The Factory Pattern
When instantiating complex objects, the logic is encapsulated within Factories.
*   **The Agent Factory (`create_agent_graph`):** This massive function doesn't just execute code; it *manufactures* the entire application. It instantiates the LLM, the RAG engine, defines the internal routing nodes, wires up the entire LangGraph `StateGraph`, and returns the fully compiled application object to the API layer.
*   **The Logger Factory (`get_logger`):** Centralizes logging instantiation. Instead of repeating handler setups across files, modules simply call `logger = get_logger("my-module")` to receive a pre-configured stream and file handler.

### 2. The Wrapper / Decorator Pattern (DRY Principle)
This pattern is used to enforce DRY principles across the agent network.
*   **The `deterministic_specialist` Function:** The Policy, Product, and Tech agents share an identical 80-line workflow. Instead of writing that logic three times, it is encapsulated in a single master function. The actual LangGraph nodes (`policy_agent`, `product_agent`) are just tiny wrappers that inject their specific domain constraints (e.g., `POLICY_PROMPT`) into the master function.

### 3. The Facade Pattern
This pattern hides massive underlying complexity behind simple interfaces.
*   **The API Facade (`app.py`):** The FastAPI `@app.post("/chat")` endpoint is a Facade for the entire AI system. The frontend sends a simple JSON object, and the endpoint handles API key validation, memory truncation, LangGraph invocation, and error sanitization entirely behind the scenes.
*   **The Retrieval Facade (`AdvancedRAG`):** In `rag_engine.py`, the `get_context()` method hides the complexity of embedding initialization, ChromaDB metadata filtering, and Multi-Query LLM expansion behind a single, clean method call.

### 4. Dependency Injection
This is a core principle for writing testable, modular Python.
*   Internal modules (`agent.py`, `rag_engine.py`) **do not** read the `GROQ_API_KEY` from `os.environ` directly. Instead, `app.py` reads the configuration at the highest level and explicitly injects the dependency downstream. This isolates configuration logic and enables easy unit testing with mock API keys.

---

## Core Agent Network (LangGraph)

The application uses a directed `StateGraph` to process and fulfill user queries efficiently. The shared state (`AgentState`) tracks the conversation history and accumulates diagnostic feedback from each node.

### 1. The Gateway Node
The initial entry point. A specialized agent acts as a strict relevance filter. It evaluates the user's message against the supported domains. If a query is off-topic, it politely rejects it, halting the graph and saving downstream processing tokens.

### 2. The Semantic Router Node
Once a query passes the Gateway, the Router evaluates the semantic intent of the message and classifies it into one of four categories, directing it to the appropriate Specialist Node.

---

## The Deterministic Pipeline (Replacing ReAct)

To solve Context Amnesia and endless loop hallucinations common in autonomous ReAct agents powered by small models, the RAG specialists (Policy, Product, Tech) utilize a highly optimized, linear 5-step workflow:

1.  **Contextualize Query:** A fast LLM call reads the conversation history and rewrites the user's latest message into a standalone query. This resolves pronouns (e.g., "how much is it?" -> "What is the price of the Apple iPhone 16?").
2.  **Grounded RAG Retrieval:** Fetches highly relevant chunks from the ChromaDB vector store using the explicitly grounded `standalone_query`.
3.  **Evaluate & Decide (Lenient Inference):** The LLM evaluates the retrieved RAG context. The `EVALUATE_RAG_PROMPT` is designed with "Lenient Inference," mandating that the model infer general policies onto specific products.
4.  **Targeted Web Search Fallback:** If RAG is deemed insufficient, the `web_search` tool (via DuckDuckGo) is called. The Web Scout sub-agent is instructed to act as a strict relevancy filter, silently discarding irrelevant search results (e.g., makeup catalogs) to prevent "Garbage In, Garbage Out" (GIGO) hallucinations.
5.  **Synthesize (Rumor Separation):** The final LLM call synthesizes the data. The `SYNTHESIZE_SEARCH_PROMPT` explicitly differentiates between confirmed facts and unconfirmed rumors or concepts.

---

## Data Ingestion & RAG Optimization (`init_db.py` & `rag_engine.py`)

*   **Domain Filtering:** During ingestion (`init_db.py`), chunks are tagged with specific `category` metadata (e.g., `category: policy`). The RAG engine uses a ChromaDB metadata filter (`filter={"category": domain}`) to ensure strict domain isolation during retrieval.
*   **Multi-Query Expansion:** The `AdvancedRAG` class utilizes LangChain's `MultiQueryRetriever` to generate multiple variations of the user's question, capturing a broader semantic net.
*   **Token Optimization:** Retrieval is explicitly limited to `k=3` chunks per query variant to prevent "Lost in the Middle" amnesia and token exhaustion on 8B parameter models.

---

## System Observability

*   **Diagnostic Streaming:** The LangGraph State is designed with a `feedback` reducer (`Annotated[List[str], operator.add]`). Each node appends its "thought process" to this list, which is streamed to the UI's debug console.
*   **Persistent Logging:** `logger.py` handles persistent server-side event tracking, writing detailed execution traces to `app.log`.

---

## Deployment Strategy (Render.com)

The application is fully containerized. The included `Dockerfile` optimizes startup latency by executing `init_db.py` during the build phase. This bakes the pre-computed ChromaDB vector database directly into the Docker image. When the container launches on Render.com via `uvicorn app:app`, the vector store is immediately available in memory, resulting in a 0-second initialization delay for the RAG engine.

---

## 📦 Project Dependencies

The following libraries drive the core architecture of the application (versions locked via Docker build):

| Library | Version | Description |
| :--- | :--- | :--- |
| `langchain` | 1.2.15 | Core framework for building LLM applications and constructing prompt pipelines. |
| `langchain-community` | 0.4.1 | Community integrations, specifically used for DocumentLoaders and local SentenceTransformers. |
| `langchain-groq` | 1.1.2 | Official integration library for the Groq API (`ChatGroq`), providing high-speed inference. |
| `langchain-huggingface`| 1.2.2 | Integration library for running HuggingFace models locally (used for embeddings). |
| `langgraph` | 1.1.9 | State machine orchestration framework used to define the multi-agent DAG. |
| `fastapi` | 0.136.1 | High-performance asynchronous web framework used to build the REST API Facade. |
| `uvicorn` | 0.46.0 | ASGI web server implementation used to run the FastAPI application. |
| `chromadb` | 1.5.8 | Open-source vector database used for persistent, metadata-filtered RAG storage. |
| `langchain-chroma` | 1.1.0 | Official LangChain wrapper enabling ChromaDB to function as a unified Retriever. |
| `joblib` | 1.5.3 | Serialization library used to load the pre-trained Machine Learning salary prediction model. |
| `pandas` | 3.0.2 | Data manipulation library used to format input features for the ML model. |
| `sentence-transformers`| 5.4.1 | Backend library for running `all-MiniLM-L6-v2` text embedding models locally. |
| `pydantic` | 2.13.3 | Data validation and settings management using Python type annotations. |
| `python-dotenv` | 1.2.2 | Utility to securely load environment variables from a `.env` file. |
| `duckduckgo-search` | 8.1.1 | Python library providing programmatic access to the DuckDuckGo search engine for web fallback. |
| `Jinja2` | 3.1.6 | Templating engine (installed natively as a standard FastAPI/Starlette dependency). |
| `aiofiles` | 25.1.0 | Asynchronous file I/O support required by FastAPI for serving `index.html`. |