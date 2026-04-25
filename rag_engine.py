import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_core.prompts import PromptTemplate

# Centralized Imports
from config import CHROMA_DIR, EMBEDDING_MODEL, GROQ_MODEL
from logger import logger

class AdvancedRAG:
    def __init__(self, groq_api_key):
        logger.debug(f"AdvancedRAG: Initializing with model {GROQ_MODEL}")
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name=GROQ_MODEL)
        
        if os.path.exists(CHROMA_DIR):
            logger.debug(f"AdvancedRAG: ChromaDB found at {CHROMA_DIR}")
        else:
            logger.error(f"AdvancedRAG: ChromaDB NOT FOUND at {CHROMA_DIR}")

        self.vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=self.embeddings
        )

    def hyde_retrieval(self, query, domain):
        """Generates a hypothetical answer and uses it for similarity search."""
        logger.debug(f"RAG: Starting HyDE retrieval for domain {domain}...")
        hyde_prompt = PromptTemplate.from_template(
            "Write a short, highly technical paragraph answering this question as it would appear in an Apple internal document: {query}"
        )
        hyde_chain = hyde_prompt | self.llm
        hypothetical_answer = hyde_chain.invoke({"query": query}).content
        
        results = self.vectorstore.similarity_search(
            hypothetical_answer, 
            k=3, 
            filter={"category": domain}
        )
        logger.debug(f"RAG: HyDE retrieval returned {len(results)} chunks.")
        return results

    def get_context(self, query, domain):
        """Advanced retrieval pipeline optimized for token efficiency."""
        logger.info(f"RAG: get_context called for {domain}")
        
        try:
            # 1. Base Retriever with Domain Filter (k=3 for efficiency)
            base_retriever = self.vectorstore.as_retriever(
                search_kwargs={"filter": {"category": domain}, "k": 3}
            )
            
            # 2. Multi-Query Expansion
            logger.debug("RAG: Expanding query via MultiQueryRetriever...")
            mq_retriever = MultiQueryRetriever.from_llm(
                retriever=base_retriever, 
                llm=self.llm
            )
            
            # 3. Direct retrieval from the expanded retriever
            logger.debug("RAG: Fetching relevant chunks...")
            all_docs = mq_retriever.invoke(query)
            
            # 4. Deduplicate and format
            seen = set()
            unique_contents = []
            for d in all_docs:
                if d.page_content not in seen:
                    unique_contents.append(d.page_content)
                    seen.add(d.page_content)
            
            logger.info(f"RAG: Successfully retrieved {len(unique_contents)} context chunks.")
            return "\n---\n".join(unique_contents)
        except Exception as e:
            logger.exception(f"RAG: Error during context retrieval: {str(e)}")
            return ""
