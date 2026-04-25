import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Centralized Imports
from config import APPLE_DATA_DIR, CHROMA_DIR, EMBEDDING_MODEL
from logger import logger

def initialize_database():
    if not os.path.exists(APPLE_DATA_DIR):
        logger.error(f"Data directory {APPLE_DATA_DIR} not found.")
        return

    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    logger.info(f"Initializing ChromaDB in {CHROMA_DIR}...")
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

    categories = ["policy", "product", "tech"]
    
    for category in categories:
        dir_path = os.path.join(APPLE_DATA_DIR, category)
        if not os.path.exists(dir_path):
            logger.warning(f"Skipping {category}: Directory not found.")
            continue
            
        logger.info(f"Processing {category} documents...")
        loader = DirectoryLoader(dir_path, glob="**/*.txt", loader_cls=TextLoader)
        docs = loader.load()
        
        for doc in docs:
            doc.metadata["category"] = category
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(docs)
        
        vectorstore.add_documents(splits)
        logger.info(f"Successfully added {len(splits)} chunks from {category}.")

    logger.info("ChromaDB initialization complete.")

if __name__ == "__main__":
    initialize_database()
