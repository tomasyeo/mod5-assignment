import os
import joblib
import pandas as pd
from langchain.tools import tool
from duckduckgo_search import DDGS
from langchain_groq import ChatGroq
from prompts import WEB_SEARCH_PROMPT
from logger import logger

def run_ddg_search(query: str) -> str:
    """Helper to run search directly using the new DDGS library."""
    logger.debug(f"Executing DDG Search (via ddgs) for: {query}")
    try:
        results = DDGS().text(query, max_results=5)
        # DDGS returns an iterator/list of dicts
        res_list = list(results)
        logger.debug(f"DDG Search returned {len(res_list)} results.")
        return str(res_list)
    except Exception as e:
        logger.error(f"DDG Search failed: {str(e)}")
        return f"Search error: {str(e)}"

@tool
def web_search(query: str) -> str:
    """FORBIDDEN: Do NOT use this tool if the provided Knowledge Base Context has any information. RAG is FIRST CLASS priority. Only use if context is empty or totally irrelevant."""
    logger.info(f"Tool called: web_search with query: {query}")
    raw_results = run_ddg_search(query)
    
    logger.debug("Summarizing search results via LLM...")
    from config import GROQ_MODEL
    llm = ChatGroq(temperature=0, model_name=GROQ_MODEL)
    summary = llm.invoke(f"{WEB_SEARCH_PROMPT}\n\nSearch Results: {raw_results}\n\nUser Query: {query}").content
    
    logger.info("web_search tool completed summary.")
    return summary

@tool
def predict_salary(age: int) -> str:
    """Predicts estimated annual salary based on age using a pre-trained ML model."""
    logger.info(f"Tool called: predict_salary for age: {age}")
    try:
        from config import MODEL_PATH
        logger.debug(f"Loading model from: {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        
        df = pd.DataFrame([[float(age)]], columns=['Age'])
        prediction = model.predict(df)[0]
        
        result = f"Based on the age of {age}, the predicted annual salary is ${prediction:,.2f}."
        logger.info(f"Salary prediction result: {result}")
        return result
    except Exception as e:
        logger.error(f"predict_salary tool error: {str(e)}")
        return f"Error predicting salary: {str(e)}"
