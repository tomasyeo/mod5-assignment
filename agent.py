import operator
from typing import TypedDict, Annotated, Sequence, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

# Centralized Imports
from config import GROQ_MODEL
from logger import logger
from rag_engine import AdvancedRAG
from tools import predict_salary, web_search, run_ddg_search
from prompts import (
    GATEWAY_PROMPT, ROUTER_PROMPT, POLICY_PROMPT, 
    PRODUCT_PROMPT, TECH_PROMPT, SALARY_PROMPT,
    CONTEXTUALIZE_QUERY_PROMPT, EVALUATE_RAG_PROMPT, 
    SYNTHESIZE_SEARCH_PROMPT, WEB_SEARCH_PROMPT
)

# Define the shared state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    feedback: Annotated[List[str], operator.add]
    next_node: str

def create_agent_graph(groq_api_key):
    llm = ChatGroq(temperature=0, model_name=GROQ_MODEL, groq_api_key=groq_api_key)
    rag = AdvancedRAG(groq_api_key)

    # --- HELPER: BUILD CHAT HISTORY STRING ---
    def get_chat_history_str(messages):
        """Converts message sequence into a readable transcript for contextualization."""
        history = ""
        # Skip the very last message as it's the current query
        for msg in messages[:-1]:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            history += f"{role}: {msg.content}\n"
        return history.strip() if history else "No previous conversation."

    # --- NODE: GATEWAY ---
    def gateway(state):
        query = state['messages'][-1].content
        logger.info(f"NODE: gateway | Query: {query[:50]}...")
        
        try:
            # SECURITY FIX: Message Bounding
            chat_history = get_chat_history_str(state["messages"])
            contextualized_gateway_prompt = f"{GATEWAY_PROMPT}\n\nRecent Conversation History:\n{chat_history}"
            
            response = llm.invoke([
                SystemMessage(content=contextualized_gateway_prompt), 
                HumanMessage(content=query)
            ]).content
            logger.debug(f"Gateway Raw Response: {response}")
            
            if "PASS" in response.upper():
                logger.info("Gateway: APPROVED")
                return {"next_node": "router", "feedback": ["Gateway: Approved."]}
            else:
                logger.warning("Gateway: REJECTED")
                # SECURITY FIX: Only use the LLM's explanation if it explicitly includes 'REJECT'.
                # This prevents an attacker from hijacking the gateway output to display malicious text.
                if "REJECT" in response.upper():
                    rejection_msg = response.upper().replace("REJECT", "").strip()
                    # Convert back to a cleaner format if possible, or just use a fallback
                    rejection_msg = "Your query was rejected as off-topic."
                else:
                    rejection_msg = "I can only assist with Apple-related queries or salary predictions."
                
                return {
                    "messages": [AIMessage(content=rejection_msg)], 
                    "next_node": END, 
                    "feedback": ["Gateway: Rejected topic."]
                }
        except Exception as e:
            logger.exception("Gateway Error")
            return {"next_node": END, "messages": [AIMessage(content="Gateway Error: Unable to validate query.")], "feedback": ["Gateway Error: An internal error occurred."]}

    # --- NODE: ROUTER ---
    def router_node(state):
        if state.get("next_node") == END: return state
        
        query = state['messages'][-1].content
        logger.info(f"NODE: router | Query: {query[:50]}...")
        
        try:
            # SECURITY FIX: Message Bounding
            chat_history = get_chat_history_str(state["messages"])
            contextualized_router_prompt = f"{ROUTER_PROMPT}\n\nRecent Conversation History:\n{chat_history}"
            
            category = llm.invoke([
                SystemMessage(content=contextualized_router_prompt),
                HumanMessage(content=query)
            ]).content.lower().strip()
            logger.debug(f"Router Classified Category: {category}")
            
            valid_categories = ['policy', 'product', 'tech', 'salary']
            target = next((c for c in valid_categories if c in category), 'tech')
            
            logger.info(f"Router directed to: {target}_agent")
            return {
                "next_node": f"{target}_agent", 
                "feedback": [f"Router: Classified as {target}."]
            }
        except Exception as e:
            logger.exception("Router Error")
            return {"next_node": "tech_agent", "feedback": ["Router: Error detected, defaulting to Tech agent."]}

    # --- DETERMINISTIC SPECIALIST WRAPPER (RAG Domains Only) ---
    def deterministic_specialist(state, domain, persona_prompt):
        original_query = state['messages'][-1].content
        logger.info(f"NODE: specialist ({domain}) | Processing sequentially...")
        logs = [f"{domain.capitalize()} Agent: Running..."]
        
        try:
            # --- STEP 1: Contextualize Query ---
            chat_history = get_chat_history_str(state["messages"])
            if chat_history != "No previous conversation.":
                logger.debug(f"{domain} Agent: Contextualizing query with history...")
                contextualize_sys_msg = CONTEXTUALIZE_QUERY_PROMPT.format(chat_history=chat_history)
                # SECURITY FIX: Message Bounding
                standalone_query = llm.invoke([
                    SystemMessage(content=contextualize_sys_msg),
                    HumanMessage(content=original_query)
                ]).content.strip()
                logger.info(f"{domain} Agent Standalone Query: {standalone_query}")
                logs.append(f"Contextualized Query: {standalone_query}")
            else:
                # If it's the very first message, the original query is the standalone query
                standalone_query = original_query
            
            # --- STEP 2: Fetch RAG Context (Using Grounded Query) ---
            logger.debug(f"{domain} Agent: Requesting RAG context for standalone query...")
            context = rag.get_context(standalone_query, domain)
            logs.append(f"{domain.capitalize()} Agent: RAG context fetched.")
            
            # --- STEP 3: Evaluate Context ---
            logger.debug(f"{domain} Agent: Evaluating RAG context sufficiency...")
            evaluate_sys_msg = persona_prompt + "\n\n" + EVALUATE_RAG_PROMPT.format(context=context)
            # SECURITY FIX: Message Bounding
            eval_response = llm.invoke([
                SystemMessage(content=evaluate_sys_msg),
                HumanMessage(content=standalone_query)
            ]).content.strip()
            
            # --- STEP 4 & 5: Branch (Answer OR Search+Synthesize) ---
            if eval_response.upper() != "SEARCH":
                # Context was sufficient!
                logger.debug(f"{domain} Agent Final RAW Response (from RAG): {eval_response}")
                logger.info(f"{domain} Agent: RAG was sufficient. Response generated.")
                return {"messages": [AIMessage(content=eval_response)], "feedback": logs + ["RAG context was sufficient."]}
            
            logger.info(f"{domain} Agent: RAG insufficient. Executing direct web search...")
            logs.append(f"Context insufficient. Searching DuckDuckGo for: {standalone_query}")
            
            # Search DuckDuckGo directly using the targeted, contextualized query
            raw_results = run_ddg_search(standalone_query)
            
            logger.debug(f"{domain} Agent: Filtering raw web search results...")
            # SECURITY FIX: Message Bounding
            search_summary = llm.invoke([
                SystemMessage(content=WEB_SEARCH_PROMPT),
                HumanMessage(content=f"Search Results: {raw_results}\n\nUser Query: {standalone_query}")
            ]).content
            
            logger.debug(f"{domain} Agent: Synthesizing final answer...")
            synth_sys_msg = persona_prompt + "\n\n" + SYNTHESIZE_SEARCH_PROMPT
            # SECURITY FIX: Mitigation for Indirect Prompt Injection.
            # We move potentially malicious search results into the HumanMessage context.
            standalone_human_msg = f"Standalone User Query: {standalone_query}\n\nWeb Search Results:\n{search_summary}\n\nKnowledge Base Context:\n{context}"
            
            # SECURITY FIX: Message Bounding
            final_response = llm.invoke([
                SystemMessage(content=synth_sys_msg),
                HumanMessage(content=standalone_human_msg)
            ]).content
            
            logger.debug(f"{domain} Agent Final RAW Response (from Synthesis): {final_response}")
            logger.info(f"{domain} Agent: Response synthesized successfully.")
            return {"messages": [AIMessage(content=final_response)], "feedback": logs + ["Answer synthesized from Web Search."]}
            
        except Exception as e:
            logger.exception(f"{domain} Agent Error")
            return {"messages": [AIMessage(content=f"Error in {domain} specialist: An internal error occurred.")], "feedback": [f"{domain} Error: Diagnostic failure."]}

    # --- SPECIALIST NODES ---
    def policy_agent(state): return deterministic_specialist(state, "policy", POLICY_PROMPT)
    def product_agent(state): return deterministic_specialist(state, "product", PRODUCT_PROMPT)
    def tech_agent(state): return deterministic_specialist(state, "tech", TECH_PROMPT)
    
    # Salary Agent retains React capabilities as it relies on the custom joblib tool
    def salary_agent(state): 
        query = state['messages'][-1].content
        logger.info(f"NODE: specialist (salary) | Processing React...")
        try:
            system_msg = SALARY_PROMPT
            agent = create_react_agent(llm, [predict_salary, web_search], prompt=system_msg)
            response = agent.invoke({"messages": state["messages"]}, config={"recursion_limit": 4})
            return {"messages": [response["messages"][-1]], "feedback": ["Salary Agent: Prediction complete."]}
        except Exception as e:
            logger.exception("Salary Agent Error")
            return {"messages": [AIMessage(content="Error predicting salary: An internal error occurred.")], "feedback": ["Salary Error: Diagnostic failure."]}

    workflow = StateGraph(AgentState)
    
    workflow.add_node("gateway", gateway)
    workflow.add_node("router", router_node)
    workflow.add_node("policy_agent", policy_agent)
    workflow.add_node("product_agent", product_agent)
    workflow.add_node("tech_agent", tech_agent)
    workflow.add_node("salary_agent", salary_agent)

    workflow.set_entry_point("gateway")

    workflow.add_conditional_edges(
        "gateway",
        lambda x: x["next_node"],
        {"router": "router", END: END}
    )

    workflow.add_conditional_edges(
        "router",
        lambda x: x["next_node"],
        {
            "policy_agent": "policy_agent",
            "product_agent": "product_agent",
            "tech_agent": "tech_agent",
            "salary_agent": "salary_agent"
        }
    )

    for node in ["policy_agent", "product_agent", "tech_agent", "salary_agent"]:
        workflow.add_edge(node, END)

    logger.info("Compiling LangGraph workflow with sequential deterministic specialists...")
    return workflow.compile()
