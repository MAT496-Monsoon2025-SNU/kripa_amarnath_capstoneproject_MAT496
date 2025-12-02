import streamlit as st
import os
import yfinance as yf
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, List
import operator
from langchain_core.messages import BaseMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="HedgeGraph India", page_icon="📈", layout="wide")

st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .main {
        background: #0e1117;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 HedgeGraph India: Autonomous Alpha Team")
st.markdown("### *Quant + Fundamental AI Agents working in Parallel*")

# --- 2. SETUP & KEYS ---
# FIXED: We set the default value to your key so it works instantly
# NOTE: Verify the key inside the quotes below is your full key!
default_key = "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"

api_input = st.sidebar.text_input("Groq API Key", value=default_key, type="password")
os.environ["GROQ_API_KEY"] = api_input

llm = None
if os.environ.get("GROQ_API_KEY"):
    # FIXED: Updated model from 3.1 (Decommissioned) to 3.3
    try:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    except Exception as e:
        st.error(f"LLM Init Error: {e}")

# --- 3. DATA & TOOLS ---

@st.cache_resource
def get_vectorstore():
    try:
        embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # Check if DB exists before trying to load
        if os.path.exists("./chroma_db"):
            return Chroma(persist_directory="./chroma_db", embedding_function=embedding)
    except:
        return None
    return None

vectorstore = get_vectorstore()

@tool
def get_trade_signals(ticker: str):
    """Calculates Support, Resistance, Bollinger Bands, and Stop Loss."""
    try:
        stock = yf.Ticker(ticker)
        # Fetch 1 year of data
        hist = stock.history(period="1y")
        
        if hist.empty: 
            return "Error: No data found for ticker."
            
        # Indicators
        hist['Support'] = hist['Low'].rolling(window=60).min()
        hist['Resistance'] = hist['High'].rolling(window=60).max()
        hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
        hist['STD_20'] = hist['Close'].rolling(window=20).std()
        hist['BB_Lower'] = hist['SMA_20'] - (hist['STD_20'] * 2)
        
        # Check if we have enough data for the rolling windows
        if len(hist) < 60:
            return "Error: Not enough historical data for analysis."

        curr = hist.iloc[-1]
        price = curr['Close']
        stop_loss = curr['Support'] * 0.97
        
        # Logic
        signal = "WAIT/HOLD"
        if price <= curr['BB_Lower']: signal = "BUY (Oversold)"
        elif price <= curr['Support']*1.02: signal = "BUY (Support)"
        
        return (
            f"Price: {price:.2f}\n"
            f"Signal Hint: {signal}\n"
            f"Support: {curr['Support']:.2f}\n"
            f"Resistance: {curr['Resistance']:.2f}\n"
            f"Stop Loss: {stop_loss:.2f}\n"
        )
    except Exception as e:
        return f"Tool Error: {str(e)}"

# --- 4. AGENT NODES ---
class AgentState(TypedDict):
    ticker: str
    messages: Annotated[List[BaseMessage], operator.add]
    tech_summary: str
    fund_summary: str
    final_report: str

def quant_node(state):
    raw_data = get_trade_signals.invoke(state['ticker'])
    if llm:
        res = llm.invoke(f"Create a specific Trading Plan (Entry, Stop Loss, Target) based on: {raw_data}")
        return {"tech_summary": res.content}
    return {"tech_summary": raw_data}

def research_node(state):
    ticker = state['ticker']
    context = "No PDF data found. (Please ingest data in notebook 2)"
    
    if vectorstore:
        try:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
            docs = retriever.invoke(f"{ticker} risks future growth")
            if docs:
                context = "\n".join([d.page_content for d in docs])
        except:
            pass

    if llm:
        res = llm.invoke(f"Summarize the fundamental risks for {ticker} based on this annual report excerpt: {context}")
        return {"fund_summary": res.content}
    return {"fund_summary": "Fundamental analysis skipped."}

def manager_node(state):
    return {"final_report": f"### 🚀 FINAL HEDGEGRAPH VERDICT\n\n**📊 TECHNICALS**\n{state['tech_summary']}\n\n---\n**📚 FUNDAMENTALS**\n{state['fund_summary']}"}

# --- 5. BUILD GRAPH ---
workflow = StateGraph(AgentState)
workflow.add_node("quant", quant_node)
workflow.add_node("research", research_node)
workflow.add_node("manager", manager_node)

workflow.add_edge(START, "quant")
workflow.add_edge(START, "research")
workflow.add_edge("quant", "manager")
workflow.add_edge("research", "manager")
workflow.add_edge("manager", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# --- 6. UI INTERACTION ---
ticker = st.sidebar.selectbox("Select Asset", ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"])

# Thread ID for Memory
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "user_123"

if st.button("Initialize Analysis"):
    if not llm:
        st.error("⚠️ API Key Required")
    else:
        with st.status("🚀 HedgeGraph Agents Running...", expanded=True) as status:
            st.write("🔹 Quant Agent: Calculating Bollinger Bands & Support...")
            st.write("🔹 Research Agent: Querying Vector Database for Risks...")
            
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            try:
                result = app.invoke({"ticker": ticker}, config=config)
                status.update(label="Analysis Complete", state="complete", expanded=False)
                st.divider()
                st.markdown(result['final_report'])
            except Exception as e:
                status.update(label="Analysis Failed", state="error", expanded=False)
                st.error(f"Error during execution: {e}")
